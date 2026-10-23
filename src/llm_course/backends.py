import os
import time
import csv
import json
import statistics
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

@dataclass
class Generation:
    text: str
    input_tokens: int | None
    output_tokens: int | None
    latency_s: float
    model: str
    mode: str


# Nowe parametry
class LLMBackend(Protocol):
    def generate(
        self, system: str, user: str, max_tokens: int = 160, *,
        temperature: float | None = None, top_k: int | None = None,
        top_p: float = 1.0, seed: int | None = None,
    ) -> Generation:
        ...


# Weryfikacja samplingu do testów
def validate_sampling(
    max_tokens: int, temperature: float | None, top_k: int | None,
    top_p: float, seed: int | None,
) -> None:
    """temperature=None oznacza greedy; liczba większa od zera próbkowanie."""
    if max_tokens < 1:
        raise ValueError("max_tokens musi być dodatnie")
    if not 0 < top_p <= 1:
        raise ValueError("top_p musi należeć do przedziału (0, 1]")
    if top_k is not None and top_k < 1:
        raise ValueError("top_k musi być dodatnie albo None")
    if temperature is None:
        if top_k is not None or top_p != 1.0 or seed is not None:
            raise ValueError("W trybie greedy pomiń top_k i seed oraz użyj top_p=1.0")
    elif temperature <= 0:
        raise ValueError("Temperatura próbkowania musi być większa od zera; None oznacza greedy")


class ApiBackend:
    def __init__(self) -> None:
        from google import genai
        # OpenAI: from openai import OpenAI

        self.model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.supports_seed = True
        # OpenAI: self.model = os.environ["OPENAI_MODEL"]
        # OpenAI: self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        # OpenAI: self.supports_seed = False

    # Przekazanie dodatkowych parametrów
    def generate(
        self, system: str, user: str, max_tokens: int = 160, *,
        temperature: float | None = None, top_k: int | None = None,
        top_p: float = 1.0, seed: int | None = None,
    ) -> Generation:
        from google.genai import types

        validate_sampling(max_tokens, temperature, top_k, top_p, seed)
        config_kwargs = {
            "system_instruction": system,
            "max_output_tokens": max_tokens,
            "temperature": 0.0 if temperature is None else temperature,
            "top_p": top_p,
        }
        if top_k is not None:
            config_kwargs["top_k"] = top_k
        if seed is not None:
            config_kwargs["seed"] = seed

        started = time.perf_counter()
        response = self.client.models.generate_content(
            model=self.model,
            contents=user,
            config=types.GenerateContentConfig(**config_kwargs),
        )
        # OpenAI (Responses API nie przyjmuje top_k ani seed):
        # if top_k is not None or seed is not None:
        #     raise ValueError("OpenAI Responses API nie obsługuje top_k ani seed")
        # response = self.client.responses.create(
        #     model=self.model,
        #     instructions=system,
        #     input=user,
        #     max_output_tokens=max_tokens,
        #     temperature=0.0 if temperature is None else temperature,
        #     top_p=top_p,
        # )
        latency_s = time.perf_counter() - started

        generated_text = response.text
        # OpenAI: generated_text = response.output_text
        if not generated_text:
            raise RuntimeError("API nie zwróciło tekstu odpowiedzi")

        # OpenAI:
        # if response.status != "completed":
        #     raise RuntimeError(
        #         f"Generowanie nie zostało zakończone: "
        #         f"{response.status}, {response.incomplete_details}"
        #     )

        usage = response.usage_metadata
        # OpenAI: usage = response.usage
        return Generation(
            text=generated_text,
            input_tokens=getattr(usage, "prompt_token_count", None),
            output_tokens=getattr(usage, "candidates_token_count", None),
            # OpenAI: input_tokens=getattr(usage, "input_tokens", None),
            # OpenAI: output_tokens=getattr(usage, "output_tokens", None),
            latency_s=latency_s,
            model=self.model,
            mode="api",
        )


class LocalBackend:
    def __init__(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.model_name = os.getenv("LOCAL_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype="auto",
        )
        self.model.to(self.device).eval()

    # Przekazanie dodatkowych parametrów
    def generate(
        self, system: str, user: str, max_tokens: int = 160, *,
        temperature: float | None = None, top_k: int | None = None,
        top_p: float = 1.0, seed: int | None = None,
    ) -> Generation:
        validate_sampling(max_tokens, temperature, top_k, top_p, seed)
        started = time.perf_counter()
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)

        generation_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": temperature is not None,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

        # Przekazanie dodatkowych parametrów
        if temperature is not None:
            generation_kwargs.update(
                temperature=temperature,
                top_p=top_p,
                # Transformers domyślnie stosuje top_k=50 przy próbkowaniu.
                top_k=0 if top_k is None else top_k,
            )
            if seed is not None:
                self.torch.manual_seed(seed)

        with self.torch.inference_mode():
            output = self.model.generate(**inputs, **generation_kwargs)
        input_tokens = int(inputs["input_ids"].shape[1])
        new_ids = output[0, input_tokens:]
        generated_text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        latency_s = time.perf_counter() - started

        return Generation(
            text=generated_text,
            input_tokens=input_tokens,
            output_tokens=int(new_ids.shape[0]),
            latency_s=latency_s,
            model=self.model_name,
            mode="local",
        )


def make_backend() -> LLMBackend:
    mode = os.getenv("LLM_MODE", "local")
    if mode == "local":
        return LocalBackend()
    if mode == "api":
        return ApiBackend()
    raise ValueError("LLM_MODE musi mieć wartość 'local' albo 'api'")


# Jeżeli ktoś nie dodał na poprzednich laboratoriach
def append_event(path: str, result: Generation, status: str = "ok") -> None:
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": status,
        **{k: v for k, v in asdict(result).items() if k != "text"},
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as file:
        file.write(json.dumps(event, ensure_ascii=False) + "\n")


# Do testów
def normalized(text: str) -> str:
    return " ".join(text.lower().split())


# Do samodzielnego napisania 
def run_experiment(backend: LLMBackend) -> None:
    pass
