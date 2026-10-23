import json
from transformers import AutoTokenizer

MODEL_NAME_V1 = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer_v1 = AutoTokenizer.from_pretrained(MODEL_NAME_V1)
MODEL_NAME_V2 = "bert-base-cased"
tokenizer_v2 = AutoTokenizer.from_pretrained(MODEL_NAME_V2)

samples = [
    "kot",
    "nieprzewidywalność",
    "Large language model",
    "Duży model językowy",
    "print('Hello!')",
    "LLM-y 🤖 kosztują 0.01 USD",
]

output_data = {}

for name, tokenizer in [(MODEL_NAME_V1, tokenizer_v1), (MODEL_NAME_V2, tokenizer_v2)]:
    print("\n", name)
    output_data[name] = []
    for text in samples:
        ids = tokenizer.encode(text, add_special_tokens=False)
        pieces = [tokenizer.decode([token_id]) for token_id in ids]
        print(f"\nTEXT:   {text!r}")
        print(f"IDS:    {ids}")
        print(f"PIECES: {pieces}")
        print(f"COUNT:  {len(ids)}")
        output_data[name].append({
            "text": text,
            "ids": ids,
            "pieces": pieces,
            "count": len(ids)
        })
    with open("lab01a.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)
