from dotenv import load_dotenv
from llm_course.backends import make_backend, append_event

def main():
    load_dotenv()
    backend = make_backend()
    result = backend.generate(
        system="Było kiedyś w pewnym mieście wielkie poruszenie.", 
        user="Wystawiano niesłychanie piękne przedstawienie."
    )
    append_event("logs/model_wywolanie.jsonl", result)
    print("log powstał w folderze logs.")

if __name__ == "__main__":
    main()