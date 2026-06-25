from ollama import chat

MODEL = "qwen2.5-coder:7b"


def ask(prompt: str, system: str = ""):

    messages = []

    if system:
        messages.append(
            {
                "role": "system",
                "content": system,
            }
        )

    messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    response = chat(
        model=MODEL,
        messages=messages,
    )

    return response["message"]["content"]