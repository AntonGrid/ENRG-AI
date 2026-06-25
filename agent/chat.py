from agent.llm import ask
from agent.project import analyze


def run():

    print("=" * 60)
    print("ENRG AI CTO")
    print("=" * 60)

    while True:

        user = input("\nYou: ")

        if user.lower() in ["exit", "quit"]:
            break

        context = analyze()

        prompt = f"""
Project:

{context}

User request:

{user}

You are CTO of ENRG.

Always:

- analyze first
- never invent files
- explain your plan
- modify only existing architecture
"""

        answer = ask(prompt)

        print("\nAI:\n")
        print(answer)