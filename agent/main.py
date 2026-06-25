from agent.config import PROJECTS
from agent.index.search import search
from agent.reader.files import read_file
from agent.llm import ask


print("=" * 60)
print("ENRG AI")
print("=" * 60)


while True:

    command = input("\nENRG AI > ").strip()

    if command in ("exit", "quit"):
        break

    if command.startswith("analyze "):

        query = command.replace("analyze ", "")

        files = search(query)

        if not files:
            print("Nothing found.")
            continue

        file = files[0]

        text = read_file(
            PROJECTS[file["project"]],
            file["path"]
        )

        prompt = f"""
You are senior software architect.

Analyze this file.

File:
{file["path"]}

Code:

{text[:12000]}
"""

        print("\nThinking...\n")

        print(ask(prompt))

    else:
        print("Unknown command.")
