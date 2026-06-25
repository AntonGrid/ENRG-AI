from agent.core.context import build
from agent.core.planner import plan


def run():

    print("=" * 60)
    print("ENRG AI CTO")
    print("=" * 60)

    while True:

        user = input("\nYou: ")

        if user.lower() in ["exit", "quit"]:
            break

        print("\nAnalyzing project...\n")

        context = build()

        print("Thinking...\n")

        answer = plan(context, user)

        print(answer)