from agent.cli.banner import show
from agent.cli.dispatcher import dispatch


def run():

    show()

    while True:

        try:

            command = input("ENRG AI > ").strip()

            if command.lower() in ("exit", "quit"):
                break

            dispatch(command)

        except KeyboardInterrupt:
            print()
            break