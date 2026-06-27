from agent.cli.registry import COMMANDS


def dispatch(command: str):

    command = command.strip()

    if not command:
        return

    parts = command.split(maxsplit=1)

    name = parts[0].lower()

    arg = ""

    if len(parts) > 1:
        arg = parts[1]

    func = COMMANDS.get(name)

    if func is None:
        print("Неизвестная команда.")
        return

    func(arg)