from agent.core.commands import analyze


def analyze_command(arg):

    if not arg:
        print("Использование: analyze <тема>")
        return

    print("\nАнализ...\n")

    print(analyze(arg))