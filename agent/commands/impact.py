from agent.graph.manager import graph
from agent.graph.query import GraphQuery


def impact_command(arg):

    if not arg:
        print("Использование: impact <имя>")
        return

    g = graph()

    result = GraphQuery(g).impact(arg)

    if result is None:
        print("Ничего не найдено.")
        return

    print("\nIMPACT ANALYSIS\n")

    print(f"Имя : {result['name']}")
    print(f"Файл: {result['file']}")

    print("\nВызывает:")

    if result["calls"]:
        for call in result["calls"]:
            print(f"  • {call}")
    else:
        print("  — ничего")

    print("\nИспользуется в:")

    if result["called_by"]:
        for call in result["called_by"]:
            print(f"  • {call}")
    else:
        print("  — нигде")