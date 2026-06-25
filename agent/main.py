from agent.index.search import search

while True:

    q = input("\nSearch: ")

    if q == "exit":
        break

    result = search(q)

    print()

    for item in result[:15]:

        print(
            f"[{item['score']:3}] "
            f"{item['project']:8} "
            f"{item['path']}"
        )

        if item["symbols"]:
            print("     ", ", ".join(item["symbols"]))