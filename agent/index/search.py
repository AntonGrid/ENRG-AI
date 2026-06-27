from agent.index.builder import build_index
from agent.index.rank import score
from agent.config import PROJECTS

INDEX = {}

for project, path in PROJECTS.items():
    print(f"Indexing {project}...")
    INDEX[project] = build_index(path)


def search(query):

    query = query.lower().strip()

    result = []

    for project, files in INDEX.items():

        for file in files:

            s = score(file, query)

            matched_symbol = None

            symbols = file.get("symbols", [])

            for symbol in symbols:

                symbol_lower = symbol.lower()

                if query == symbol_lower:
                    matched_symbol = symbol
                    s += 10000
                    break

                if query in symbol_lower:
                    matched_symbol = symbol
                    s += 5000

            if s <= 0:
                continue

            item = file.copy()
            item["project"] = project
            item["score"] = s
            item["matched_symbol"] = matched_symbol

            result.append(item)

    result.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return result
