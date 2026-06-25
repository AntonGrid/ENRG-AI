from agent.index.builder import build_index
from agent.config import PROJECTS

INDEX = {}

for project, path in PROJECTS.items():
    print(f"Indexing {project}...")
    INDEX[project] = build_index(path)


def search(query: str):

    query = query.lower()

    scored = []

    for project, files in INDEX.items():

        for file in files:

            score = 0

            # имя файла
            if query in file["name"].lower():
                score += 100

            # путь
            if query in file["path"].lower():
                score += 50

            # функции / классы
            for symbol in file["symbols"]:
                if query == symbol.lower():
                    score += 200
                elif query in symbol.lower():
                    score += 120

            # содержимое
            if query in file["content"]:
                score += 10

            if score > 0:

                scored.append({
                    "project": project,
                    "path": file["path"],
                    "score": score,
                    "symbols": file["symbols"]
                })

    scored.sort(key=lambda x: x["score"], reverse=True)

    return scored