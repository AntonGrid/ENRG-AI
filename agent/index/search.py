from agent.index.builder import build_index
from agent.index.rank import score
from agent.config import PROJECTS

INDEX = {}

for project, path in PROJECTS.items():
    print(f"Indexing {project}...")
    INDEX[project] = build_index(path)


def search(query):

    result = []

    for project, files in INDEX.items():

        for file in files:

            s = score(file, query.lower())

            if s > 0:

                file["project"] = project
                file["score"] = s

                result.append(file)

    result.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return result
