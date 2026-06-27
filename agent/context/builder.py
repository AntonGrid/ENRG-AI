from agent.config import PROJECTS
from agent.index.search import search
from agent.reader.files import read_file


def build_context(plan, limit: int = 5):

    context = []
    added = set()

    #
    # 1. Сначала главный файл
    #

    if plan.get("matches"):

        main = plan["matches"][0]

        try:

            text = read_file(
                PROJECTS[main["project"]],
                main["path"]
            )

            context.append({

                "project": main["project"],
                "path": main["path"],
                "content": text[:6000]

            })

            added.add(
                (main["project"], main["path"])
            )

        except Exception:
            pass

    #
    # 2. Затем остальные найденные файлы
    #

    for file in plan.get("matches", [])[1:]:

        if len(context) >= limit:
            break

        key = (
            file["project"],
            file["path"]
        )

        if key in added:
            continue

        try:

            text = read_file(
                PROJECTS[file["project"]],
                file["path"]
            )

            context.append({

                "project": file["project"],
                "path": file["path"],
                "content": text[:6000]

            })

            added.add(key)

        except Exception:
            continue

    #
    # 3. Если ничего не нашли —
    #    используем старый механизм
    #

    if not context:

        files = search(plan["query"])

        for file in files[:limit]:

            try:

                text = read_file(
                    PROJECTS[file["project"]],
                    file["path"]
                )

                context.append({

                    "project": file["project"],
                    "path": file["path"],
                    "content": text[:6000]

                })

            except Exception:
                continue

    return context
