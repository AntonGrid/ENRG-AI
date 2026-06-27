from agent.db.database import connect
from agent.knowledge.build import build
from agent.config import PROJECTS


def rebuild():

    db = connect()

    db.execute("DELETE FROM files")
    db.execute("DELETE FROM symbols")
    db.execute("DELETE FROM imports")

    for project, root in PROJECTS.items():

        nodes = build(project, root)

        for node in nodes:

            cur = db.execute(
                """
                INSERT INTO files
                (project, path, name, extension, content)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    node.project,
                    node.path,
                    node.name,
                    node.extension,
                    node.content,
                ),
            )

            file_id = cur.lastrowid

            for symbol in node.symbols:
                db.execute(
                    "INSERT INTO symbols (file_id, symbol) VALUES (?, ?)",
                    (file_id, symbol),
                )

            for imp in node.imports:
                db.execute(
                    "INSERT INTO imports (file_id, import_name) VALUES (?, ?)",
                    (file_id, imp),
                )

    db.commit()
    db.close()

    print("Knowledge Base rebuilt.")
