import sqlite3


def connect(path: str = "knowledge.db"):
    """Open (or create) the knowledge SQLite database.

    ``path`` is overridable so tests can use a temporary file instead of
    polluting the repository root with ``knowledge.db``.
    """

    db = sqlite3.connect(path)

    db.execute("""
    CREATE TABLE IF NOT EXISTS files (

        id INTEGER PRIMARY KEY,

        project TEXT,

        path TEXT,

        name TEXT,

        extension TEXT,

        content TEXT
    )
    """)

    db.execute("""
    CREATE TABLE IF NOT EXISTS symbols (

        id INTEGER PRIMARY KEY,

        file_id INTEGER,

        symbol TEXT
    )
    """)

    db.execute("""
    CREATE TABLE IF NOT EXISTS imports (

        id INTEGER PRIMARY KEY,

        file_id INTEGER,

        import_name TEXT
    )
    """)

    db.commit()

    return db
