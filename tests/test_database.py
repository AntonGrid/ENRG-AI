"""Tests for the knowledge database (agent/db)."""

from agent.db.database import connect
from agent.db.indexer import rebuild


def test_connect_creates_schema(tmp_path):
    db = connect(str(tmp_path / "test.db"))
    tables = {
        row[0]
        for row in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    assert {"files", "symbols", "imports"} <= tables
    db.close()


def test_rebuild_indexes_given_projects(sample_project, tmp_path):
    db_path = str(tmp_path / "knowledge.db")
    rebuild(paths={"demo": str(sample_project)}, db_path=db_path)

    db = connect(db_path)
    files = db.execute(
        "SELECT project, path, name FROM files ORDER BY path"
    ).fetchall()
    db.close()

    assert len(files) == 2
    assert files[0][0] == "demo"          # project
    assert files[0][1] == "mod.py"        # path
    assert files[0][2] == "mod.py"        # name
