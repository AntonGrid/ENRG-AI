from pathlib import Path


def read_file(root, relative_path):

    path = Path(root) / relative_path

    return path.read_text(
        encoding="utf-8",
        errors="ignore"
    )
