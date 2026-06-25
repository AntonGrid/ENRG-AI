from pathlib import Path
from agent.knowledge.model import Node
from agent.index.builder import extract_symbols


IGNORE = {
    ".git",
    ".venv",
    "__pycache__",
    "node_modules",
    "target",
    "dist",
    "build",
}


TEXT = {
    ".py",
    ".rs",
    ".js",
    ".ts",
    ".html",
    ".css",
    ".md",
    ".toml",
    ".json",
    ".yaml",
    ".yml",
}


def build(project, root):

    nodes = []

    root = Path(root)

    for file in root.rglob("*"):

        if not file.is_file():
            continue

        if any(part in IGNORE for part in file.parts):
            continue

        if file.suffix.lower() not in TEXT:
            continue

        try:
            text = file.read_text(
                encoding="utf-8",
                errors="ignore"
            )
        except:
            continue

        imports = []

        for line in text.splitlines():

            s = line.strip()

            if s.startswith("use "):
                imports.append(s)

            elif s.startswith("import "):
                imports.append(s)

            elif s.startswith("from "):
                imports.append(s)

            elif s.startswith("const "):
                imports.append(s)

            elif s.startswith("require("):
                imports.append(s)

        nodes.append(
            Node(
                project=project,
                path=str(file.relative_to(root)),
                name=file.name,
                extension=file.suffix,
                symbols=extract_symbols(text),
                imports=imports,
                content=text,
            )
        )

    return nodes
