from pathlib import Path
import re

TEXT_EXTENSIONS = {
    ".py", ".rs", ".js", ".ts",
    ".html", ".css", ".md",
    ".toml", ".json", ".yaml", ".yml"
}

IGNORE = {
    ".git",
    "__pycache__",
    "node_modules",
    "target",
    ".venv",
    "dist",
    "build",
}

PATTERNS = [
    r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)",          # Python
    r"class\s+([a-zA-Z_][a-zA-Z0-9_]*)",        # Python class
    r"pub\s+fn\s+([a-zA-Z_][a-zA-Z0-9_]*)",     # Rust pub
    r"fn\s+([a-zA-Z_][a-zA-Z0-9_]*)",           # Rust
    r"function\s+([a-zA-Z_][a-zA-Z0-9_]*)",     # JS
    r"const\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=",    # JS
]


def extract_symbols(text):

    symbols = set()

    for pattern in PATTERNS:
        symbols.update(re.findall(pattern, text))

    return sorted(symbols)


def build_index(root):

    root = Path(root)

    files = []

    for path in root.rglob("*"):

        if not path.is_file():
            continue

        if any(part in IGNORE for part in path.parts):
            continue

        if path.suffix.lower() not in TEXT_EXTENSIONS:
            continue

        try:
            content = path.read_text(
                encoding="utf-8",
                errors="ignore"
            )
        except Exception:
            content = ""

        files.append({
            "name": path.name,
            "path": str(path.relative_to(root)),
            "content": content.lower(),
            "symbols": extract_symbols(content),
        })

    return files
