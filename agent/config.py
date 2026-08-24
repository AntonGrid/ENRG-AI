OLLAMA_HOST = "http://127.0.0.1:11434"
MODEL = "qwen2.5-coder:7b"

PROJECTS = {
    "agent": "agent",
    "enrg": "projects/ENRG",
    "enrg-landing": "projects/enrg-landing",
    "axis-core": "projects/Axis-core",
}

SYSTEM_PROMPT = """
You are ENRG AI Engineer.

Your mission:

- Understand ENRG AI itself.
- Understand ENRG Protocol.
- Understand enrg-landing.
- Never invent files.
- Analyze before editing.
- Prefer safe modifications.
- Explain your reasoning briefly.
"""

IMPORTANT_FILES = [
    "README.md",
    "Anchor.toml",
    "Cargo.toml",
    "package.json",
    "server.js",
    "script.js",
    "index.html",
]
