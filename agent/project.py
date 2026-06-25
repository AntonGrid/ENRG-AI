from pathlib import Path
from agent.config import PROJECTS, IMPORTANT_FILES


def read(path: Path):
    try:
        return path.read_text(encoding="utf-8")[:4000]
    except:
        return ""


def analyze():

    result = []

    for project_name, project_path in PROJECTS.items():

        project = Path(project_path)

        result.append(f"\n===== {project_name.upper()} =====\n")

        for file in IMPORTANT_FILES:

            found = list(project.rglob(file))

            for f in found[:1]:
                result.append(f"\n### FILE: {f.relative_to(project)}\n")
                result.append(read(f))

    return "\n".join(result)