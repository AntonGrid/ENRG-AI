from agent.config import PROJECTS
from agent.knowledge.build import build

for project, root in PROJECTS.items():

    print(f"\n=== {project.upper()} ===")

    nodes = build(project, root)

    print("Files:", len(nodes))

    symbols = sum(len(n.symbols) for n in nodes)
    imports = sum(len(n.imports) for n in nodes)

    print("Symbols:", symbols)
    print("Imports:", imports)

    print("\nTop 10 files:\n")

    for node in nodes[:10]:
        print(node.path)
