from agent.knowledge.build import build
from agent.config import PROJECTS


def architecture():

    for project, root in PROJECTS.items():

        print(f"\n{project.upper()}")

        nodes = build(project, root)

        groups = {}

        for node in nodes:

            folder = node.path.split("/")[0]

            groups.setdefault(folder, []).append(node)

        for folder in sorted(groups):

            print(f"\n📁 {folder}")

            for node in groups[folder][:10]:

                print("   ├──", node.name)

            if len(groups[folder]) > 10:
                print("   └── ...")
