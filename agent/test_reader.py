from agent.config import PROJECTS
from agent.index.search import search
from agent.reader.files import read_file

result = search("mint")

if not result:
    print("Nothing found")
    exit()

file = result[0]

print("=" * 60)
print(file["path"])
print("=" * 60)

text = read_file(
    PROJECTS[file["project"]],
    file["path"]
)

print(text[:3000])
