import ast
from pathlib import Path

from .models import Node


IGNORE = {
    ".venv",
    "__pycache__",
    "node_modules",
    ".git",
    "build",
    "dist",
    "target",
}


class Visitor(ast.NodeVisitor):

    def __init__(self, graph, current):

        self.graph = graph

        self.current = current

    def visit_Call(self, node):

        if isinstance(node.func, ast.Name):

            name = node.func.id

            if name in self.graph:

                self.graph[self.current].calls.add(name)

                self.graph[name].called_by.add(self.current)

        self.generic_visit(node)


class GraphBuilder:

    def __init__(self):

        self.nodes = {}

    def add(self, name, typ, file):

        if name not in self.nodes:

            self.nodes[name] = Node(

                name=name,

                type=typ,

                file=file

            )

    def build(self, root):

        root = Path(root)

        files = []

        #
        # PASS 1
        #

        for file in root.rglob("*.py"):

            if any(x in file.parts for x in IGNORE):
                continue

            try:

                tree = ast.parse(
                    file.read_text(
                        encoding="utf-8"
                    )
                )

            except Exception:

                continue

            files.append((tree, file))

            rel = str(file.relative_to(root))

            for node in ast.walk(tree):

                if isinstance(node, ast.FunctionDef):

                    self.add(
                        node.name,
                        "function",
                        rel
                    )

                elif isinstance(node, ast.ClassDef):

                    self.add(
                        node.name,
                        "class",
                        rel
                    )

        #
        # PASS 2
        #

        for tree, file in files:

            for node in ast.walk(tree):

                if isinstance(node, ast.FunctionDef):

                    Visitor(
                        self.nodes,
                        node.name
                    ).visit(node)

        return self.nodes
