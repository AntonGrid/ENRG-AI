from dataclasses import dataclass


@dataclass
class Node:

    project: str

    path: str

    name: str

    extension: str

    symbols: list

    imports: list

    content: str
