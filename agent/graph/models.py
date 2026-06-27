from dataclasses import dataclass, field


@dataclass
class Node:

    name: str

    type: str

    file: str

    calls: set = field(default_factory=set)

    called_by: set = field(default_factory=set)
