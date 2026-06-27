class GraphQuery:

    def __init__(self, graph):

        self.graph = graph

    def impact(self, name):

        if name not in self.graph:
            return None

        node = self.graph[name]

        return {
            "name": node.name,
            "file": node.file,
            "calls": sorted(node.calls),
            "called_by": sorted(node.called_by),
        }
