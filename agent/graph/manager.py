from .builder import GraphBuilder


_graph = None


def graph():

    global _graph

    if _graph is None:

        _graph = GraphBuilder().build(".")

    return _graph