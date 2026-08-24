"""Tests for the call graph (agent/graph)."""

from agent.graph.builder import GraphBuilder
from agent.graph.query import GraphQuery


def test_graph_builds_nodes(sample_project):
    graph = GraphBuilder().build(str(sample_project))
    assert "hello" in graph
    assert "world" in graph
    assert "Thing" in graph
    assert graph["Thing"].type == "class"


def test_graph_records_calls(sample_project):
    graph = GraphBuilder().build(str(sample_project))
    assert "hello" in graph["world"].calls
    assert "world" in graph["hello"].called_by


def test_impact_query(sample_project):
    graph = GraphBuilder().build(str(sample_project))
    result = GraphQuery(graph).impact("hello")
    assert result is not None
    assert result["name"] == "hello"
    assert result["calls"] == []
    assert result["called_by"] == ["world"]


def test_impact_unknown_symbol_returns_none(sample_project):
    graph = GraphBuilder().build(str(sample_project))
    assert GraphQuery(graph).impact("missing") is None
