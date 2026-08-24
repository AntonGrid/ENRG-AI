"""Tests for knowledge extraction (agent/knowledge)."""

from agent.knowledge.build import build


def test_build_creates_one_node_per_file(sample_project):
    nodes = build("demo", str(sample_project))
    assert len(nodes) == 2


def test_build_extracts_symbols_and_imports(sample_project):
    nodes = build("demo", str(sample_project))

    mod = next(n for n in nodes if n.path == "mod.py")
    assert mod.project == "demo"
    assert "hello" in mod.symbols
    assert "world" in mod.symbols

    other = next(n for n in nodes if n.path == "other.py")
    assert "Thing" in other.symbols
    assert "import os" in other.imports
    assert other.extension == ".py"
