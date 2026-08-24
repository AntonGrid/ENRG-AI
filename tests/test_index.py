"""Tests for the symbol index (agent/index)."""

from agent.index.builder import extract_symbols, build_index
from agent.index.rank import score


def test_extract_symbols_python():
    symbols = extract_symbols("def foo():\n    pass\n\nclass Bar:\n    pass\n")
    assert "foo" in symbols
    assert "Bar" in symbols


def test_extract_symbols_rust_js():
    symbols = extract_symbols(
        "pub fn verify() {}\n"
        "function mint() {}\n"
        "const MAX = 10;\n"
    )
    assert "verify" in symbols
    assert "mint" in symbols
    assert "MAX" in symbols


def test_extract_symbols_returns_sorted_unique():
    symbols = extract_symbols("def a():\n    pass\n\ndef a():\n    pass\n")
    assert symbols == ["a"]


def test_build_index_indexes_text_files(sample_project):
    files = build_index(str(sample_project))
    paths = {f["path"] for f in files}
    assert "mod.py" in paths
    assert "other.py" in paths


def test_build_index_extracts_symbols(sample_project):
    files = build_index(str(sample_project))
    mod = next(f for f in files if f["path"] == "mod.py")
    assert "hello" in mod["symbols"]
    assert "world" in mod["symbols"]


def test_score_ranks_rust_program_higher():
    low = {
        "path": "readme.md",
        "name": "readme.md",
        "content": "mint",
        "symbols": [],
    }
    high = {
        "path": "programs/enrg-mvp/src/instructions/mint.rs",
        "name": "mint.rs",
        "content": "pub fn mint_energy() {}".lower(),
        "symbols": ["mint_energy"],
    }
    assert score(high, "mint") > score(low, "mint")


def test_score_is_zero_when_no_match():
    f = {
        "path": "a/b.py",
        "name": "b.py",
        "content": "def nothing(): pass",
        "symbols": [],
    }
    assert score(f, "merkle") == 0
