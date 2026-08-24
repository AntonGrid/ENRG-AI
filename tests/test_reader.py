"""Tests for the file reader (agent/reader)."""

from agent.reader.files import read_file


def test_read_file_returns_content(sample_project):
    content = read_file(str(sample_project), "mod.py")
    assert "def hello()" in content


def test_read_file_raises_on_missing_file(sample_project):
    import pytest

    with pytest.raises(FileNotFoundError):
        read_file(str(sample_project), "missing.py")
