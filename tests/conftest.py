import sys
from pathlib import Path

import pytest

# Make `agent.*` importable regardless of the CWD pytest is launched from.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def sample_project(tmp_path):
    """A tiny two-file Python project used by index/graph/knowledge tests."""

    (tmp_path / "mod.py").write_text(
        "def hello():\n"
        "    return 1\n"
        "\n"
        "def world():\n"
        "    return hello()\n",
        encoding="utf-8",
    )

    (tmp_path / "other.py").write_text(
        "import os\n"
        "\n"
        "class Thing:\n"
        "    pass\n",
        encoding="utf-8",
    )

    return tmp_path
