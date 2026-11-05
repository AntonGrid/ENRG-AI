"""The genome: the core is reachable, fingerprinted, and refuses to move.

The whole claim of the autopoietic layer rests on ``entrain_infer/theorem.py`` and
``entrain_infer/detector.py`` not changing while the organism grows organs around
them. These tests are that claim, executed: the digests are recomputed from the
files, :func:`genome.verify` is asked to accept its own answer and to refuse a
tampered one, and the adapter is checked to hand the core its own object rather
than a look-alike.

If someone edits either file, ``test_verify_accepts_its_own_fingerprint`` still
passes (the digest is recomputed) — but the digests recorded by any earlier
report change, which is exactly the alarm the mechanism exists to raise.
"""
import numpy as np
import pytest

from agent.physics import genome


def test_the_core_is_reachable_and_is_the_one_in_the_workspace():
    root = genome.core_root()
    assert (root / "theorem.py").is_file()
    assert (root / "detector.py").is_file()
    assert genome.core().detector.__file__.endswith("detector.py")
    assert genome.core().theorem.__file__.endswith("theorem.py")


def test_the_genome_is_exactly_two_sha256_digests():
    digest = genome.fingerprint()
    assert set(digest) == set(genome.CORE_FILES)
    assert all(len(value) == 64 for value in digest.values())
    assert all(set(value) <= set("0123456789abcdef") for value in digest.values())


def test_verify_accepts_its_own_fingerprint():
    genome.verify(genome.fingerprint())  # must not raise


def test_verify_refuses_a_changed_detector():
    tampered = dict(genome.fingerprint())
    tampered["detector.py"] = "0" * 64
    with pytest.raises(genome.GenomeFrozenError) as error:
        genome.verify(tampered)
    assert "detector.py" in str(error.value)


def test_verify_refuses_a_file_that_is_not_part_of_the_genome():
    with pytest.raises(genome.GenomeFrozenError):
        genome.verify({"report.py": "0" * 64})


def test_the_observation_adapter_builds_the_core_object():
    x = np.array([0.1, -0.2, 0.3, 0.05])
    observation = genome.observation(x, 0.05, hours=128.0, source="test:stream")
    assert observation.source == "test:stream"
    assert observation.sigma == pytest.approx(0.05)
    assert observation.hours == pytest.approx(128.0)
    assert np.allclose(observation.x, x)


def test_describe_names_the_two_files_and_the_package():
    described = genome.describe()
    assert described["package"] == "entrain_infer"
    assert set(described["fingerprint"]) == set(genome.CORE_FILES)
    assert described["modules"] == list(genome.CORE_MODULES)


def test_the_environment_variable_is_how_an_operator_points_at_a_core(tmp_path, monkeypatch):
    fake = tmp_path / "entrain_infer"
    fake.mkdir()
    (fake / "detector.py").write_text("# not the real one\n", encoding="utf-8")
    monkeypatch.setenv(genome.ENV_PATH, str(tmp_path))
    genome.core_root.cache_clear() if hasattr(genome.core_root, "cache_clear") else None
    assert genome.core_root() == fake
    monkeypatch.delenv(genome.ENV_PATH)

    monkeypatch.setenv(genome.ENV_PATH, str(tmp_path / "nothing_here"))
    with pytest.raises(genome.GenomeError):
        genome.core_root()
