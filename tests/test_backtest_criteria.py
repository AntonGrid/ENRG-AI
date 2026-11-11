"""The declaration a report is judged against: written before the run, hashed afterwards.

These tests are about the integrity of the claim, not about the strategy: that the default
declaration quotes the same numbers the code actually uses, that the hash is a function of
the payload and not of how it was typed, that a gate edited after the run is caught, and
that the honest half of the document — what a passing run would still not have shown — is
inside the hash rather than beside it.
"""
from __future__ import annotations

import json

import pytest

from agent.backtest.criteria import (
    ACCEPTANCE,
    DECLARED_SEED,
    DOES_NOT_SHOW,
    VERSION,
    Criteria,
    CriteriaError,
    declared,
    verify,
)
from agent.backtest.power import DECLARED_ALPHA, DECLARED_POWER
from agent.backtest.walkforward import GATE


def test_the_default_declaration_quotes_the_numbers_the_code_uses():
    criteria = declared()
    assert criteria.alpha == pytest.approx(DECLARED_ALPHA)
    assert criteria.power == pytest.approx(DECLARED_POWER)
    assert criteria.gate == {key: float(value) for key, value in GATE.items()}
    assert criteria.seed == DECLARED_SEED
    assert criteria.edge_bps == pytest.approx(float(GATE["edge_bps"]))
    assert criteria.version == VERSION


def test_the_hash_reads_the_payload_and_not_the_typing():
    forward = Criteria(gate={"edge_bps": 30.0, "t_stat": 2.0})
    backward = Criteria(gate={"t_stat": 2.0, "edge_bps": 30.0})
    assert forward.fingerprint() == backward.fingerprint()
    assert forward.fingerprint() == forward.fingerprint()
    assert len(forward.fingerprint()) == 64


def test_a_different_declaration_is_a_different_hash():
    assert Criteria().fingerprint() != Criteria(alpha=0.01).fingerprint()
    assert Criteria().fingerprint() != Criteria(power=0.8).fingerprint()
    assert Criteria().fingerprint() != Criteria(seed=1).fingerprint()
    assert Criteria().fingerprint() != Criteria(gate=dict(GATE, edge_bps=10.0)).fingerprint()


def test_a_gate_edited_after_the_run_does_not_verify():
    document = declared().document({"trades": 19, "edge_bps": -12.0})
    assert verify(document) == ""
    document["criteria"]["gate"]["edge_bps"] = 5.0
    reason = verify(document)
    assert "does not match the declared" in reason
    assert "changed after the run" in reason


def test_an_edited_disclaimer_does_not_verify_either():
    document = declared().document()
    document["criteria"]["does_not_show"] = []
    assert "does not match" in verify(document)


def test_a_document_that_declared_nothing_is_refused_with_a_sentence():
    assert verify({}) == "no criteria block: the run was not read against anything declared"
    assert "not an object" in verify({"criteria": [], "criteria_sha256": "00"})
    assert "upheld" not in verify({})
    bare = declared().document()
    del bare["criteria_sha256"]
    assert "nothing proves they were not edited" in verify(bare)


def test_the_document_is_json_without_infinities():
    document = declared().document(
        {"trades": 1, "min_edge_bps": None, "note": "a single trade resolves nothing"}
    )
    written = json.dumps(document, allow_nan=False, sort_keys=True)
    assert "Infinity" not in written
    assert json.loads(written)["criteria_sha256"] == declared().fingerprint()


def test_write_round_trips_through_disk(tmp_path):
    path = declared().write(tmp_path / "nested" / "criteria.json", {"pairs": ["btcusdt"]})
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert verify(payload) == ""
    assert payload["run"]["pairs"] == ["btcusdt"]
    assert path.read_text(encoding="utf-8").endswith("}\n")


def test_the_honest_half_travels_with_the_claim():
    criteria = declared()
    assert criteria.does_not_show == list(DOES_NOT_SHOW)
    assert len(criteria.does_not_show) >= 5
    joined = " ".join(criteria.does_not_show)
    assert "null.py" in joined
    assert "profit" in joined
    assert "overlap" in joined


def test_the_acceptance_checks_name_the_resolution_and_the_power():
    joined = " ".join(ACCEPTANCE)
    assert "min_edge_bps" in joined
    assert "0.90 +/- 0.02" in joined
    assert "alpha * [0.5, 2.0]" in joined
    assert any("lookahead" in item for item in ACCEPTANCE)


def test_an_impossible_risk_is_refused_before_it_reaches_a_report():
    with pytest.raises(CriteriaError, match="not a risk"):
        Criteria(alpha=0.0)
    with pytest.raises(CriteriaError, match="not a chance"):
        Criteria(power=1.0)
    with pytest.raises(CriteriaError, match="not a chance"):
        Criteria(power=0.0)
