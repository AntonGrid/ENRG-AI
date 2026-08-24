"""Tests for the intent detection (agent/core)."""

from agent.core.intent import detect


def test_detect_analyze_by_default():
    result = detect("mint")
    assert result == {"intent": "analyze", "target": "mint"}


def test_detect_empty_query():
    result = detect("   ")
    assert result["intent"] == "analyze"
    assert result["target"] == ""


def test_detect_explicit_commands():
    assert detect("impact hello")["intent"] == "impact"
    assert detect("impact hello")["target"] == "hello"
    assert detect("find merkle")["intent"] == "find"
    assert detect("explain attestation")["intent"] == "explain"
    assert detect("fix bug")["intent"] == "fix"


def test_detect_unknown_word_falls_back_to_analyze():
    result = detect("blabla hello")
    assert result["intent"] == "analyze"
    assert result["target"] == "blabla hello"
