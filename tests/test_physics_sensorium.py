"""The sensorium: parsers without a network, organs without a medium.

Two things are asserted, and the second is the interesting one.

**Parsers** are pure functions of a payload, so every feed is tested here with a
handful of dicts — no network, no cache, no fixture files. A feed that renames a
column upstream fails in this file rather than in a notebook at midnight.

**Dreams** are the organism's own output, and two of them must be *hard*: a replay
at λ near 1 is nearly the world it replays, and a phase-randomised surrogate keeps
the world's exact marginal distribution and its periodogram. These tests assert
that this is true of the generators, because the demonstration's honesty rests on
the surrogate being unbeatable by local statistics — if a later change made the
surrogate easier, the claim "only memory can refuse it" would quietly become false.
"""
import numpy as np
import pytest

from agent.physics import sensorium
from agent.physics.sensorium import (
    DOMAIN_BEHAVIOR,
    DOMAIN_FINANCE,
    DOMAIN_PHYSICS,
    DREAM_RED,
    DREAM_REPLAY,
    DREAM_RHYTHM,
    DREAM_SURROGATE,
    Perception,
    SensorError,
    Sensorium,
    build_organ,
    dream,
    hear,
    parse_attention,
    parse_commits,
    parse_dataset,
    parse_finance,
    parse_solar,
    parse_space_weather,
    phase_randomise,
    shape,
)

# --------------------------------------------------------------------------- #
# Parsers: the shape of each payload, as the upstream services actually send it
# --------------------------------------------------------------------------- #


def test_finance_parser_takes_closes_and_refuses_a_short_series():
    payload = {"close": [100.0, 101.5, float("nan")] + [102.0 + i for i in range(70)]}
    values = parse_finance(payload)
    assert values.size == 72
    assert np.all(np.isfinite(values))
    with pytest.raises(SensorError):
        parse_finance({"close": [1.0, 2.0]})


def test_space_weather_prefers_kp_index_and_skips_broken_rows():
    rows = [
        {"kp_index": 2, "estimated_kp": 1.67},
        {"kp_index": None, "estimated_kp": 3.0},
    ]
    rows += [{"kp_index": 1 + (i % 3)} for i in range(80)]
    values = parse_space_weather(rows)
    assert values.size == 81
    assert set(np.unique(values)) <= {1.0, 2.0, 3.0}
    with pytest.raises(SensorError):
        parse_space_weather([{"nothing": 1}])


def test_solar_parser_drops_the_minus_one_months():
    rows = [{"ssn": 12.0}, {"ssn": -1.0}, {"ssn": 40.5}] + [{"ssn": 100.0}] * 80
    values = parse_solar(rows)
    assert values.size == 82
    assert np.all(values >= 0.0)


def test_attention_parser_counts_daily_views():
    items = [{"views": 1000 + index} for index in range(70)]
    values = parse_attention({"items": items})
    assert values.size == 70
    assert values[-1] == 1069


def test_commit_parser_makes_gaps_and_refuses_a_short_log():
    from datetime import datetime, timedelta

    base = datetime(2026, 1, 1, 3, 0, 0)
    stamps = [
        {"commit": {"author": {"date": (base + timedelta(hours=6 * index))
                               .strftime("%Y-%m-%dT%H:%M:%SZ")}}}
        for index in range(280)  # seventy days of a six-hourly cadence
    ]
    gaps = parse_commits(stamps, mode="gaps")
    counts = parse_commits(stamps, mode="counts")
    assert gaps.size == 279
    assert np.allclose(gaps, 6.0)
    assert counts.size == 70  # four commits a day
    with pytest.raises(SensorError):
        parse_commits(stamps[:3])


def test_dataset_parser_reads_a_bundled_recording_and_refuses_a_stranger():
    values = parse_dataset("iris", limit=512)
    assert values.size == 512
    with pytest.raises(SensorError):
        parse_dataset("not_a_dataset")


# --------------------------------------------------------------------------- #
# Perception: provenance is glued to the bytes
# --------------------------------------------------------------------------- #


def test_two_streams_with_the_same_numbers_share_a_fingerprint():
    values = np.linspace(0.0, 1.0, sensorium.MIN_SAMPLES)
    first = Perception(
        values=values, origin=sensorium.WORLD, source="x", domain=DOMAIN_FINANCE
    )
    second = Perception(
        values=values.copy(), origin=sensorium.SELF, source="y", domain=DOMAIN_PHYSICS
    )
    assert first.fingerprint == second.fingerprint
    changed = Perception(
        values=values * 2.0, origin=sensorium.WORLD, source="x", domain=DOMAIN_FINANCE
    )
    assert changed.fingerprint != first.fingerprint


def test_a_perception_refuses_a_bad_origin_a_short_stream_or_a_nan():
    values = np.zeros(sensorium.MIN_SAMPLES)
    with pytest.raises(SensorError):
        Perception(values=values, origin="neither", source="x", domain=DOMAIN_FINANCE)
    with pytest.raises(SensorError):
        Perception(
            values=values[:8], origin=sensorium.WORLD, source="x",
            domain=DOMAIN_FINANCE,
        )
    bad = values.copy()
    bad[0] = np.nan
    with pytest.raises(SensorError):
        Perception(values=bad, origin=sensorium.WORLD, source="x", domain=DOMAIN_FINANCE)


# --------------------------------------------------------------------------- #
# Dreams: the organism's own voice, with its provenance already on it
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("kind", [DREAM_RED, DREAM_RHYTHM])
def test_an_unanchored_dream_is_the_self_and_is_reproducible(kind):
    first = dream(kind, length=256, seed=3)
    second = dream(kind, length=256, seed=3)
    other = dream(kind, length=256, seed=4)
    assert first.origin == sensorium.SELF
    assert first.domain == sensorium.DOMAIN_SELF
    assert first.fingerprint == second.fingerprint
    assert first.fingerprint != other.fingerprint


def test_an_anchored_dream_without_an_anchor_is_refused():
    with pytest.raises(SensorError):
        dream(DREAM_REPLAY, length=128)
    with pytest.raises(SensorError):
        dream(DREAM_SURROGATE, length=128)


def test_a_surrogate_keeps_the_marginal_and_the_periodogram_of_its_anchor():
    rng = np.random.default_rng(0)
    anchor = np.cumsum(rng.normal(size=512))  # a random walk: heavy, red, non-Gaussian
    surrogate = phase_randomise(anchor, seed=1)
    assert np.allclose(np.sort(anchor), np.sort(surrogate))  # identical histogram
    power_anchor = np.abs(np.fft.rfft(anchor - anchor.mean()))
    power_surrogate = np.abs(np.fft.rfft(surrogate - surrogate.mean()))
    ratio = power_surrogate / np.maximum(power_anchor, 1e-9)
    assert 0.7 < float(np.median(ratio)) < 1.4  # the rank map perturbs it a little
    assert not np.allclose(anchor, surrogate)  # and it is not the anchor itself


def test_a_replay_at_lambda_near_one_is_nearly_the_memory_it_replays():
    memory = np.sin(np.linspace(0.0, 20.0, 256)) * 10.0
    faithful = sensorium.dream_replay(memory, lam=1.0, epsilon=0.0, seed=1)
    loose = sensorium.dream_replay(memory, lam=0.05, epsilon=0.0, seed=1)
    assert np.corrcoef(faithful, memory)[0, 1] > 0.99
    assert np.corrcoef(loose, memory)[0, 1] < 0.5


# --------------------------------------------------------------------------- #
# Shape: four numbers that survive a change of units
# --------------------------------------------------------------------------- #


def test_white_noise_has_no_volatility_memory_and_a_random_walk_has_a_red_slope():
    rng = np.random.default_rng(7)
    white = shape(rng.normal(size=512))
    walk = shape(np.cumsum(rng.normal(size=512)))
    assert abs(white.volatility) < 0.2
    assert white.slope > -0.6
    assert walk.slope < -1.0
    assert white.as_array().shape == (len(sensorium.SIGNATURE_FIELDS),)


def test_shape_is_scale_free():
    rng = np.random.default_rng(11)
    values = np.cumsum(rng.normal(size=256)) + 1000.0
    first = shape(values).as_array()
    second = shape(values * 7.5 - 250.0).as_array()
    assert np.allclose(first, second, atol=1e-9)


# --------------------------------------------------------------------------- #
# Organs: eight channels, a bias the core can profile out, and a measured sigma
# --------------------------------------------------------------------------- #


def test_the_cochlea_reports_eight_finite_channels_for_a_flat_medium():
    rng = np.random.default_rng(3)
    organ = build_organ("cochlea")
    row = organ.read(rng.normal(size=128))
    assert row.shape == (sensorium.BANDS,)
    assert np.all(np.isfinite(row))


def test_a_tone_inside_a_band_pulls_a_band_near_its_own_centre():
    organ = build_organ("cochlea")
    time = np.arange(512, dtype=float)
    edges = organ.edges()
    centre = float(np.sqrt(edges[4] * edges[5]))
    tone = np.sin(2.0 * np.pi * centre * time + 0.3)
    row = organ.read(tone)
    assert int(np.argmax(np.abs(row))) in (4, 5)


def test_a_step_moves_the_two_ends_of_the_segments_organ_in_opposite_directions():
    organ = build_organ("segments")
    values = np.concatenate([np.zeros(64), np.ones(64)])
    row = organ.read(values)
    assert row.size == sensorium.BANDS
    assert abs(row[0]) > 0.9 and abs(row[-1]) > 0.9
    assert row[0] * row[-1] < 0.0


def test_an_unknown_organ_is_refused():
    with pytest.raises(SensorError):
        build_organ("nose")


# --------------------------------------------------------------------------- #
# The sensorium: fixation, coverage, and what it refuses to pretend
# --------------------------------------------------------------------------- #


def test_fixation_hands_the_core_its_own_observation_with_this_streams_sigma():
    rng = np.random.default_rng(5)
    stream = Perception(
        values=rng.normal(size=400),
        origin=sensorium.WORLD,
        source="test:noise",
        domain=DOMAIN_PHYSICS,
    )
    surface = Sensorium()
    episode = surface.fixate(stream)
    assert len(episode) >= 2
    assert episode.fingerprint == stream.fingerprint
    sigmas = {reading.sigma for reading in episode.readings}
    assert len(sigmas) == 1  # one background per stream, not per fixation
    assert sigmas.pop() > 0.0
    assert episode.readings[0].x.size == sensorium.BANDS
    assert episode.readings[0].observation.source == "test:noise"


def test_coverage_counts_the_domains_and_names_the_missing_ones():
    surface = Sensorium()
    rng = np.random.default_rng(1)

    def stream(domain, source):
        return Perception(
            values=rng.normal(size=128), origin=sensorium.WORLD, source=source,
            domain=domain,
        )

    finance = stream(DOMAIN_FINANCE, "f")
    physics = stream(DOMAIN_PHYSICS, "p")
    behaviour = stream(DOMAIN_BEHAVIOR, "b")
    assert surface.coverage([finance, physics])["ok"] is False
    assert surface.coverage([finance, physics])["missing"] == [DOMAIN_BEHAVIOR]
    assert surface.coverage([finance, physics, behaviour])["ok"] is True


def test_a_feed_that_fails_is_a_warning_not_a_simulation():
    surface = Sensorium()
    heard = surface.hear_all(["dataset:iris", "file:/nonexistent/stream.csv"])
    assert len(heard) == 1
    assert heard[0].source == "recorded:iris"
    assert any("no such file" in warning for warning in surface.warnings)


def test_an_unknown_feed_spec_is_refused():
    with pytest.raises(SensorError):
        hear("telepathy:someone")


def test_the_manifest_carries_every_stream_and_its_coverage():
    surface = Sensorium()
    stream = surface.hear("dataset:iris", limit=256)
    manifest = surface.manifest([stream])
    assert manifest["organ"] == "cochlea"
    assert manifest["coverage"]["domains"] == ["recorded"]
    assert manifest["streams"][0]["source"] == "recorded:iris"
