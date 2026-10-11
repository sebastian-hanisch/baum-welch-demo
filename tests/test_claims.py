"""Jede Zahl aus dem README, nachgerechnet über dieselben Auswertungsfunktionen wie die App (feste Trainings- und Testfahrten).

Plattformrobust: Bänder statt exakter Werte (CI installiert neueste numpy/scipy)."""

import numpy as np
import pytest

import bwl_constants as C
import bwl_evaluation as ev


@pytest.fixture(scope="module")
def default_analysis():
    return ev.analyse(ev.Settings())


@pytest.fixture(scope="module")
def noise_rows():
    return {r["sigma"]: r for r in ev.sweep_noise(ev.Settings())}


@pytest.fixture(scope="module")
def start_rows():
    return ev.sweep_starts(ev.Settings())


@pytest.fixture(scope="module")
def data_rows():
    return {r["n"]: r for r in ev.sweep_data(ev.Settings())}


def test_learning_raises_the_likelihood_but_lowers_the_hit_rate(default_analysis):
    a = default_analysis
    assert a.ll_learned > a.ll_hand + 40 and a.ll_hand > a.ll_truth + 30         # gelernt > Hand > wahre Schrittverteilung
    assert a.ll_learned == pytest.approx(-826, abs=15) and a.ll_hand == pytest.approx(-879, abs=15) and a.ll_truth == pytest.approx(-932, abs=20)
    assert a.hit["hand"][0] == pytest.approx(0.75, abs=0.06)
    assert a.hit["learned"][0] == pytest.approx(0.60, abs=0.07)
    assert a.hit["hand"][0] > a.hit["learned"][0] + 0.07
    assert a.hit["truth"][0] == pytest.approx(0.72, abs=0.06)


def test_true_step_distribution_is_a_single_class(default_analysis):
    assert default_analysis.w_truth[1] == 1.0 and default_analysis.w_truth[[0, 2, 3]].max() < 0.01


def test_at_low_noise_learning_works_and_gains_nothing(noise_rows):
    r = noise_rows[5.0]
    assert r["hand"]["hit"] == pytest.approx(0.99, abs=0.02)
    for name, v in r["starts"].items():
        assert v["w"][1] == 1.0 and v["w"][3] < 0.02, name              # richtige Struktur gelernt
        assert abs(v["hit"] - r["hand"]["hit"]) < 0.03, name


def test_at_high_noise_learned_weights_are_worse_than_the_hand_values(noise_rows):
    r30, r40 = noise_rows[30.0], noise_rows[40.0]
    assert r30["hand"]["hit"] == pytest.approx(0.75, abs=0.06)
    assert r30["starts"]["gleich"]["hit"] == pytest.approx(0.61, abs=0.07)
    assert r30["starts"]["gleich"]["ll"] > r30["hand"]["ll"] + 25
    assert r40["starts"]["gleich"]["hit"] < r40["hand"]["hit"] - 0.07


def test_different_starts_end_in_different_places_with_similar_likelihood(start_rows):
    lls = [r["ll"] for r in start_rows]
    hits = [r["hit"] for r in start_rows]
    assert max(lls) - min(lls) < 15
    assert max(hits) - min(hits) > 0.07
    ws = np.array([r["w"] for r in start_rows])
    assert np.max(np.abs(ws - ws[0])) > 0.4                              # die Gewichte unterscheiden sich stark


def test_more_data_does_not_change_the_result(data_rows):
    hits = [data_rows[n]["hit"] for n in C.DATA_SWEEP]
    assert max(hits) - min(hits) < 0.04
    assert np.max(np.abs(data_rows[5]["w"] - data_rows[40]["w"])) < 0.15


def test_without_driving_direction_learning_still_does_not_beat_the_hand_values():
    """Gegenprobe zur Erklärung „fehlende Fahrtrichtung“: auch bei einem Weg, der umkehren darf, werden die wahren Gewichte nicht gefunden und die Handparameter nicht geschlagen."""
    r = ev.check_without_direction(ev.Settings())
    assert r["w"][1] < 0.95 and r["w"][0] > 0.8                           # nicht die wahre Struktur (nur „eine Zelle“)
    assert r["hit_learned"] <= r["hit_hand"] + 0.02
    assert r["hit_hand"] == pytest.approx(0.70, abs=0.06) and r["hit_learned"] == pytest.approx(0.66, abs=0.06)
    assert r["ll"] < r["ll_hand"] + 5.0                                    # EM landet nicht über den Handparametern: ein schlechteres stationäres Ergebnis
