"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; Permalink begrenzt und rastet ein."""

import bwl_constants as C
import bwl_evaluation as ev
import bwl_presets as P


def _settings_from(p):
    return ev.Settings(sigma=p["sg"], n_train=p["n"], n_iter=p["it"], start=p["start"], seed=p["seed"])


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_regler_bounds():
    for name, p in C.PRESETS.items():
        assert C.SIGMA_MIN <= p["sg"] <= C.SIGMA_MAX and p["n"] in C.TRAIN_CHOICES, name
        assert C.ITER_MIN <= p["it"] <= C.ITER_MAX and p["start"] in C.START_CHOICES and 0 <= p["seed"] <= C.SEED_MAX, name


def test_default_preset_equals_the_default_settings():
    assert _settings_from(C.PRESETS["Standard (Voreinstellung)"]) == ev.Settings()


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_permalink_clamps_numbers_and_snaps_choices():
    it = P.SETTING_SPECS["iter_slider"]
    assert P.clamp_value(it, 1000) == C.ITER_MAX and P.clamp_value(it, -2) == C.ITER_MIN
    st = P.SETTING_SPECS["start_select"]
    assert P.clamp_value(st, "gleich") == "gleich" and P.clamp_value(st, "irgendwas") is None


def test_the_presets_show_what_their_help_text_claims():
    base = C.PRESETS["Standard (Voreinstellung)"]
    assert C.PRESETS["Wenig Rauschen"]["sg"] < base["sg"]
    assert C.PRESETS["Start von Hand"]["start"] == "von Hand"
    assert C.PRESETS["Viel Training"]["n"] > base["n"]
