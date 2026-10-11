"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets."""

# Straßengitter: NODES x NODES Kreuzungen, Straßen in SPACING-Abstand, Zustände = Straßenstücke („Zellen“) von CELL Metern
NODES = 7
CELL = 10.0
SPACING = 50                      # fest: 420 Zellen
T_STEPS = 60
HIT_RADIUS = 15.0                 # „getroffen“: Netzabstand zur wahren Zelle höchstens so groß (m)
MAX_HOPS = 4
N_CLASSES = 4                     # Schrittklassen 0 (bleiben), 1, 2, 3 Zellen weiter
STEP_HAND_BETA = 5.0              # Handparameter: Gewicht ~ exp(-|Schrittlänge - CELL| / beta)

# Reglergrenzen
SIGMA_MIN, SIGMA_MAX, DEFAULT_SIGMA = 5.0, 40.0, 30.0
TRAIN_CHOICES = (5, 10, 20, 40)
DEFAULT_TRAIN = 10
ITER_MIN, ITER_MAX, DEFAULT_ITER = 1, 20, 10
START_CHOICES = ("von Hand", "gleich", "Stehenbleiben bevorzugt", "zufällig")
DEFAULT_START = "gleich"
SEED_MAX, DEFAULT_SEED = 999999, 35

# feste Instanzen: Training aus Seeds ab TRAIN_SEED0, Test aus festen anderen Seeds
TRAIN_SEED0 = 600
TEST_SEEDS = tuple(range(900, 930))
SIGMA_SWEEP = (5.0, 10.0, 20.0, 30.0, 40.0)
RANDOM_STARTS = 3                 # Zahl der zufälligen Starts im Experiment zu lokalen Optima
SWEEP_ITER = 8
DATA_SWEEP = (5, 10, 20, 40)

COLOR_HAND = "#c97b2a"
COLOR_LEARNED = "#4c78a8"
COLOR_TRUTH = "#2F6B65"
COLOR_START = "#9a9a9a"

PRESETS = {
    "Standard (Voreinstellung)": {"sg": DEFAULT_SIGMA, "n": DEFAULT_TRAIN, "it": DEFAULT_ITER, "start": DEFAULT_START, "seed": DEFAULT_SEED},
    "Wenig Rauschen": {"sg": 5.0, "n": DEFAULT_TRAIN, "it": DEFAULT_ITER, "start": DEFAULT_START, "seed": DEFAULT_SEED},
    "Start von Hand": {"sg": DEFAULT_SIGMA, "n": DEFAULT_TRAIN, "it": DEFAULT_ITER, "start": "von Hand", "seed": DEFAULT_SEED},
    "Viel Training": {"sg": DEFAULT_SIGMA, "n": 40, "it": DEFAULT_ITER, "start": DEFAULT_START, "seed": DEFAULT_SEED},
}
PRESET_HELP = {
    "Standard (Voreinstellung)": "30 m GPS-Rauschen, 10 Trainingsfahrten, Start mit gleich gewichteten Schrittklassen.",
    "Wenig Rauschen": "5 m Rauschen: die Messungen zeigen die Schritte fast direkt, das Lernen kann gelingen.",
    "Start von Hand": "Der Lauf beginnt bei den von Hand gesetzten Übergängen.",
    "Viel Training": "40 statt 10 Fahrten: mehr Daten, gleiches Rauschen.",
}
