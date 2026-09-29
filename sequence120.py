import os
import re
import numpy as np
from collections import defaultdict

# =========================
# PARAMETRY
# =========================

DATA_DIR = "binary_downloads"

WINDOW_SIZE = 500
MIN_ACTIVE_RATIO = 0.20
MIN_STATE_DURATION = 2

EMOTIONS = [
    "anger",
    "disgust",
    "happiness",
    "sadness",
    "surprise"
]

pattern = re.compile(
    r"(P\d{2})(DL|DR|UL|UR)(FR|LU|XP)(.+)\.npy",
    re.IGNORECASE
)

# =========================
# PARSER
# =========================

def parse_filename(fname):
    m = pattern.match(fname)
    if not m:
        return None

    return {
        "user": m.group(1),
        "camera": m.group(2),
        "system": m.group(3),
        "emotion": m.group(4).lower()
    }

# =========================
# GRUPOWANIE
# =========================

groups = defaultdict(dict)

for f in os.listdir(DATA_DIR):
    if not f.endswith(".npy"):
        continue

    info = parse_filename(f)
    if not info:
        continue

    if info["emotion"] not in EMOTIONS:
        continue

    key = (info["user"], info["camera"], info["system"])
    groups[key][info["emotion"]] = os.path.join(DATA_DIR, f)

print("Liczba sekwencji:", len(groups))

# =========================
# MAPOWANIE EMOCJI
# =========================

emotion_to_id = {e: i + 1 for i, e in enumerate(EMOTIONS)}

# =========================
# BUDOWA SEKWENCJI
# =========================

all_sequences = []

for key, files in groups.items():

    signals = {}
    length = None

    for emo, path in files.items():
        arr = np.load(path).astype(np.uint8)
        signals[emo] = arr
        length = len(arr)

    if length is None:
        continue

    n_windows = length // WINDOW_SIZE

    sequence = []
    prev_state = None
    state_count = 0

    for w in range(n_windows):

        start = w * WINDOW_SIZE
        end = start + WINDOW_SIZE

        active = []

        for emo in EMOTIONS:

            if emo not in signals:
                continue

            window = signals[emo][start:end]
            ratio = window.mean()

            if ratio >= MIN_ACTIVE_RATIO:
                active.append(emo)

        # ❌ brak emocji → pomijamy okno (NIE neutral!)
        if len(active) == 0:
            continue

        state = tuple(sorted(active))

        # kompresja stanów
        if state == prev_state:
            state_count += 1
            continue
        else:
            if prev_state is not None and state_count >= MIN_STATE_DURATION:
                sequence.append(prev_state)

            prev_state = state
            state_count = 1

    # flush
    if prev_state is not None:
        sequence.append(prev_state)

    if len(sequence) > 0:
        all_sequences.append(sequence)

print("Gotowe sekwencje:", len(all_sequences))

# =========================
# ZAPIS SPMF
# =========================

OUT = "spade_input_no_neutral.txt"

with open(OUT, "w") as f:

    for seq in all_sequences:

        line = []

        for itemset in seq:
            for emo in itemset:
                line.append(str(emotion_to_id[emo]))
            line.append("-1")

        line.append("-2")
        f.write(" ".join(line) + "\n")

print("Zapisano:", OUT)