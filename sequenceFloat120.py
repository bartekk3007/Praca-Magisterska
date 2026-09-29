import os
import re
import numpy as np
from collections import defaultdict

# =========================================================
# CONFIG
# =========================================================

DATA_DIR = "float_downloads"
OUT_DIR = "spade_output"

os.makedirs(OUT_DIR, exist_ok=True)

WINDOW_SIZE = 500

VALUE_THRESHOLD = 0.20
ACTIVE_RATIO_THRESHOLD = 0.20

# =========================================================
# SWITCHES
# =========================================================

USE_NEUTRAL = False     # True/False
USE_EMPTY = False       # True = zapisuj EMPTY jako 8, False = pomijaj

EMPTY_ID = 8

# =========================================================
# EMOTIONS
# =========================================================

EMOTIONS_NO_NEUTRAL = ["anger", "disgust", "happiness", "sadness", "surprise"]
EMOTIONS_NEUTRAL = ["anger", "disgust", "happiness", "sadness", "surprise", "neutral"]

EMOTIONS = EMOTIONS_NEUTRAL if USE_NEUTRAL else EMOTIONS_NO_NEUTRAL

emotion_to_id = {e: i + 1 for i, e in enumerate(EMOTIONS)}

# =========================================================
# FILE PATTERN
# =========================================================

pattern = re.compile(
    r"(P\d{2})(DL|DR|UL|UR)(FR|LU|XP)(.+)\.npy",
    re.IGNORECASE
)

# =========================================================
# GROUP FILES
# =========================================================

groups = defaultdict(dict)

for fname in os.listdir(DATA_DIR):

    if not fname.endswith(".npy"):
        continue

    m = pattern.match(fname)
    if not m:
        continue

    participant = m.group(1)
    camera = m.group(2)
    system = m.group(3)
    emotion = m.group(4).lower()

    if emotion not in EMOTIONS:
        continue

    groups[(participant, camera, system)][emotion] = os.path.join(DATA_DIR, fname)

# deterministyczna kolejność
sorted_keys = sorted(groups.keys())

print("SEQUENCES FOUND:", len(sorted_keys))

# =========================================================
# STATS
# =========================================================

all_sequences = []
total_windows = 0
empty_windows = 0
ties = 0

tie_details = []

# =========================================================
# PROCESS
# =========================================================

for seq_id, key in enumerate(sorted_keys):

    participant, camera, system = key
    files = groups[key]

    signals = {}
    length = None

    # load signals
    for emotion in EMOTIONS:

        if emotion not in files:
            continue

        arr = np.load(files[emotion]).astype(np.float64)
        signals[emotion] = arr

        if length is None:
            length = len(arr)

    if length is None:
        continue

    n_windows = length // WINDOW_SIZE

    sequence = []

    for w in range(n_windows):

        total_windows += 1

        start = w * WINDOW_SIZE
        end = start + WINDOW_SIZE

        candidates = {}

        # =================================================
        # EMOTION SCORING
        # =================================================

        for emotion in EMOTIONS:

            if emotion not in signals:
                continue

            window = signals[emotion][start:end]

            active_ratio = np.count_nonzero(
                window >= VALUE_THRESHOLD
            ) / len(window)

            if active_ratio < ACTIVE_RATIO_THRESHOLD:
                continue

            rms = np.sqrt(np.mean(window * window))

            candidates[emotion] = rms

        # =================================================
        # EMPTY CASE
        # =================================================

        if len(candidates) == 0:

            empty_windows += 1

            if USE_EMPTY:
                sequence.append(EMPTY_ID)

            continue

        # =================================================
        # PICK WINNER
        # =================================================

        max_rms = max(candidates.values())

        winners = [
            e for e, v in candidates.items()
            if v == max_rms
        ]

        if len(winners) > 1:
            ties += 1
            tie_details.append((participant, camera, system, w, winners))

        winner = sorted(winners)[0]
        sequence.append(emotion_to_id[winner])

    all_sequences.append(sequence)

# =========================================================
# OUTPUT FILES
# =========================================================

suffix = f"neutral_{USE_NEUTRAL}_empty_{USE_EMPTY}"

out_file = os.path.join(OUT_DIR, f"spade_{suffix}.txt")
tie_file = os.path.join(OUT_DIR, f"ties_{suffix}.txt")

# =========================================================
# SAVE SPADE
# =========================================================

with open(out_file, "w") as f:

    for seq in all_sequences:

        line = []

        for x in seq:
            line.append(str(x))
            line.append("-1")

        line.append("-2")

        f.write(" ".join(line) + "\n")

# =========================================================
# SAVE TIES
# =========================================================

with open(tie_file, "w") as f:

    for t in tie_details:
        f.write(f"{t}\n")

# =========================================================
# SUMMARY
# =========================================================

print("\n===========================")
print("DONE")
print("===========================")

print("Sequences:", len(all_sequences))
print("Total windows:", total_windows)
print("Empty windows:", empty_windows)
print("Ties:", ties)

if total_windows > 0:
    print("Empty %:", 100 * empty_windows / total_windows)
    print("Tie %:", 100 * ties / total_windows)

print("\nOUTPUT:")
print("SPADE:", out_file)
print("TIES :", tie_file)