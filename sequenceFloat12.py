import os
import re
import numpy as np
from collections import defaultdict, Counter

# =========================================================
# CONFIG
# =========================================================

DATA_DIR = "float_downloads"
OUT_DIR = "spade_user_merged"

WINDOW_SIZE = 2500

USE_NEUTRAL = False
EMPTY_MODE = "zero"   # "skip" | "zero"

RMS_THRESHOLD = 0.20
ACTIVITY_RATIO = 0.20

EPS = 1e-8

os.makedirs(OUT_DIR, exist_ok=True)

suffix = f"neutral_{USE_NEUTRAL}_empty_{EMPTY_MODE}"
OUT_FILE = os.path.join(OUT_DIR, f"spade_input_{suffix}.txt")

# =========================================================
# EMOTIONS
# =========================================================

EMOTIONS_NEUTRAL = ["anger", "disgust", "happiness", "sadness", "surprise", "neutral"]
EMOTIONS_NO_NEUTRAL = ["anger", "disgust", "happiness", "sadness", "surprise"]

EMOTIONS = EMOTIONS_NEUTRAL if USE_NEUTRAL else EMOTIONS_NO_NEUTRAL
emotion_to_id = {e: i + 1 for i, e in enumerate(EMOTIONS)}

# =========================================================
# PARSER
# =========================================================

pattern = re.compile(r"(P\d{2})(DL|DR|UL|UR)(FR|LU|XP)(.+)\.npy", re.IGNORECASE)

def parse(fname):
    m = pattern.match(fname)
    if not m:
        return None
    return m.group(1), m.group(2), m.group(3), m.group(4).lower()

# =========================================================
# LOAD & GROUP
# =========================================================

groups = defaultdict(dict)

for f in os.listdir(DATA_DIR):
    if not f.endswith(".npy"):
        continue

    parsed = parse(f)
    if not parsed:
        continue

    user, cam, sys, emo = parsed

    if emo not in EMOTIONS:
        continue

    groups[(user, cam, sys)][emo] = os.path.join(DATA_DIR, f)

print("SEQUENCES FOUND:", len(groups))

# =========================================================
# RMS FUNCTION (stable)
# =========================================================

def rms(x):
    x = x.astype(np.float64)
    return np.sqrt(np.mean(x * x) + EPS)

# =========================================================
# PROCESS USER SEQUENCES
# =========================================================

spade_lines = []
meta_table = []

empty_windows = 0
tie_windows = 0

user_to_lines = defaultdict(list)

seq_id_global = 0

for (user, cam, sys), files in sorted(groups.items()):

    signals = {}
    length = None

    for emo, path in files.items():
        arr = np.load(path).astype(np.float32)
        signals[emo] = arr
        length = len(arr)

    if length is None:
        continue

    n_windows = length // WINDOW_SIZE

    sequence = []

    for w in range(n_windows):

        start = w * WINDOW_SIZE
        end = start + WINDOW_SIZE

        scores = {}

        for emo in EMOTIONS:

            if emo not in signals:
                continue

            window = signals[emo][start:end]

            if len(window) == 0:
                continue

            score = rms(window)

            # optional pruning
            if score < RMS_THRESHOLD:
                continue

            # activity ratio filter
            active_ratio = np.mean(window >= RMS_THRESHOLD)

            if active_ratio < ACTIVITY_RATIO:
                continue

            scores[emo] = score

        # =====================================================
        # EMPTY CASE
        # =====================================================

        if len(scores) == 0:
            empty_windows += 1

            if EMPTY_MODE == "skip":
                continue
            else:
                sequence.append([0])  # 0 = EMPTY TOKEN
                continue

        # =====================================================
        # PICK MAX RMS
        # =====================================================

        max_val = max(scores.values())
        best = [e for e, v in scores.items() if abs(v - max_val) < 1e-9]

        if len(best) > 1:
            tie_windows += 1

        chosen = best[0]  # deterministic (lexicographic due to EMOTIONS order)

        sequence.append([emotion_to_id[chosen]])

        meta_table.append([
            seq_id_global,
            start,
            cam,
            sys,
            user
        ])

    # convert to SPADE line
    line = []

    for itemset in sequence:
        for v in itemset:
            line.append(str(v))
        line.append("-1")

    line.append("-2")

    user_to_lines[user].append(" ".join(line))

    seq_id_global += 1

# =========================================================
# MERGE USERS INTO 10 LINES
# =========================================================

final_lines = []

for user in sorted(user_to_lines.keys()):
    merged = " ".join(user_to_lines[user])
    final_lines.append(merged)

# =========================================================
# SAVE SPADE INPUT
# =========================================================

with open(OUT_FILE, "w") as f:
    for line in final_lines:
        f.write(line + "\n")

# =========================================================
# SAVE META
# =========================================================

meta_path = os.path.join(OUT_DIR, f"meta_{suffix}.csv")

np.savetxt(
    meta_path,
    meta_table,
    fmt="%s",
    delimiter=",",
    header="seq_id,moment,camera,system,user",
    comments=""
)

# =========================================================
# REPORT
# =========================================================

print("\nDONE")
print("OUTPUT:", OUT_FILE)
print("META:", meta_path)
print("EMPTY WINDOWS:", empty_windows)
print("TIE WINDOWS:", tie_windows)
print("USERS:", len(final_lines))