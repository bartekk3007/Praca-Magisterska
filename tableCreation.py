import os
import re
import numpy as np
from collections import defaultdict

# =========================================================
# CONFIG
# =========================================================

DATA_DIR = "binary_downloads"
WINDOW_SIZE = 2500
WINDOW_THRESHOLD = 0.20
USE_NEUTRAL = False

# =========================================================
# EMOTIONS
# =========================================================

EMOTIONS_NEUTRAL = ["anger", "disgust", "happiness", "sadness", "surprise", "neutral"]
EMOTIONS_NO_NEUTRAL = ["anger", "disgust", "happiness", "sadness", "surprise"]

EMOTIONS = EMOTIONS_NEUTRAL if USE_NEUTRAL else EMOTIONS_NO_NEUTRAL
emotion_to_id = {e: i + 1 for i, e in enumerate(EMOTIONS)}

# =========================================================
# RULES
# =========================================================

if USE_NEUTRAL:
    RULES = [(4, 6), (6, 4), (3, 6), (6, 3), (2, 6), (6, 2)]
else:
    RULES = [(3, 4), (4, 3), (2, 4), (4, 2), (1, 4), (4, 1)]

# =========================================================
# FILE PARSER
# =========================================================

pattern = re.compile(r"(P\d{2})(DL|DR|UL|UR)(FR|LU|XP)(.+)\.npy", re.IGNORECASE)

# =========================================================
# LOAD GROUPS
# =========================================================

groups = defaultdict(dict)

for f in os.listdir(DATA_DIR):
    if not f.endswith(".npy"):
        continue

    m = pattern.match(f)
    if not m:
        continue

    participant = m.group(1)
    camera = m.group(2)
    system = m.group(3)
    emotion = m.group(4).lower()

    if emotion not in EMOTIONS:
        continue

    key = (participant, camera, system)
    groups[key][emotion] = os.path.join(DATA_DIR, f)

# =========================================================
# DETERMINISTIC ORDER
# =========================================================

sorted_keys = sorted(groups.keys())
key_to_seq_id = {k: i for i, k in enumerate(sorted_keys)}

print("SEQUENCES:", len(sorted_keys))

# =========================================================
# OUTPUTS
# =========================================================

meta_table = []
rule_table = []
spade_sequences = []

# =========================================================
# SPADE INPUT GENERATION + META TABLE
# =========================================================

for key in sorted_keys:

    participant, camera, system = key
    seq_id = key_to_seq_id[key]
    files = groups[key]

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

    for w in range(n_windows):

        start = w * WINDOW_SIZE
        end = start + WINDOW_SIZE

        active = set()

        for emo in EMOTIONS:
            if emo not in signals:
                continue

            if signals[emo][start:end].mean() >= WINDOW_THRESHOLD:
                active.add(emotion_to_id[emo])

        if len(active) == 0:
            continue

        # META TABLE (1:1 with itemset)
        meta_table.append([
            seq_id,
            start,
            camera,
            system,
            participant
        ])

        sequence.append(active)

    if len(sequence) > 0:
        spade_sequences.append((seq_id, sequence))

# =========================================================
# SAVE SPADE INPUT
# =========================================================

suffix = "neutral" if USE_NEUTRAL else "no_neutral"
spade_file = f"spade_input_{suffix}.txt"

with open(spade_file, "w") as f:

    for _, seq in spade_sequences:

        line = []

        for itemset in seq:
            for x in sorted(itemset):
                line.append(str(x))
            line.append("-1")

        line.append("-2")
        f.write(" ".join(line) + "\n")

print("SPADE INPUT SAVED")

# =========================================================
# PARSE SPADE INPUT (TIME SOURCE)
# =========================================================

def parse_spade(line):
    tokens = line.strip().split()

    seq = []
    itemset = []

    for t in tokens:
        if t == "-1":
            seq.append(set(itemset))
            itemset = []
        elif t == "-2":
            break
        else:
            itemset.append(int(t))

    return seq

# =========================================================
# BUILD RULE TABLE FROM SPADE INPUT
# =========================================================

with open(spade_file) as f:

    for seq_id, line in enumerate(f):

        seq = parse_spade(line)

        for rid, (A, B) in enumerate(RULES):

            for t in range(len(seq) - 1):

                if A in seq[t] and B in seq[t + 1]:

                    rule_table.append([
                        rid,
                        seq_id,
                        (t) * WINDOW_SIZE,
                        (t + 1) * WINDOW_SIZE
                    ])

# =========================================================
# SAVE TABLES
# =========================================================

np.savetxt(
    f"meta_table_{suffix}.csv",
    meta_table,
    fmt="%s",
    delimiter=",",
    header="seq_id,moment,camera,system,participant",
    comments=""
)

np.savetxt(
    f"rule_table_{suffix}.csv",
    rule_table,
    fmt="%s",
    delimiter=",",
    header="rule_id,seq_id,od,do",
    comments=""
)

print("DONE")
print("META TABLE + RULE TABLE GENERATED (SPADE CONSISTENT)")