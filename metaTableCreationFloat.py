import os
import re
import numpy as np
from collections import defaultdict

# =====================================================
# CONFIG
# =====================================================

DATA_DIR = "float_downloads"
OUT_DIR = "spade_output"

WINDOW_SIZE = 2500
THRESHOLD = 0.20

os.makedirs(OUT_DIR, exist_ok=True)

# =====================================================
# EMOTIONS (nieużywane w output, ale zostaje do logiki)
# =====================================================

EMOTIONS_NEUTRAL = ["anger", "disgust", "happiness", "sadness", "surprise", "neutral"]
EMOTIONS_NO_NEUTRAL = ["anger", "disgust", "happiness", "sadness", "surprise"]

# =====================================================
# PARSER
# =====================================================

pattern = re.compile(r"(P\d{2})(DL|DR|UL|UR)(FR|LU|XP)(.+)\.npy", re.IGNORECASE)

def parse(fname):
    m = pattern.match(fname)
    if not m:
        return None
    return m.group(1), m.group(2), m.group(3), m.group(4).lower()

# =====================================================
# LOAD GROUPS (DETERMINISTIC)
# =====================================================

groups = defaultdict(dict)

for f in sorted(os.listdir(DATA_DIR)):
    if not f.endswith(".npy"):
        continue

    parsed = parse(f)
    if not parsed:
        continue

    user, cam, sys, emo = parsed
    groups[(user, cam, sys)][emo] = os.path.join(DATA_DIR, f)

groups = dict(sorted(groups.items()))

# =====================================================
# BUILD META
# =====================================================

def build_meta(use_neutral=True, empty_mode="True"):

    emotions = EMOTIONS_NEUTRAL if use_neutral else EMOTIONS_NO_NEUTRAL

    meta = []
    seq_id_global = 0
    empty_windows = 0

    for (user, cam, sys), files in groups.items():

        signals = {}
        length = None

        for emo, path in files.items():
            if emo not in emotions:
                continue

            arr = np.load(path).astype(np.float32)
            signals[emo] = arr
            length = len(arr)

        if length is None:
            continue

        n_windows = length // WINDOW_SIZE

        for w in range(n_windows):

            moment = w * WINDOW_SIZE

            scores = {}

            for emo in emotions:

                if emo not in signals:
                    continue

                window = signals[emo][moment:moment+WINDOW_SIZE]

                score = np.mean(window)

                if score >= THRESHOLD:
                    scores[emo] = score

            # =========================
            # EMPTY CASE
            # =========================

            if len(scores) == 0:

                empty_windows += 1

                if empty_mode == "False":
                    continue

                meta.append([
                    seq_id_global,
                    moment,
                    cam,
                    sys,
                    user
                ])

            else:

                meta.append([
                    seq_id_global,
                    moment,
                    cam,
                    sys,
                    user
                ])

        seq_id_global += 1

    return meta, empty_windows

# =====================================================
# RUN CONFIGS
# =====================================================

configs = [
    (True, "True"),
    (True, "False"),
    (False, "True"),
    (False, "False"),
]

for use_neutral, empty_mode in configs:

    meta, empty_count = build_meta(use_neutral, empty_mode)

    suffix = f"neutral_{use_neutral}_empty_{empty_mode}"

    out_path = os.path.join(OUT_DIR, f"meta_{suffix}.csv")

    np.savetxt(
        out_path,
        meta,
        fmt="%s",
        delimiter=",",
        header="seq_id,moment,camera,system,user",
        comments=""
    )

    print("\nDONE:", suffix)
    print("ROWS:", len(meta))
    print("EMPTY WINDOWS:", empty_count)
    print("SAVED:", out_path)