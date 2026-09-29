import os
import re
import numpy as np
import pandas as pd
from collections import defaultdict

# =====================================================
# CONFIG
# =====================================================

DATA_DIR = "float_downloads"
OUT_DIR = "spade_output"

WINDOW_SIZE = 1
RMS_THRESHOLD = 0.2

USE_NEUTRAL = True

os.makedirs(OUT_DIR, exist_ok=True)

# =====================================================
# EMOTIONS
# =====================================================

EMOTIONS_NO_NEUTRAL = ["anger", "disgust", "happiness", "sadness", "surprise"]
EMOTIONS_NEUTRAL = ["anger", "disgust", "happiness", "sadness", "surprise", "neutral"]

EMOTIONS = EMOTIONS_NEUTRAL if USE_NEUTRAL else EMOTIONS_NO_NEUTRAL
emotion_to_id = {e: i + 1 for i, e in enumerate(EMOTIONS)}

EMPTY_ID = 0

# =====================================================
# TARGET USER FILTER (<<< TUTAJ JEST P01 FILTER >>>)
# =====================================================

TARGET_USER = "P01"   # <<< ZMIANA TU DLA INNYCH OSÓB

# jeśli chcesz wyłączyć filtr:
# TARGET_USER = None

# =====================================================
# PARSER
# =====================================================

pattern = re.compile(r"(P\d{2})(DL|DR|UL|UR)(FR|LU|XP)(.+)\.npy", re.IGNORECASE)

# =====================================================
# LOAD ONLY P01 DATA
# =====================================================

groups = defaultdict(dict)

for f in os.listdir(DATA_DIR):

    if not f.endswith(".npy"):
        continue

    m = pattern.match(f)
    if not m:
        continue

    user, cam, sys, emo = m.group(1), m.group(2), m.group(3), m.group(4).lower()

    # =================================================
    # >>> FAST FILTER (P01 ONLY) <<<
    # =================================================
    if TARGET_USER is not None and user != TARGET_USER:
        continue

    if emo not in EMOTIONS:
        continue

    groups[(user, cam, sys)][emo] = os.path.join(DATA_DIR, f)

print("SEQUENCES LOADED:", len(groups))

# =====================================================
# RMS FUNCTION
# =====================================================

def rms(x):
    x = x.astype(np.float64)
    return np.sqrt(np.mean(x * x))

# =====================================================
# SEGMENTS OUTPUT
# =====================================================

segments = []

# =====================================================
# PROCESS ONLY P01
# =====================================================

for seq_id, ((user, cam, sys), files) in enumerate(sorted(groups.items())):

    print(f"[DEBUG] START SEQUENCE seq_id={seq_id} user={user} cam={cam} sys={sys}")

    signals = {}
    length = None

    for emo in EMOTIONS:
        if emo in files:
            arr = np.load(files[emo]).astype(np.float64)
            signals[emo] = arr
            length = len(arr)

    if length is None:
        continue

    n_windows = length // WINDOW_SIZE

    prev_label = None
    start_idx = 0

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

            scores[emo] = rms(window)

        # =================================================
        # EMPTY CASE
        # =================================================

        if len(scores) == 0:
            label = EMPTY_ID
        else:
            best_emo, best_val = max(scores.items(), key=lambda x: x[1])

            if best_val < RMS_THRESHOLD:
                label = EMPTY_ID
            else:
                label = emotion_to_id[best_emo]

        # =================================================
        # SEGMENTATION
        # =================================================

        if prev_label is None:
            prev_label = label
            start_idx = w

        elif label != prev_label:

            segments.append([
                seq_id,
                user,
                cam,
                sys,
                prev_label,
                start_idx * WINDOW_SIZE,
                w * WINDOW_SIZE,
                (w - start_idx)
            ])

            prev_label = label
            start_idx = w

    # last segment
    if prev_label is not None:
        segments.append([
            seq_id,
            user,
            cam,
            sys,
            prev_label,
            start_idx * WINDOW_SIZE,
            n_windows * WINDOW_SIZE,
            (n_windows - start_idx)
        ])

# =====================================================
# SAVE OUTPUT
# =====================================================

df = pd.DataFrame(segments, columns=[
    "seq_id",
    "user",
    "camera",
    "system",
    "emotion_id",
    "start",
    "end",
    "duration_windows"
])

out_file = os.path.join(
    OUT_DIR,
    f"emotion_segments_P01_neutral_{USE_NEUTRAL}.csv"
)

df.to_csv(out_file, index=False)

# =====================================================
# STATS
# =====================================================

stats = df.groupby("emotion_id")["duration_windows"].agg(["count", "sum"]).reset_index()

stats_file = out_file.replace(".csv", "_stats.csv")
stats.to_csv(stats_file, index=False)

print("\nDONE")
print("Segments:", len(df))
print("Saved:", out_file)
print("Stats:", stats_file)