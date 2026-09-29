import os
import re
import csv
import numpy as np
from collections import defaultdict

# =====================================================
# CONFIG
# =====================================================

DATA_DIR = "float_downloads"

USER_FILTER = "P01"   # None = wszyscy, "P01" = tylko P01

THRESHOLD = 0.20

OUT_FILE = "emotion_values_P01_time_sorted.csv"

# =====================================================
# EMOTIONS
# =====================================================

EMOTIONS = [
    "anger",
    "disgust",
    "happiness",
    "sadness",
    "surprise",
    "neutral"
]

# =====================================================
# FILE PATTERN
# =====================================================

pattern = re.compile(
    r"(P\d{2})(DL|DR|UL|UR)(FR|LU|XP)(.+)\.npy",
    re.IGNORECASE
)

# =====================================================
# CLEAN FUNCTION
# =====================================================

def clean_value(v):
    if np.isnan(v):
        return np.nan
    if abs(v) < 1e-4:
        return 0.0
    return round(float(v), 5)

# =====================================================
# LOAD FILES
# =====================================================

groups = defaultdict(dict)

for fname in os.listdir(DATA_DIR):

    if not fname.endswith(".npy"):
        continue

    m = pattern.match(fname)
    if not m:
        continue

    user, camera, system, emotion = m.groups()
    emotion = emotion.lower()

    # =================================================
    # FILTER USER (P01 ONLY)
    # =================================================
    if USER_FILTER is not None and user != USER_FILTER:
        continue

    if emotion not in EMOTIONS:
        continue

    groups[(user, camera, system)][emotion] = os.path.join(DATA_DIR, fname)

print("SEQUENCES:", len(groups))

# =====================================================
# LOAD ALL SIGNALS
# =====================================================

signals_all = {}
length_all = {}

for seq_id, key in enumerate(sorted(groups.keys())):

    user, camera, system = key

    signals = {}
    length = None

    for emotion in EMOTIONS:

        if emotion not in groups[key]:
            continue

        arr = np.load(groups[key][emotion], mmap_mode="r")
        signals[emotion] = arr

        if length is None:
            length = len(arr)

    signals_all[seq_id] = {
        "key": key,
        "signals": signals,
        "length": length
    }

# =====================================================
# FIND GLOBAL MAX LENGTH
# =====================================================

max_len = max(v["length"] for v in signals_all.values())

print("MAX LENGTH:", max_len)

# =====================================================
# HEADER
# =====================================================

header = [
    "moment",
    "seq_id",
    "user",
    "camera",
    "system"
] + EMOTIONS + [
    "dominant_emotion",
    "is_empty",
    "active_emotions"
]

# =====================================================
# PROCESS TIME-GLOBAL
# =====================================================

with open(OUT_FILE, "w", newline="", encoding="utf8") as f:

    writer = csv.writer(f)
    writer.writerow(header)

    total_rows = 0

    for moment in range(max_len):

        for seq_id, data in signals_all.items():

            user, camera, system = data["key"]
            signals = data["signals"]
            length = data["length"]

            if moment >= length:
                continue

            values = {}
            active = []

            max_emotion = None
            max_value = -1.0

            # =========================================
            # EMOTION VALUES
            # =========================================

            for emotion in EMOTIONS:

                if emotion not in signals:
                    value = np.nan
                else:
                    value = clean_value(signals[emotion][moment])

                values[emotion] = value

                if (not np.isnan(value)) and value >= THRESHOLD:
                    active.append(emotion)

                    if value > max_value:
                        max_value = value
                        max_emotion = emotion

            # =========================================
            # DOMINANT
            # =========================================

            if len(active) == 0:
                dominant = "EMPTY"
                is_empty = 1
            else:
                dominant = max_emotion
                is_empty = 0

            row = [
                moment,
                seq_id,
                user,
                camera,
                system
            ]

            row += [values[e] for e in EMOTIONS]
            row += [dominant, is_empty, ";".join(active)]

            writer.writerow(row)

            total_rows += 1

            # =========================================
            # DEBUG
            # =========================================

            if total_rows % 10000 == 0:
                print("processed:", f"{total_rows:,}")

        # debug per moment
        if moment % 10 == 0:
            print("finished moment:", moment)

print("\nDONE")
print("ROWS:", f"{total_rows:,}")
print("SAVED:", OUT_FILE)