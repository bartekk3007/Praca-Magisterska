import pandas as pd
import numpy as np
from statsmodels.stats.inter_rater import fleiss_kappa

# =====================================================
# CONFIG
# =====================================================

INPUT_FILE = "spade_output/emotion_segments_P01_neutral_True.csv"

OUTPUT_TABLE = "frame_votes_P01.csv"
OUTPUT_KAPPA = "frame_votes_P01_kappa.txt"

# =====================================================
# LOAD
# =====================================================

df = pd.read_csv(INPUT_FILE)

# =====================================================
# JUDGE NAME
# =====================================================

df["judge"] = (
    df["camera"].astype(str)
    + "_"
    + df["system"].astype(str)
)

judges = [
    "DL_FR","DL_LU","DL_XP",
    "DR_FR","DR_LU","DR_XP",
    "UL_FR","UL_LU","UL_XP",
    "UR_FR","UR_LU","UR_XP"
]

# =====================================================
# PREPARE SEGMENTS
# =====================================================

segments = {}

for judge in judges:

    seg = (
        df[df["judge"] == judge]
        .sort_values("start")
        .reset_index(drop=True)
    )

    segments[judge] = seg

# =====================================================
# MAX TIME
# =====================================================

max_time = int(df["end"].max())

print("MAX TIME:", max_time)

# =====================================================
# CURRENT SEGMENT POINTERS
# =====================================================

ptr = {}
current_emotion = {}

for judge in judges:

    ptr[judge] = 0

    if len(segments[judge]) > 0:
        current_emotion[judge] = int(
            segments[judge].iloc[0]["emotion_id"]
        )
    else:
        current_emotion[judge] = 0

# =====================================================
# BUILD FRAME TABLE
# =====================================================

rows = []

for frame in range(max_time):

    if frame % 1000 == 0:
        print(
            f"{frame:,}/{max_time:,} "
            f"({100*frame/max_time:.2f}%)"
        )

    row = {
        "frame": frame
    }

    emotions = []

    for judge in judges:

        segs = segments[judge]

        while (
            ptr[judge] < len(segs) - 1
            and frame >= segs.iloc[ptr[judge]]["end"]
        ):
            ptr[judge] += 1

        emotion = int(
            segs.iloc[ptr[judge]]["emotion_id"]
        )

        row[judge] = emotion
        emotions.append(emotion)

    counts = pd.Series(emotions).value_counts()

    row["majority_emotion"] = int(counts.index[0])
    row["majority_count"] = int(counts.iloc[0])

    rows.append(row)

# =====================================================
# SAVE TABLE
# =====================================================

frame_table = pd.DataFrame(rows)

frame_table.to_csv(
    OUTPUT_TABLE,
    index=False
)

print("Saved:", OUTPUT_TABLE)

# =====================================================
# FLEISS KAPPA
# =====================================================

emotion_ids = sorted(
    frame_table[judges]
    .stack()
    .unique()
)

emotion_to_col = {
    e: i
    for i, e in enumerate(emotion_ids)
}

ratings = np.zeros(
    (
        len(frame_table),
        len(emotion_ids)
    ),
    dtype=np.int32
)

for r, row in frame_table.iterrows():

    for judge in judges:

        emotion = row[judge]

        ratings[
            r,
            emotion_to_col[emotion]
        ] += 1

    if r % 10000 == 0:
        print(
            f"Kappa matrix: {r:,}/{len(frame_table):,}"
        )

kappa = fleiss_kappa(ratings)

with open(OUTPUT_KAPPA, "w") as f:

    f.write(
        f"Fleiss Kappa = {kappa:.10f}\n"
    )

    f.write(
        f"Frames = {len(frame_table)}\n"
    )

    f.write(
        f"Judges = {len(judges)}\n"
    )

    f.write(
        f"Categories = {len(emotion_ids)}\n"
    )

print()
print("FLEISS KAPPA:", kappa)
print("Saved:", OUTPUT_KAPPA)