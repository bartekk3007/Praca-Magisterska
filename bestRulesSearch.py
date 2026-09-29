import pandas as pd
import os

# =====================================================
# CONFIG
# =====================================================

FILE = "spade_output/emotion_segments_P01_neutral_True.csv"
OUT_FILE = "spade_output/affective_rules_P01.csv"

# =====================================================
# RULES (tak jak podałeś)
# =====================================================

RULES = {
    (5, 6), (6, 5),
    (4, 6), (6, 4),
    (3, 6), (6, 3),
    (2, 6), (6, 2),
    (1, 6), (6, 1)
}

# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_csv(FILE)

print("Loaded segments:", len(df))

# =====================================================
# SORT (ważne!)
# =====================================================

df = df.sort_values(["seq_id", "start"]).reset_index(drop=True)

# =====================================================
# DETECT RULES
# =====================================================

rows = []

for seq_id in df["seq_id"].unique():

    sub = df[df["seq_id"] == seq_id].copy().reset_index(drop=True)

    for i in range(len(sub) - 1):

        e1 = sub.loc[i, "emotion_id"]
        e2 = sub.loc[i + 1, "emotion_id"]

        pair = (e1, e2)

        if pair in RULES:

            rows.append([
                seq_id,
                sub.loc[i, "end"],     # moment przejścia
                e1,
                e2,
                sub.loc[i, "duration_windows"],
                sub.loc[i + 1, "duration_windows"]
            ])

# =====================================================
# SAVE
# =====================================================

out_df = pd.DataFrame(rows, columns=[
    "seq_id",
    "transition_time",
    "from_emotion",
    "to_emotion",
    "from_duration",
    "to_duration"
])

out_df.to_csv(OUT_FILE, index=False)

print("\nDONE")
print("Rules found:", len(out_df))
print("Saved:", OUT_FILE)