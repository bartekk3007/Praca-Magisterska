import pandas as pd
import numpy as np
from statsmodels.stats.inter_rater import fleiss_kappa

# =====================================================
# CONFIG
# =====================================================

INPUT_FILE = "spade_output/frame_votes_P01.csv"

# =====================================================
# LOAD
# =====================================================

df = pd.read_csv(INPUT_FILE)

print("Rows:", len(df))

# =====================================================
# FLEISS FUNCTION
# =====================================================

def compute_fleiss(df, cols):

    emotions = sorted(
        df[cols]
        .stack()
        .unique()
    )

    emo_to_col = {
        e: i
        for i, e in enumerate(emotions)
    }

    table = np.zeros(
        (len(df), len(emotions)),
        dtype=np.int32
    )

    for r, (_, row) in enumerate(df.iterrows()):

        if r % 100000 == 0:
            print(
                f"Processing {r:,}/{len(df):,}"
            )

        for col in cols:

            emotion = row[col]

            table[
                r,
                emo_to_col[emotion]
            ] += 1

    return fleiss_kappa(table)

# =====================================================
# GLOBAL KAPPA
# =====================================================

all_cols = [
    "DL_FR","DL_LU","DL_XP",
    "DR_FR","DR_LU","DR_XP",
    "UL_FR","UL_LU","UL_XP",
    "UR_FR","UR_LU","UR_XP"
]

global_kappa = compute_fleiss(
    df,
    all_cols
)

print("\nGLOBAL KAPPA")
print(global_kappa)

# =====================================================
# CAMERA KAPPA
# =====================================================

camera_groups = {
    "DL": ["DL_FR","DL_LU","DL_XP"],
    "DR": ["DR_FR","DR_LU","DR_XP"],
    "UL": ["UL_FR","UL_LU","UL_XP"],
    "UR": ["UR_FR","UR_LU","UR_XP"]
}

camera_results = []

print("\nCAMERA KAPPA")

for camera, cols in camera_groups.items():

    kappa = compute_fleiss(
        df,
        cols
    )

    camera_results.append([
        camera,
        kappa
    ])

    print(camera, kappa)

camera_df = pd.DataFrame(
    camera_results,
    columns=[
        "camera",
        "fleiss_kappa"
    ]
)

camera_df.to_csv(
    "camera_kappa.csv",
    index=False
)

# =====================================================
# SYSTEM KAPPA
# =====================================================

system_groups = {
    "FR": ["DL_FR","DR_FR","UL_FR","UR_FR"],
    "LU": ["DL_LU","DR_LU","UL_LU","UR_LU"],
    "XP": ["DL_XP","DR_XP","UL_XP","UR_XP"]
}

system_results = []

print("\nSYSTEM KAPPA")

for system, cols in system_groups.items():

    kappa = compute_fleiss(
        df,
        cols
    )

    system_results.append([
        system,
        kappa
    ])

    print(system, kappa)

system_df = pd.DataFrame(
    system_results,
    columns=[
        "system",
        "fleiss_kappa"
    ]
)

system_df.to_csv(
    "system_kappa.csv",
    index=False
)

# =====================================================
# SAVE SUMMARY
# =====================================================

with open(
        "spade_output/kappa_summary.txt",
    "w"
) as f:

    f.write(
        f"GLOBAL_KAPPA={global_kappa}\n\n"
    )

    f.write(
        "CAMERA_KAPPA\n"
    )

    for _, row in camera_df.iterrows():

        f.write(
            f"{row['camera']}: "
            f"{row['fleiss_kappa']}\n"
        )

    f.write("\n")

    f.write(
        "SYSTEM_KAPPA\n"
    )

    for _, row in system_df.iterrows():

        f.write(
            f"{row['system']}: "
            f"{row['fleiss_kappa']}\n"
        )

print("\nDONE")
print("Saved:")
print("camera_kappa.csv")
print("system_kappa.csv")
print("kappa_summary.txt")