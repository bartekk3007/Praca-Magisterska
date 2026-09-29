import os
import pandas as pd
from itertools import product

# =========================
# PARAMS
# =========================

DATA_DIR = "spade_files"

PARTICIPANTS = [f"P{str(i).zfill(2)}" for i in range(1, 11)]
CAMERAS = ["DL", "DR", "UL", "UR"]
SYSTEMS = ["FR", "LU", "XP"]

# =========================
# PARSE FILE NAME (opcjonalne sprawdzenie zgodności)
# =========================

def parse_filename(fname):
    # P01DLFRanger.npy
    try:
        p = fname[:3]
        cam = fname[3:5]
        sys = fname[5:7]
        return p, cam, sys
    except:
        return None

# =========================
# BUILD META TABLE
# =========================

def build_meta_table():

    meta = []
    sid = 0

    for p, cam, sys in product(PARTICIPANTS, CAMERAS, SYSTEMS):

        meta.append({
            "ID_SZEREGU": sid,
            "PARTICIPANT": p,
            "CAMERA": cam,
            "SYSTEM": sys
        })

        sid += 1

    return meta

# =========================
# OPTIONAL: sanity check with files
# =========================

def check_files(meta):
    files = os.listdir(DATA_DIR)

    expected = set()
    for m in meta:
        expected.add(m["PARTICIPANT"] + m["CAMERA"] + m["SYSTEM"])

    existing = set()
    for f in files:
        if f.endswith(".npy"):
            parsed = parse_filename(f)
            if parsed:
                existing.add("".join(parsed))

    missing = expected - existing

    print("Missing combinations:", len(missing))
    if len(missing) > 0:
        print(list(missing)[:10])

# =========================
# RUN
# =========================

meta = build_meta_table()

df = pd.DataFrame(meta)
df.to_csv("meta_table.csv", index=False)

print("DONE")
print("Rows:", len(df))

check_files(meta)