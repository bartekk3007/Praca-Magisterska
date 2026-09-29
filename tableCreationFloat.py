import os
import re
import csv
from collections import defaultdict

# =====================================================
# CONFIG
# =====================================================

SPADE_DIR = "spade_output"
DATA_DIR = "float_downloads"
OUT_DIR = "spade_output"

WINDOW_SIZE = 100
MIN_SUPPORT_COMBINATIONS = 7

# =====================================================
# RULE SETS
# =====================================================

RULES_NEUTRAL_TRUE = [
    (5,6), (6,5),
    (4,6), (6,4),
    (3,6), (6,3),
    (2,6), (6,2),
    (1,6), (6,1)
]

RULES_NEUTRAL_FALSE = [
    (4,5), (5,4),
    (3,5), (5,3),
    (2,5), (5,2),
    (1,5), (5,1)
]

# =====================================================
# BUILD seq_id -> user,camera,system
# =====================================================

pattern = re.compile(
    r"(P\d{2})(DL|DR|UL|UR)(FR|LU|XP)(.+)\.npy",
    re.IGNORECASE
)

groups = {}

for fname in os.listdir(DATA_DIR):
    if not fname.endswith(".npy"):
        continue

    m = pattern.match(fname)
    if not m:
        continue

    user = m.group(1)
    camera = m.group(2)
    system = m.group(3)

    groups[(user, camera, system)] = True

sorted_keys = sorted(groups.keys())

seq_info = {}

for seq_id, (user, camera, system) in enumerate(sorted_keys):
    seq_info[seq_id] = {
        "user": user,
        "camera": camera,
        "system": system
    }

print("SEQUENCES:", len(seq_info))

# =====================================================
# SPADE PARSER
# =====================================================

def parse_line(line):
    seq = []
    cur = []

    for t in line.strip().split():
        if t == "-2":
            break

        if t == "-1":
            if cur:
                seq.append(int(cur[0]))
            cur = []
        else:
            cur.append(t)

    return seq

# =====================================================
# CHOOSE RULES
# =====================================================

def choose_rules(filename):
    if "neutral_True" in filename:
        return RULES_NEUTRAL_TRUE
    return RULES_NEUTRAL_FALSE

# =====================================================
# PROCESS FILES
# =====================================================

for file in sorted(os.listdir(SPADE_DIR)):

    if not file.startswith("spade_"):
        continue

    if not file.endswith(".txt"):
        continue

    path = os.path.join(SPADE_DIR, file)

    print("\n================================")
    print("PROCESSING:", file)
    print("================================")

    rules = choose_rules(file)

    rule_map = {
        rule: idx
        for idx, rule in enumerate(rules)
    }

    with open(path, encoding="utf8") as f:
        lines = [l.strip() for l in f if l.strip()]

    # =================================================
    # COLLECT RULE OCCURRENCES
    # =================================================

    candidates = {}

    for seq_id, line in enumerate(lines):

        seq = parse_line(line)

        if seq_id not in seq_info:
            continue

        user = seq_info[seq_id]["user"]
        camera = seq_info[seq_id]["camera"]
        system = seq_info[seq_id]["system"]

        for i in range(len(seq) - 1):

            pair = (seq[i], seq[i + 1])

            if pair not in rule_map:
                continue

            rule_id = rule_map[pair]

            od = i * WINDOW_SIZE
            do = (i + 1) * WINDOW_SIZE

            key = (
                rule_id,
                user,
                od,
                do
            )

            if key not in candidates:
                candidates[key] = {
                    "combos": set(),
                    "seq_ids": set(),
                    "debug": []
                }

            candidates[key]["combos"].add((camera, system))
            candidates[key]["seq_ids"].add(seq_id)

            candidates[key]["debug"].append({
                "seq_id": seq_id,
                "user": user,
                "camera": camera,
                "system": system,
                "line_idx": i
            })

    print("RAW RULE WINDOWS:", len(candidates))

    # =================================================
    # MAJORITY FILTER
    # =================================================

    rows = []

    for key, data in candidates.items():

        support = len(data["combos"])

        if support < MIN_SUPPORT_COMBINATIONS:
            continue

        rule_id, user, od, do = key
        rule = rules[rule_id]

        # =================================================
        # DEBUG OUTPUT
        # =================================================

        print("\n[SUPPORTED RULE FOUND]")
        print("Rule ID:", rule_id, "Rule:", rule)
        print("User:", user)
        print("Time window:", od, "-", do)
        print("Support:", support)

        for d in data["debug"]:
            print(
                f"  seq_id={d['seq_id']} "
                f"user={d['user']} "
                f"camera={d['camera']} "
                f"system={d['system']} "
                f"line={d['line_idx']}"
            )

        # =================================================
        # SAVE ROW
        # =================================================

        for seq_id in sorted(data["seq_ids"]):

            rows.append([
                rule_id,
                seq_id,
                od,
                do,
                support
            ])

    rows.sort(
        key=lambda x: (
            x[1],
            x[0],
            x[2]
        )
    )

    # =================================================
    # SAVE
    # =================================================

    suffix = file.replace("spade_", "").replace(".txt", "")

    out_path = os.path.join(
        OUT_DIR,
        f"table_rules_majority_{suffix}.csv"
    )

    with open(out_path, "w", newline="", encoding="utf8") as f:

        writer = csv.writer(f)

        writer.writerow([
            "rule_id",
            "seq_id",
            "od",
            "do",
            "support_combinations"
        ])

        writer.writerows(rows)

    print("RULES AFTER FILTER:", len(rows))
    print("SAVED:", out_path)

print("\nDONE")