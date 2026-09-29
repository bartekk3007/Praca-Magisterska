import pandas as pd
import numpy as np
from collections import Counter

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text


# =====================================================
# CONFIG
# =====================================================

INPUT_FILE = "emotion_values_P01_time_sorted.csv"

DATASET_FILE = "rule_count_tree_dataset.csv"
REPORT_FILE = "rule_count_tree_report.txt"

NROWS = None           # None = cały plik
THRESHOLD = 0.20
TEST_SIZE = 0.25
RANDOM_STATE = 42

EMOTIONS = [
    "anger",
    "disgust",
    "happiness",
    "sadness",
    "surprise",
    "neutral",
]

RULES = {
    (5, 6), (6, 5),
    (4, 6), (6, 4),
    (3, 6), (6, 3),
    (2, 6), (6, 2),
    (1, 6), (6, 1),
}

CLASS_LABELS = {
    0: "0",
    1: "1",
    2: "2",
    3: "3_plus",
}


# =====================================================
# HELPERS
# =====================================================

def parse_active(row):
    raw = row.get("active_emotions", "")
    if pd.isna(raw) or raw == "":
        return set()
    return {e.strip() for e in str(raw).split(";") if e.strip()}


def emotion_active(row, emotion):
    if emotion in parse_active(row):
        return True
    return float(row[emotion]) >= THRESHOLD


def emotion_top2(row, emotion):
    values = [float(row[e]) for e in EMOTIONS]
    target = float(row[emotion])
    top2 = sorted(values, reverse=True)[:2]
    return target in top2


def distance_not_dominant(row, emotion):
    if not emotion_active(row, emotion):
        return None
    if row["dominant_emotion"] == emotion:
        return None
    values = [float(row[e]) for e in EMOTIONS]
    return max(values) - float(row[emotion])


def pct(count, total):
    if total == 0:
        return 0.0
    return count / total * 100


def same_rule_at_combo(row_before, row_after, emo_old, emo_new):
    return (
        row_before["dominant_emotion"] == emo_old
        and row_after["dominant_emotion"] == emo_new
    )


def combos_with_same_rule(now_map, after_map, emo_old, emo_new):
    fired = []
    for cs in now_map:
        if cs not in after_map:
            continue
        if same_rule_at_combo(now_map[cs], after_map[cs], emo_old, emo_new):
            fired.append(cs)
    return fired


def summarize_rows(rows, emotion):
    n = len(rows)
    active_count = 0
    top2_count = 0
    distances = []

    for _, row in rows:
        active = emotion_active(row, emotion)
        if active:
            active_count += 1
            if emotion_top2(row, emotion):
                top2_count += 1
        dist = distance_not_dominant(row, emotion)
        if dist is not None:
            distances.append(dist)

    return {
        "n": n,
        "active_count": active_count,
        "active_pct": pct(active_count, n),
        "top2_when_active_pct": pct(top2_count, active_count) if active_count else 0.0,
        "dist_mean": float(np.mean(distances)) if distances else -1.0,
        "dist_std": float(np.std(distances)) if distances else -1.0,
    }


def summarize_hidden_both_active(pairs, emo_old, emo_new):
    n = len(pairs)
    both_count = 0
    before_top2_count = 0
    after_top2_count = 0
    before_distances = []
    after_distances = []

    for _, row_before, row_after in pairs:
        old_active = emotion_active(row_before, emo_old)
        new_active = emotion_active(row_after, emo_new)

        if not (old_active and new_active):
            continue

        both_count += 1
        if emotion_top2(row_before, emo_old):
            before_top2_count += 1
        if emotion_top2(row_after, emo_new):
            after_top2_count += 1

        old_dist = distance_not_dominant(row_before, emo_old)
        new_dist = distance_not_dominant(row_after, emo_new)
        if old_dist is not None:
            before_distances.append(old_dist)
        if new_dist is not None:
            after_distances.append(new_dist)

    return {
        "n": n,
        "both_count": both_count,
        "both_pct": pct(both_count, n),
        "before_top2_when_both_pct": pct(before_top2_count, both_count) if both_count else 0.0,
        "after_top2_when_both_pct": pct(after_top2_count, both_count) if both_count else 0.0,
        "before_dist_mean": float(np.mean(before_distances)) if before_distances else -1.0,
        "after_dist_mean": float(np.mean(after_distances)) if after_distances else -1.0,
    }


def to_class_label(n_hidden_both):
    return min(int(n_hidden_both), 3)


# =====================================================
# LOAD
# =====================================================

read_kwargs = {}
if NROWS is not None:
    read_kwargs["nrows"] = NROWS

df = pd.read_csv(INPUT_FILE, **read_kwargs)
df["cam_sys"] = df["camera"] + "_" + df["system"]
df = df.sort_values(["moment", "cam_sys"])
moments = {m: g for m, g in df.groupby("moment")}

print(f"Wczytano wierszy: {len(df)}")
print(f"Momenty: {df['moment'].min()} - {df['moment'].max()} ({len(moments)} unikalnych)")


# =====================================================
# BUILD DATASET: one row = unique moment + rule
# =====================================================

rows = []

for moment in sorted(moments):
    if moment + 1 not in moments:
        continue

    now = moments[moment]
    after = moments[moment + 1]

    now_map = {r.cam_sys: r for _, r in now.iterrows()}
    after_map = {r.cam_sys: r for _, r in after.iterrows()}

    rules_seen_this_step = set()

    for cs, row_before in now_map.items():
        row_after = after_map.get(cs)
        if row_after is None:
            continue

        emo_old = row_before["dominant_emotion"]
        emo_new = row_after["dominant_emotion"]

        if emo_old not in EMOTIONS or emo_new not in EMOTIONS:
            continue
        if emo_old == emo_new:
            continue

        e1 = EMOTIONS.index(emo_old) + 1
        e2 = EMOTIONS.index(emo_new) + 1
        if (e1, e2) not in RULES:
            continue

        rule_key = (emo_old, emo_new)
        if rule_key in rules_seen_this_step:
            continue
        rules_seen_this_step.add(rule_key)

        fired_dominant = combos_with_same_rule(now_map, after_map, emo_old, emo_new)
        if not fired_dominant:
            continue

        hidden_cs = [
            other_cs for other_cs in now_map
            if other_cs not in fired_dominant and other_cs in after_map
        ]

        hidden_before_rows = [(other_cs, now_map[other_cs]) for other_cs in hidden_cs]
        hidden_after_rows = [(other_cs, after_map[other_cs]) for other_cs in hidden_cs]
        hidden_pairs = [
            (other_cs, now_map[other_cs], after_map[other_cs])
            for other_cs in hidden_cs
        ]

        hb = summarize_rows(hidden_before_rows, emo_old)
        ha = summarize_rows(hidden_after_rows, emo_new)
        hboth = summarize_hidden_both_active(hidden_pairs, emo_old, emo_new)

        agreement = sum(
            1 for _, r in after.iterrows() if r["dominant_emotion"] == emo_new
        ) / 12

        target_raw = hboth["both_count"]
        target_class = to_class_label(target_raw)

        rows.append({
            "moment": int(moment),
            "from_emotion": emo_old,
            "to_emotion": emo_new,
            "from_emotion_id": e1,
            "to_emotion_id": e2,
            "agreement": agreement,
            "dominant_rule_count": len(fired_dominant),
            "hidden_combo_count": len(hidden_cs),
            "hidden_before_active_pct": hb["active_pct"],
            "hidden_before_top2_pct": hb["top2_when_active_pct"],
            "hidden_before_dist_mean": hb["dist_mean"],
            "hidden_before_dist_std": hb["dist_std"],
            "hidden_after_active_pct": ha["active_pct"],
            "hidden_after_top2_pct": ha["top2_when_active_pct"],
            "hidden_after_dist_mean": ha["dist_mean"],
            "hidden_after_dist_std": ha["dist_std"],
            "hidden_both_active_count": hboth["both_count"],
            "hidden_both_active_pct": hboth["both_pct"],
            "hidden_both_before_top2_pct": hboth["before_top2_when_both_pct"],
            "hidden_both_after_top2_pct": hboth["after_top2_when_both_pct"],
            "hidden_both_before_dist_mean": hboth["before_dist_mean"],
            "hidden_both_after_dist_mean": hboth["after_dist_mean"],
            "target_count_raw": target_raw,
            "target_class_id": target_class,
            "target_class_label": CLASS_LABELS[target_class],
        })


dataset = pd.DataFrame(rows)
dataset.to_csv(DATASET_FILE, index=False)

if dataset.empty:
    raise ValueError("Brak danych do uczenia drzewa.")


# =====================================================
# TRAIN DECISION TREE
# =====================================================

feature_cols = [
    "from_emotion_id",
    "to_emotion_id",
    "agreement",
    "dominant_rule_count",
    "hidden_combo_count",
    "hidden_before_active_pct",
    "hidden_before_top2_pct",
    "hidden_before_dist_mean",
    "hidden_before_dist_std",
    "hidden_after_active_pct",
    "hidden_after_top2_pct",
    "hidden_after_dist_mean",
    "hidden_after_dist_std",
    "hidden_both_active_pct",
    "hidden_both_before_top2_pct",
    "hidden_both_after_top2_pct",
    "hidden_both_before_dist_mean",
    "hidden_both_after_dist_mean",
]

X = dataset[feature_cols].fillna(-1.0)
y = dataset["target_class_id"]

class_counts = Counter(y)
can_stratify = min(class_counts.values()) >= 2 and len(class_counts) >= 2

if can_stratify:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
else:
    X_train, X_test, y_train, y_test = X, X, y, y

clf = DecisionTreeClassifier(
    max_depth=4,
    min_samples_leaf=20,
    class_weight="balanced",
    random_state=RANDOM_STATE,
)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3])
tree_text = export_text(clf, feature_names=feature_cols)

importance_df = (
    pd.DataFrame({
        "feature": feature_cols,
        "importance": clf.feature_importances_,
    })
    .sort_values("importance", ascending=False)
    .reset_index(drop=True)
)


# =====================================================
# REPORT
# =====================================================

with open(REPORT_FILE, "w", encoding="utf8") as f:
    f.write("DECISION TREE FOR RULE-COUNT CLASSES\n\n")
    f.write(f"Input: {INPUT_FILE}\n")
    if NROWS is not None:
        f.write(f"Rows loaded: {NROWS}\n")
    f.write(f"Threshold: {THRESHOLD}\n")
    f.write(f"Dataset rows: {len(dataset)}\n")
    f.write("Target classes: 0, 1, 2, 3_plus\n\n")

    f.write("CLASS DISTRIBUTION\n")
    for class_id in [0, 1, 2, 3]:
        count = int((y == class_id).sum())
        f.write(f"  {CLASS_LABELS[class_id]}: {count}\n")
    f.write("\n")

    f.write("TRAIN / TEST\n")
    f.write(f"  train rows: {len(X_train)}\n")
    f.write(f"  test rows: {len(X_test)}\n")
    f.write(f"  stratified split: {can_stratify}\n")
    f.write(f"  accuracy: {acc:.4f}\n\n")

    f.write("CONFUSION MATRIX\n")
    f.write("rows=true, cols=pred, labels=[0,1,2,3_plus]\n")
    f.write(np.array2string(cm))
    f.write("\n\n")

    f.write("CLASSIFICATION REPORT\n")
    f.write(
        classification_report(
            y_test,
            y_pred,
            labels=[0, 1, 2, 3],
            target_names=[CLASS_LABELS[i] for i in [0, 1, 2, 3]],
            zero_division=0,
        )
    )
    f.write("\n")

    f.write("FEATURE IMPORTANCE\n")
    for _, row in importance_df.iterrows():
        f.write(f"  {row['feature']}: {row['importance']:.6f}\n")
    f.write("\n")

    f.write("TREE\n")
    f.write(tree_text)

print("DONE")
print(f"Saved dataset: {DATASET_FILE}")
print(f"Saved report: {REPORT_FILE}")
print(f"Dataset rows: {len(dataset)}")
print(f"Accuracy: {acc:.4f}")
