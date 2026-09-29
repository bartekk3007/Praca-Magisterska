import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text


REPORT_INPUT = "emotion_change_report.txt"
DATASET_FILE = "rule_count_tree_from_report_dataset.csv"
REPORT_FILE = "rule_count_tree_from_report_report.txt"

TEST_SIZE = 0.25
RANDOM_STATE = 42

EMOTION_TO_ID = {
    "anger": 1,
    "disgust": 2,
    "happiness": 3,
    "sadness": 4,
    "surprise": 5,
    "neutral": 6,
}

CLASS_LABELS = {
    0: "0",
    1: "1",
    2: "2",
    3: "3_plus",
}


def parse_percent(text):
    return float(text.replace("%", "").strip())


def to_class_label(n_hidden_both):
    return min(int(n_hidden_both), 3)


with open(REPORT_INPUT, "r", encoding="utf8") as f:
    report = f.read()

blocks = re.split(r"\n={60}\n", report)
event_blocks = []
in_events = False
for block in blocks:
    if "EVENT DETAILS" in block:
        in_events = True
        continue
    if in_events and block.strip().startswith("MOMENT "):
        event_blocks.append(block.strip())

rows = []

for block in event_blocks:
    lines = [line.rstrip() for line in block.splitlines() if line.strip()]

    row = {}

    m = re.search(r"^MOMENT (\d+) -> (\d+)$", block, re.M)
    if not m:
        continue
    row["moment"] = int(m.group(1))
    row["next_moment"] = int(m.group(2))

    m = re.search(r"^([A-Z]{2}_[A-Z]{2}): ([a-z]+) -> ([a-z]+)$", block, re.M)
    if not m:
        continue
    row["detected_cs"] = m.group(1)
    row["from_emotion"] = m.group(2)
    row["to_emotion"] = m.group(3)
    row["from_emotion_id"] = EMOTION_TO_ID[row["from_emotion"]]
    row["to_emotion_id"] = EMOTION_TO_ID[row["to_emotion"]]

    m = re.search(r"^MEAN AGREEMENT: ([0-9.]+)%$", block, re.M)
    row["agreement"] = float(m.group(1)) if m else np.nan

    m = re.search(
        r"^Hidden with both active .*: (\d+)/(\d+)$",
        block,
        re.M,
    )
    row["hidden_both_active_count"] = int(m.group(1)) if m else 0
    row["hidden_combo_count"] = int(m.group(2)) if m else 0

    m = re.search(r"^Dominant rule also in: (.+)$", block, re.M)
    dominant_other = m.group(1).strip() if m else "(brak)"
    if dominant_other == "(brak)":
        row["dominant_rule_count"] = 1
    else:
        row["dominant_rule_count"] = 1 + len(
            [x.strip() for x in dominant_other.split(",") if x.strip()]
        )

    m = re.search(
        r"BEFORE CHANGE \(other camera-system, old emotion\)\n"
        r"  [a-z]+ active: ([0-9.]+)% \(\d+/\d+\)\n"
        r"  TOP2 when active: ([0-9.]+)%",
        block,
        re.M,
    )
    row["hidden_before_active_pct"] = float(m.group(1)) if m else np.nan
    row["hidden_before_top2_pct"] = float(m.group(2)) if m else np.nan

    m = re.search(
        r"AFTER CHANGE \(other camera-system, new emotion\)\n"
        r"  [a-z]+ active: ([0-9.]+)% \(\d+/\d+\)\n"
        r"  TOP2 when active: ([0-9.]+)%",
        block,
        re.M,
    )
    row["hidden_after_active_pct"] = float(m.group(1)) if m else np.nan
    row["hidden_after_top2_pct"] = float(m.group(2)) if m else np.nan

    m = re.search(
        r"HIDDEN BOTH ACTIVE \(other cam-sys: [a-z]+ przed \+ [a-z]+ po\)\n"
        r"  Both active: ([0-9.]+)% \((\d+)/(\d+)\)\n"
        r"  [a-z]+ TOP2 when both: ([0-9.]+)%\n"
        r"  [a-z]+ TOP2 when both: ([0-9.]+)%",
        block,
        re.M,
    )
    row["hidden_both_active_pct"] = float(m.group(1)) if m else np.nan
    row["hidden_both_before_top2_pct"] = float(m.group(4)) if m else np.nan
    row["hidden_both_after_top2_pct"] = float(m.group(5)) if m else np.nan

    m = re.search(r"Before distance mean \([^)]+\): ([0-9.]+)", block, re.M)
    row["hidden_both_before_dist_mean"] = float(m.group(1)) if m else -1.0

    m = re.search(r"After distance mean \([^)]+\): ([0-9.]+)", block, re.M)
    row["hidden_both_after_dist_mean"] = float(m.group(1)) if m else -1.0

    row["target_count_raw"] = row["hidden_both_active_count"]
    row["target_class_id"] = to_class_label(row["target_count_raw"])
    row["target_class_label"] = CLASS_LABELS[row["target_class_id"]]

    rows.append(row)

dataset = pd.DataFrame(rows)
dataset.to_csv(DATASET_FILE, index=False)

feature_cols = [
    "from_emotion_id",
    "to_emotion_id",
    "agreement",
    "dominant_rule_count",
    "hidden_combo_count",
    "hidden_before_active_pct",
    "hidden_before_top2_pct",
    "hidden_after_active_pct",
    "hidden_after_top2_pct",
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
    pd.DataFrame({"feature": feature_cols, "importance": clf.feature_importances_})
    .sort_values("importance", ascending=False)
    .reset_index(drop=True)
)

with open(REPORT_FILE, "w", encoding="utf8") as f:
    f.write("DECISION TREE FROM EMOTION CHANGE REPORT\n\n")
    f.write(f"Input report: {REPORT_INPUT}\n")
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
    for _, imp in importance_df.iterrows():
        f.write(f"  {imp['feature']}: {imp['importance']:.6f}\n")
    f.write("\n")

    f.write("TREE\n")
    f.write(tree_text)

print("DONE")
print(f"Saved dataset: {DATASET_FILE}")
print(f"Saved report: {REPORT_FILE}")
print(f"Dataset rows: {len(dataset)}")
print(f"Accuracy: {acc:.4f}")
