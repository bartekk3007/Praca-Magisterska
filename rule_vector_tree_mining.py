import itertools
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text


# =====================================================
# CONFIG
# =====================================================

INPUT_FILE = "rule_occurrence_vectors.csv"
TREE_REPORT = "rule_vector_tree_report.txt"
RULES_REPORT = "rule_vector_association_rules.txt"
RULES_CSV = "rule_vector_association_rules.csv"

TEST_SIZE = 0.25
RANDOM_STATE = 42
MAX_DEPTH = 4
MIN_SAMPLES_LEAF = 5

# Apriori / reguły asocjacyjne
MIN_SUPPORT = 0.001      # min. udział w zbiorze (5%)
MIN_CONFIDENCE = 0.05   # min. confidence A -> C
MAX_ANTECEDENT_SIZE = 4 # max liczba elementów w warunku


# =====================================================
# LOAD
# =====================================================

df = pd.read_csv(INPUT_FILE)

wystapienie_cols = [c for c in df.columns if c.startswith("Wystapienie_")]
feature_cols = wystapienie_cols + ["from_emotion_id", "to_emotion_id"]

X = df[feature_cols].fillna(0)
y = df["target_class"]

print(f"Wczytano wierszy: {len(df)}")
print("Rozkład target_class:")
print(y.value_counts().sort_index())


# =====================================================
# 1. DRZEWO DECYZYJNE
# =====================================================

class_counts = Counter(y)
can_stratify = min(class_counts.values()) >= 2 and len(class_counts) >= 2

if can_stratify:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
else:
    X_train, X_test, y_train, y_test = X, X, y, y

clf = DecisionTreeClassifier(
    max_depth=MAX_DEPTH,
    min_samples_leaf=MIN_SAMPLES_LEAF,
    class_weight="balanced",
    random_state=RANDOM_STATE,
)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
labels = sorted(y.unique())
cm = confusion_matrix(y_test, y_pred, labels=labels)
tree_text = export_text(clf, feature_names=feature_cols)

importance = (
    pd.DataFrame({"feature": feature_cols, "importance": clf.feature_importances_})
    .sort_values("importance", ascending=False)
)

with open(TREE_REPORT, "w", encoding="utf8") as f:
    f.write("DECISION TREE FROM rule_occurrence_vectors.csv\n\n")
    f.write(f"Rows: {len(df)}\n")
    f.write(f"Features: {len(feature_cols)}\n")
    f.write(f"Target: target_class\n")
    f.write(f"Train: {len(X_train)}, Test: {len(X_test)}\n")
    f.write(f"Stratified split: {can_stratify}\n")
    f.write(f"Accuracy: {acc:.4f}\n\n")

    f.write("CLASS DISTRIBUTION\n")
    for cls, cnt in sorted(class_counts.items()):
        f.write(f"  {cls}: {cnt}\n")
    f.write("\n")

    f.write("CONFUSION MATRIX\n")
    f.write(f"labels={labels}\n")
    f.write(np.array2string(cm))
    f.write("\n\n")

    f.write("CLASSIFICATION REPORT\n")
    f.write(
        classification_report(y_test, y_pred, labels=labels, zero_division=0)
    )
    f.write("\n")

    f.write("FEATURE IMPORTANCE\n")
    for _, row in importance.iterrows():
        f.write(f"  {row['feature']}: {row['importance']:.6f}\n")
    f.write("\n")

    f.write("TREE\n")
    f.write(tree_text)

print(f"Drzewo: accuracy={acc:.4f}, zapisano {TREE_REPORT}")


# =====================================================
# 2. REGUŁY ASOCJACYJNE (support, confidence, prior)
# =====================================================

def row_to_items(row):
    """Transakcja: aktywne wystąpienia kamera-system + typ reguły."""
    items = {f"Regula={row['Regula']}"}
    for col in wystapienie_cols:
        if int(row[col]) == 1:
            items.add(col.replace("Wystapienie_", "CS_"))
    items.add(f"TARGET={int(row['target_class'])}")
    return items


transactions = [row_to_items(row) for _, row in df.iterrows()]
n_trans = len(transactions)

# support pojedynczych elementów
item_counts = Counter()
for t in transactions:
    item_counts.update(t)

# kandydaci antecedent: wystąpienia CS + reguła (bez TARGET)
antecedent_candidates = sorted({
    item for item in item_counts
    if not item.startswith("TARGET=")
})

consequent_candidates = sorted({
    item for item in item_counts if item.startswith("TARGET=")
})

def support_of(itemset):
  count = sum(1 for t in transactions if itemset.issubset(t))
  return count / n_trans

rules = []

for r in range(1, MAX_ANTECEDENT_SIZE + 1):
    for antecedent_tuple in itertools.combinations(antecedent_candidates, r):
        antecedent = frozenset(antecedent_tuple)
        ant_support = support_of(antecedent)
        if ant_support < MIN_SUPPORT:
            continue

        ant_count = sum(1 for t in transactions if antecedent.issubset(t))

        for consequent in consequent_candidates:
            if consequent in antecedent:
                continue

            both = antecedent | {consequent}
            both_count = sum(1 for t in transactions if both.issubset(t))
            rule_support = both_count / n_trans
            if rule_support < MIN_SUPPORT:
                continue

            confidence = both_count / ant_count if ant_count else 0.0
            if confidence < MIN_CONFIDENCE:
                continue

            # prior / a priori = P(antecedent) = support warunku
            prior = ant_support

            rules.append({
                "antecedent": "; ".join(sorted(antecedent)),
                "consequent": consequent,
                "support": rule_support,
                "confidence": confidence,
                "prior": prior,
                "count": both_count,
                "antecedent_count": ant_count,
            })

rules_df = pd.DataFrame(rules)
if not rules_df.empty:
    rules_df = rules_df.sort_values(
        ["confidence", "support", "count"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

rules_df.to_csv(RULES_CSV, index=False)

with open(RULES_REPORT, "w", encoding="utf8") as f:
    f.write("ASSOCIATION RULES FROM rule_occurrence_vectors.csv\n\n")
    f.write(f"Transactions: {n_trans}\n")
    f.write(f"MIN_SUPPORT: {MIN_SUPPORT}\n")
    f.write(f"MIN_CONFIDENCE: {MIN_CONFIDENCE}\n")
    f.write(f"MAX_ANTECEDENT_SIZE: {MAX_ANTECEDENT_SIZE}\n\n")

    f.write("DEFINICJE\n")
    f.write("  support(A->C)     = P(A i C) = ile wierszy ma A i C / N\n")
    f.write("  confidence(A->C)  = P(C|A) = support(A,C) / support(A)\n")
    f.write("  prior / a priori  = P(A) = support samego warunku A\n\n")

    f.write(f"Znaleziono reguł: {len(rules_df)}\n\n")

    if rules_df.empty:
        f.write("Brak reguł przy podanych progach.\n")
    else:
        for i, row in rules_df.head(50).iterrows():
            f.write(
                f"{i+1}. {row['antecedent']} => {row['consequent']}\n"
                f"   support={row['support']:.4f}, "
                f"confidence={row['confidence']:.4f}, "
                f"prior={row['prior']:.4f} "
                f"({int(row['count'])}/{n_trans})\n\n"
            )

print(f"Reguły asocjacyjne: {len(rules_df)}, zapisano {RULES_CSV} i {RULES_REPORT}")
print("DONE")
