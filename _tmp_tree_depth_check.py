import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("rule_occurrence_vectors.csv")
wyst = [c for c in df.columns if c.startswith("Wystapienie_")]
feature_cols = wyst + ["from_emotion_id", "to_emotion_id"]
X = df[feature_cols].fillna(0)
y = df["target_class"]
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

configs = [
    {"name": "jak w raporcie", "max_depth": 4, "min_samples_leaf": 5},
    {"name": "glebiej 8", "max_depth": 8, "min_samples_leaf": 5},
    {"name": "bez limitu depth", "max_depth": None, "min_samples_leaf": 5},
    {"name": "pelne drzewo", "max_depth": None, "min_samples_leaf": 1},
]

for cfg in configs:
    name = cfg.pop("name")
    clf = DecisionTreeClassifier(class_weight="balanced", random_state=42, **cfg)
    clf.fit(Xtr, ytr)
    imp = dict(zip(feature_cols, clf.feature_importances_))
    acc = accuracy_score(yte, clf.predict(Xte))
    print(
        f"{name}: depth={clf.get_depth()}, leaves={clf.get_n_leaves()}, acc={acc:.3f}"
    )
    print(
        f"  from_emotion_id={imp['from_emotion_id']:.4f}, "
        f"to_emotion_id={imp['to_emotion_id']:.4f}"
    )
    top = sorted(imp.items(), key=lambda x: -x[1])[:5]
    top_str = ", ".join(f"{k}: {v:.3f}" for k, v in top)
    print(f"  top: {top_str}")
    print()
