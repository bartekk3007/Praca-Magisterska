import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("rule_occurrence_vectors.csv")
wyst = [c for c in df.columns if c.startswith("Wystapienie_")]
for c in wyst:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

df = df.dropna(subset=["target_class"])
y = df["target_class"].astype(int)
X = df[wyst]

short = ["_".join(c.split("_")[2:]) for c in wyst]
X.columns = short

Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

imp4 = pd.Series(
    DecisionTreeClassifier(
        max_depth=4, min_samples_leaf=5, class_weight="balanced", random_state=42
    )
    .fit(Xtr, ytr)
    .feature_importances_,
    index=short,
)
imp8 = pd.Series(
    DecisionTreeClassifier(
        max_depth=8, min_samples_leaf=5, class_weight="balanced", random_state=42
    )
    .fit(Xtr, ytr)
    .feature_importances_,
    index=short,
)
impU = pd.Series(
    DecisionTreeClassifier(
        max_depth=None, min_samples_leaf=5, class_weight="balanced", random_state=42
    )
    .fit(Xtr, ytr)
    .feature_importances_,
    index=short,
)

rows = []
for s in short:
    v = X[s] > 0
    p1 = 100 * v[y == 1].mean()
    p2 = 100 * v[y == 2].mean()
    p3 = 100 * v[y == 3].mean()
    rows.append(
        {
            "kamera_system": s,
            "imp_depth4": round(imp4[s], 4),
            "imp_depth8": round(imp8[s], 4),
            "imp_unlimited": round(impU[s], 4),
            "imp_avg": round((imp4[s] + imp8[s] + impU[s]) / 3, 4),
            "pct_klasa1": round(p1, 1),
            "pct_klasa2": round(p2, 1),
            "pct_klasa3": round(p3, 1),
            "delta_3_vs_1": round(p3 - p1, 1),
        }
    )

out = pd.DataFrame(rows).sort_values("imp_avg", ascending=False).reset_index(drop=True)
out.index = out.index + 1
print(out.to_string())
print()
print("Ranking (srednia importance depth4 / depth8 / unlimited):")
for i, r in out.iterrows():
    print(
        f"{i:2d}. {r['kamera_system']:6s}  "
        f"imp={r['imp_avg']:.3f}  "
        f"klasa1={r['pct_klasa1']:.0f}% | "
        f"2={r['pct_klasa2']:.0f}% | "
        f"3={r['pct_klasa3']:.0f}%  "
        f"d3-1={r['delta_3_vs_1']:+.0f}pp"
    )
