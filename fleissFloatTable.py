import os
import pandas as pd
from statsmodels.stats.inter_rater import fleiss_kappa
import numpy as np
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

# =========================================================
# CONFIG
# =========================================================

SPADE_DIR = "spade_output"

USE_NEUTRAL = True
USE_EMPTY = True

suffix = f"neutral_{USE_NEUTRAL}_empty_{USE_EMPTY}"

RULE_FILE = os.path.join(
    SPADE_DIR,
    f"table_rules_majority_{suffix}.csv"
)

META_FILE = os.path.join(
    SPADE_DIR,
    f"meta_{suffix}.csv"
)

OUT_FILE = os.path.join(
    SPADE_DIR,
    f"fleiss_input_{suffix}.csv"
)

# =========================================================
# LOAD
# =========================================================

rules = pd.read_csv(RULE_FILE)
meta = pd.read_csv(META_FILE)

print("rules:", len(rules))
print("meta :", len(meta))

# =========================================================
# MERGE RULES + META
# =========================================================

merged = rules.merge(
    meta,
    on="seq_id",
    how="left"
)

# =========================================================
# zachowujemy tylko reguły które naprawdę
# wystąpiły w momencie odpowiadającym 'od'
# =========================================================

merged = merged[
    merged["moment"] == merged["od"]
].copy()

print("matched:", len(merged))

# =========================================================
# JUDGE NAME
# =========================================================

merged["judge"] = (
    merged["camera"].astype(str)
    + "_"
    + merged["system"].astype(str)
)

# =========================================================
# 12 sędziów
# =========================================================

judges = [
    "DL_FR",
    "DL_LU",
    "DL_XP",
    "DR_FR",
    "DR_LU",
    "DR_XP",
    "UL_FR",
    "UL_LU",
    "UL_XP",
    "UR_FR",
    "UR_LU",
    "UR_XP"
]

# =========================================================
# BUILD FLEISS TABLE
# =========================================================

rows = {}

for _, row in merged.iterrows():

    key = (
        int(row["rule_id"]),
        int(row["moment"]),
        str(row["user"])
    )

    if key not in rows:

        rows[key] = {
            "rule_id": int(row["rule_id"]),
            "moment": int(row["moment"]),
            "user": str(row["user"])
        }

        for j in judges:
            rows[key][j] = 0

    judge = row["judge"]

    if judge in rows[key]:
        rows[key][judge] = 1

# =========================================================
# DATAFRAME
# =========================================================

fleiss = pd.DataFrame(rows.values())

fleiss = fleiss.sort_values(
    ["user", "rule_id", "moment"]
)

judge_cols = [
    "DL_FR","DL_LU","DL_XP",
    "DR_FR","DR_LU","DR_XP",
    "UL_FR","UL_LU","UL_XP",
    "UR_FR","UR_LU","UR_XP"
]

# liczba zer w każdej linii
fleiss["num_zeros"] = (fleiss[judge_cols] == 0).sum(axis=1)

# opcjonalnie: liczba jedynek
fleiss["num_ones"] = (fleiss[judge_cols] == 1).sum(axis=1)

print(fleiss[["rule_id", "moment", "user", "num_zeros", "num_ones"]].head())

# =========================================================
# SAVE
# =========================================================

fleiss.to_csv(
    OUT_FILE,
    index=False
)

print()
print("DONE")
print("Rows:", len(fleiss))
print("Saved:", OUT_FILE)

judge_cols = [
    "DL_FR","DL_LU","DL_XP",
    "DR_FR","DR_LU","DR_XP",
    "UL_FR","UL_LU","UL_XP",
    "UR_FR","UR_LU","UR_XP"
]

ratings = fleiss[judge_cols].to_numpy(dtype=np.int32)

# Fleiss matrix: [no, yes]
table = np.zeros((len(ratings), 2), dtype=np.int32)

table[:, 1] = ratings.sum(axis=1)
table[:, 0] = len(judge_cols) - table[:, 1]

kappa = fleiss_kappa(table)

# =========================================================
# SAVE RESULT
# =========================================================

kappa_file = OUT_FILE.replace(".csv", "_kappa.txt")

with open(kappa_file, "w") as f:
    f.write(f"Fleiss Kappa: {kappa:.10f}\n")
    f.write(f"Rows: {len(fleiss)}\n")
    f.write(f"Judges: {len(judge_cols)}\n")

print("\nFLEISS KAPPA:", kappa)
print("Saved:", kappa_file)

# =========================================================
# FLEISS KAPPA PER RULE
# =========================================================

rules_kappa = []

for rule_id in sorted(fleiss["rule_id"].unique()):
    df_r = fleiss[fleiss["rule_id"] == rule_id]
    if len(df_r) == 0:
        continue
    ratings_r = df_r[judge_cols].to_numpy(dtype=np.int32)
    # Fleiss format: [no, yes]
    table_r = np.zeros((len(ratings_r), 2), dtype=np.int32)
    table_r[:, 1] = ratings_r.sum(axis=1)
    table_r[:, 0] = len(judge_cols) - table_r[:, 1]
    # jeśli brak wariancji → κ nie ma sensu
    if np.all(table_r[:, 1] == 0) or np.all(table_r[:, 0] == 0):
        kappa_r = 0.0
    else:
        kappa_r = fleiss_kappa(table_r)
    rules_kappa.append([
        rule_id,
        kappa_r,
        len(df_r),
        df_r[judge_cols].sum().sum()
    ])

rules_kappa_df = pd.DataFrame(
    rules_kappa,
    columns=["rule_id", "fleiss_kappa", "n_rows", "total_positives"]
)

# sortowanie: najlepsze reguły na górze
rules_kappa_df = rules_kappa_df.sort_values(
    "fleiss_kappa",
    ascending=False
)

# =========================================================
# SAVE PER-RULE KAPPA
# =========================================================

rules_kappa_file = OUT_FILE.replace(".csv", "_per_rule_kappa.csv")
rules_kappa_df.to_csv(rules_kappa_file, index=False)
print("\nPER-RULE FLEISS KAPPA SAVED:", rules_kappa_file)
print(rules_kappa_df.head(10))


participant_results = []
for participant in fleiss["user"].unique():
    df_p = fleiss[fleiss["user"] == participant]
    if len(df_p) == 0:
        continue
    ratings = df_p[judge_cols].to_numpy(dtype=np.int32)
    table = np.zeros((len(ratings), 2))
    table[:, 1] = ratings.sum(axis=1)
    table[:, 0] = len(judge_cols) - table[:, 1]
    if np.all(table[:, 1] == 0) or np.all(table[:, 0] == 0):
        kappa = 0.0
    else:
        kappa = fleiss_kappa(table)
    participant_results.append([
        participant,
        kappa,
        len(df_p),
        df_p[judge_cols].sum().sum()
    ])

participant_df = pd.DataFrame(
    participant_results,
    columns=["participant", "fleiss_kappa", "n_rows", "total_positives"]
)
participant_df = participant_df.sort_values("fleiss_kappa", ascending=False)
out_part_file = OUT_FILE.replace(".csv", "_per_participant_kappa.csv")
participant_df.to_csv(out_part_file, index=False)
print("\nPER-PARTICIPANT KAPPA SAVED:", out_part_file)
print(participant_df)

# =========================================================
# FLEISS KAPPA BY PARTICIPANT FEATURES (FIXED)
# =========================================================

chars_path = os.path.join(SPADE_DIR, "participantsCharacteristics.csv")
chars = pd.read_csv(chars_path)

fleiss_feat = fleiss.merge(
    chars,
    left_on="user",
    right_on="participant",
    how="left"
)

# =========================================================
# FULL FEATURE SPACE (IMPORTANT)
# =========================================================

feature_levels = {
    "gender": ["Male", "Female"],
    "beard": ["No", "Some", "Heavy"],
    "moustache": ["No", "Some"],
    "glasses": ["Yes", "No"]
}

feature_cols = list(feature_levels.keys())

# =========================================================
# SAFE KAPPA FUNCTION
# =========================================================

def compute_kappa(df_sub):
    if len(df_sub) == 0:
        return np.nan

    ratings = df_sub[judge_cols].to_numpy(dtype=np.int32)

    table = np.zeros((len(ratings), 2), dtype=np.int32)
    table[:, 1] = ratings.sum(axis=1)
    table[:, 0] = len(judge_cols) - table[:, 1]

    if np.all(table[:, 1] == 0) or np.all(table[:, 0] == 0):
        return 0.0

    return fleiss_kappa(table)

# =========================================================
# COMPUTE FEATURE TABLE
# =========================================================

feature_results = []

for feat in feature_cols:

    for value in feature_levels[feat]:

        df_sub = fleiss_feat[fleiss_feat[feat] == value]

        kappa = compute_kappa(df_sub)

        feature_results.append([
            feat,
            value,
            kappa,
            len(df_sub),
            int(df_sub[judge_cols].sum().sum()) if len(df_sub) > 0 else 0
        ])

feature_df = pd.DataFrame(
    feature_results,
    columns=[
        "feature",
        "value",
        "fleiss_kappa",
        "n_rows",
        "total_positives"
    ]
)

feature_df = feature_df.sort_values(
    ["feature", "fleiss_kappa"],
    ascending=[True, False]
)

# =========================================================
# SAVE
# =========================================================

out_feat_file = OUT_FILE.replace(".csv", "_kappa_by_features.csv")
feature_df.to_csv(out_feat_file, index=False)

print("\nFEATURE KAPPA SAVED:", out_feat_file)
print(feature_df)

# GENEROWANIE DRZEWA
tree_df = feature_df.copy()
tree_df = tree_df.dropna(subset=["fleiss_kappa"])
X = tree_df[["feature", "value"]]
y = tree_df["fleiss_kappa"].astype(float)
encoder = OneHotEncoder(sparse_output=False)
X_encoded = encoder.fit_transform(X)
feature_names = encoder.get_feature_names_out(["feature", "value"])
X_encoded = pd.DataFrame(X_encoded, columns=feature_names)
tree_model = DecisionTreeRegressor(
    max_depth=4,
    min_samples_leaf=2,
    random_state=42
)
tree_model.fit(X_encoded, y)
importance = pd.DataFrame({
    "feature": feature_names,
    "importance": tree_model.feature_importances_
}).sort_values("importance", ascending=False)
importance_file = os.path.join(SPADE_DIR, f"tree_importance_{suffix}.csv")
importance.to_csv(importance_file, index=False)
print("\nTOP FEATURES:")
print(importance.head(10))
plt.figure(figsize=(22, 10))
plot_tree(
    tree_model,
    feature_names=feature_names,
    filled=True,
    rounded=True,
    fontsize=10
)
tree_file = os.path.join(SPADE_DIR, f"tree_kappa_{suffix}.png")
plt.savefig(tree_file, dpi=300, bbox_inches="tight")
plt.close()
print("\nTREE SAVED:", tree_file)