import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ============================
# Wczytanie danych
# ============================

# Zmień nazwę pliku na swoją
df = pd.read_csv("rule_occurrence_vectors.csv")

# ============================
# Cechy wejściowe
# ============================

features = [
    'Wystapienie_1_DL_FR',
    'Wystapienie_2_DL_LU',
    'Wystapienie_3_DL_XP',
    'Wystapienie_4_DR_FR',
    'Wystapienie_5_DR_LU',
    'Wystapienie_6_DR_XP',
    'Wystapienie_7_UL_FR',
    'Wystapienie_8_UL_LU',
    'Wystapienie_9_UL_XP',
    'Wystapienie_10_UR_FR',
    'Wystapienie_11_UR_LU',
    'Wystapienie_12_UR_XP',
    'ile_regul_wykryto'
]

X = df[features]
y = df["target_class"]

# ============================
# Podział danych
# ============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ============================
# Budowa drzewa
# ============================

tree = DecisionTreeClassifier(
    criterion="gini",
    max_depth=4,
    min_samples_leaf=2,
    random_state=42
)

tree.fit(X_train, y_train)

# ============================
# Predykcja
# ============================

y_pred = tree.predict(X_test)

print("\n===== ACCURACY =====")
print(accuracy_score(y_test, y_pred))

print("\n===== RAPORT =====")
print(classification_report(y_test, y_pred))

print("\n===== MACIERZ POMYŁEK =====")
print(confusion_matrix(y_test, y_pred))

# ============================
# Ważność cech
# ============================

importance = pd.DataFrame({
    "Feature": features,
    "Importance": tree.feature_importances_
}).sort_values("Importance", ascending=False)

print("\n===== WAŻNOŚĆ CECH =====")
print(importance)

importance.to_csv("importance.csv", index=False)

# ============================
# Reguły IF-THEN
# ============================

rules = export_text(tree, feature_names=features)

print("\n===== REGUŁY =====")
print(rules)

with open("drzewo_reguly.txt", "w", encoding="utf-8") as f:
    f.write(rules)

# ============================
# Rysowanie drzewa
# ============================

plt.figure(figsize=(22, 12))

plot_tree(
    tree,
    feature_names=features,
    class_names=[str(c) for c in sorted(df["target_class"].unique())],
    filled=True,
    rounded=True,
    fontsize=9
)

plt.tight_layout()
plt.savefig("drzewo_decyzyjne.png", dpi=300)
plt.show()

print("\nPliki zapisane:")
print(" - importance.csv")
print(" - drzewo_reguly.txt")
print(" - drzewo_decyzyjne.png")