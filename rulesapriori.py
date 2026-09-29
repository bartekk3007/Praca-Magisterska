import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Wczytanie danych
df = pd.read_csv("rule_occurrence_vectors.csv")

# Wybór kolumn binarnych
cols = [c for c in df.columns if c.startswith("Wystapienie_")]

# Zamiana 0/1 na True/False
basket = df[cols].astype(bool)

# Dodajemy Regula jako element transakcji
for r in df["Regula"].unique():
    basket[f"Regula={r}"] = (df["Regula"] == r)

# Dodajemy target_class jako element transakcji
for t in df["target_class"].unique():
    basket[f"target={t}"] = (df["target_class"] == t)

# Apriori
frequent = apriori(basket, min_support=0.001, use_colnames=True)

# Reguły asocjacyjne
rules = association_rules(
    frequent,
    metric="confidence",
    min_threshold=0.5
)

print(rules[["antecedents", "consequents", "support", "confidence", "lift"]])

# Czytelniejsza wersja reguł
rules_out = rules.copy()

rules_out["antecedents"] = rules_out["antecedents"].apply(
    lambda x: ", ".join(sorted(list(x)))
)

rules_out["consequents"] = rules_out["consequents"].apply(
    lambda x: ", ".join(sorted(list(x)))
)

rules_out = rules_out[
    ["antecedents", "consequents", "support", "confidence", "lift"]
]

rules_out = rules_out.sort_values(by=["support", "confidence"], ascending=False)
rules_out.to_csv("apriori_association_rules.csv", index=False, encoding="utf-8-sig")

print("Plik association_rules.csv został zapisany.")