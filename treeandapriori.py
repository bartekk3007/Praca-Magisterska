import pandas as pd
from itertools import combinations

# -------------------------------
# Wczytanie danych
# -------------------------------

df = pd.read_csv("rule_occurrence_vectors.csv")

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
    'Wystapienie_12_UR_XP'
]

N = len(df)

wyniki = []

# -------------------------------------------------------
# Sprawdzamy kombinacje długości 1,2,3
# -------------------------------------------------------

for r in [1,2,3]:

    for combo in combinations(features, r):

        # rekordy spełniające wszystkie reguły
        mask = (df[list(combo)] == 1).all(axis=1)

        liczba_combo = mask.sum()

        if liczba_combo == 0:
            continue

        support = liczba_combo / N

        # pomijamy bardzo rzadkie kombinacje
        if support < 0.05:
            continue

        # dla każdej klasy liczymy confidence i lift
        for target in sorted(df.target_class.unique()):
            mask_target = df.target_class == target
            wspolne = (mask & mask_target).sum()
            if wspolne == 0:
                continue
            confidence = wspolne / liczba_combo
            support_target = mask_target.mean()
            lift = confidence / support_target
            wyniki.append({
                "reguly": ", ".join(combo),
                "target_class": target,
                "support": round(support,4),
                "confidence": round(confidence,4),
                "lift": round(lift,4),
                "liczba_przypadkow": wspolne
            })

# --------------------------------------
# Wyniki
# --------------------------------------

wyniki = pd.DataFrame(wyniki)

wyniki = wyniki.sort_values(
    ["lift","confidence"],
    ascending=False
)

print(wyniki)

wyniki.to_csv("rules_association.csv", index=False)

print("\nZapisano rules_association.csv")