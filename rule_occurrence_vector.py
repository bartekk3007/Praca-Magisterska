import pandas as pd
import numpy as np


# =====================================================
# CONFIG
# =====================================================

INPUT_FILE = "emotion_values_P01_time_sorted.csv"
OUTPUT_FILE = "rule_occurrence_vectors.csv"

NROWS = None          # None = cały plik
PROGRESS_EVERY = 500

# format pod drzewo decyzyjne: 1 wiersz = moment + reguła (wykryta przez dominant)
# cechy: Wystapienie_<kamera_system> (0/1, emocje ponad próg)
# target: ile_regul_wykryto lub target_class (0/1/2/3_plus)

THRESHOLD = 0.20

EMOTIONS = [
    "anger",
    "disgust",
    "happiness",
    "sadness",
    "surprise",
    "neutral",
]

# pary (id_stara, id_nowa) – jak w pozostałych skryptach
RULES = [
    (5, 6), (6, 5),
    (4, 6), (6, 4),
    (3, 6), (6, 3),
    (2, 6), (6, 2),
    (1, 6), (6, 1),
]

CAM_SYSTEMS = [
    "DL_FR", "DL_LU", "DL_XP",
    "DR_FR", "DR_LU", "DR_XP",
    "UL_FR", "UL_LU", "UL_XP",
    "UR_FR", "UR_LU", "UR_XP",
]

ID_TO_EMO = {i + 1: e for i, e in enumerate(EMOTIONS)}
RULE_PAIRS = [(ID_TO_EMO[e1], ID_TO_EMO[e2]) for e1, e2 in RULES]


# =====================================================
# HELPERS
# =====================================================

def parse_active(row):
    raw = row.get("active_emotions", "")
    if pd.isna(raw) or raw == "":
        return set()
    return {e.strip() for e in str(raw).split(";") if e.strip()}


def emotion_active(row, emotion):
    """Emocja aktywna: w active_emotions lub wartość >= próg."""
    if emotion in parse_active(row):
        return True
    val = row.get(emotion)
    if pd.isna(val):
        return False
    return float(val) >= THRESHOLD


def rule_occurred_active(row_before, row_after, emo_old, emo_new):
    """Wystąpienie w wektorze: stara emocja aktywna w n, nowa aktywna w n+1 (ponad próg)."""
    return (
        emotion_active(row_before, emo_old)
        and emotion_active(row_after, emo_new)
    )


def dominant_rule_detected(row_before, row_after, emo_old, emo_new):
    """Reguła wykryta przez dominant_emotion: stara w n, nowa w n+1."""
    return (
        row_before["dominant_emotion"] == emo_old
        and row_after["dominant_emotion"] == emo_new
    )


def to_target_class(count):
    return min(int(count), 3)


def rule_label(emo_old, emo_new):
    return f"{emo_old}->{emo_new}"


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
moment_list = sorted(moments)

print(f"Wczytano wierszy: {len(df)}")
print(f"Momenty: {df['moment'].min()} - {df['moment'].max()} ({len(moments)} unikalnych)")
print(f"Próg aktywności: {THRESHOLD}")
print("Budowanie wektorów reguł...")


# =====================================================
# MAIN
# =====================================================

rows = []
processed_pairs = 0

for moment in moment_list:
    if moment + 1 not in moments:
        continue

    processed_pairs += 1
    if processed_pairs % PROGRESS_EVERY == 0:
        print(f"  przetworzono par momentów: {processed_pairs}, wierszy wektora: {len(rows)}")

    now = moments[moment]
    after = moments[moment + 1]

    now_map = {r.cam_sys: r for _, r in now.iterrows()}
    after_map = {r.cam_sys: r for _, r in after.iterrows()}

    for emo_old, emo_new in RULE_PAIRS:
        # wiersz tylko gdy reguła wykryta przez dominant w >=1 kombinacji
        dominant_any = False

        for cs in CAM_SYSTEMS:
            row_before = now_map.get(cs)
            row_after = after_map.get(cs)
            if row_before is None or row_after is None:
                continue
            if dominant_rule_detected(row_before, row_after, emo_old, emo_new):
                dominant_any = True
                break

        if not dominant_any:
            continue

        occurrences = {}
        total_active = 0

        for i, cs in enumerate(CAM_SYSTEMS, start=1):
            row_before = now_map.get(cs)
            row_after = after_map.get(cs)

            if row_before is None or row_after is None:
                occurred = 0
            else:
                occurred = int(rule_occurred_active(row_before, row_after, emo_old, emo_new))

            occurrences[cs] = occurred
            total_active += occurred

        row = {
            "ID_Moment": int(moment),
            "Regula": rule_label(emo_old, emo_new),
            "from_emotion": emo_old,
            "to_emotion": emo_new,
            "from_emotion_id": EMOTIONS.index(emo_old) + 1,
            "to_emotion_id": EMOTIONS.index(emo_new) + 1,
        }

        for i, cs in enumerate(CAM_SYSTEMS, start=1):
            row[f"Wystapienie_{i}_{cs}"] = occurrences[cs]

        row["ile_regul_wykryto"] = total_active
        row["target_class"] = to_target_class(total_active)
        rows.append(row)


# =====================================================
# SAVE
# =====================================================

out_df = pd.DataFrame(rows)

# kolejność kolumn
base_cols = [
    "ID_Moment", "Regula", "from_emotion", "to_emotion",
    "from_emotion_id", "to_emotion_id",
]
wystapienie_cols = [
    f"Wystapienie_{i}_{cs}" for i, cs in enumerate(CAM_SYSTEMS, start=1)
]
out_df = out_df[base_cols + wystapienie_cols + ["ile_regul_wykryto", "target_class"]]

out_df.to_csv(OUTPUT_FILE, index=False)

print(
    f"Koniec: par momentów={processed_pairs}, "
    f"wierszy wektora (moment+reguła)={len(out_df)}"
)
print(f"Zapisano: {OUTPUT_FILE}")
