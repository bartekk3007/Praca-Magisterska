import pandas as pd
from pathlib import Path
import re
import logging
import numpy as np


log = logging.Logger(name='rulemining', level=logging.INFO)


log.info("SSSSS")
# ==========================================
# KONFIGURACJA
# ==========================================

DATA_FOLDER = "csv_downloads"
PARTICIPANT_FILTER = ["P01", "P02"]
ENGINE_FILTER = ["XP"]  # FR / LU / XP

MAX_ROWS = 50_000  # ile ramek z każdego pliku
LAGS = [1, 2, 3]  # lag czasowy do rule mining

OUTPUT_DATASET = "emotion_dataset.csv"
OUTPUT_LAGGED = "emotion_dataset_lagged.csv"

# ==========================================
# PARSOWANIE NAZWY PLIKU
# ==========================================

def parse_filename(filename):
    name = Path(filename).stem
    match = re.match(r"(P\d+)(UL|UR|DL|DR)(FR|LU|XP)(.+)", name)
    if not match:
        return None
    participant, camera, engine, emotion = match.groups()
    return participant, camera, engine, emotion.lower()

# ==========================================
# WCZYTANIE CSV
# ==========================================

def load_all_csv(folder):
    features = []
    files = list(Path(folder).glob("*.csv"))
    print(f"Znaleziono {len(files)} plików")

    for i, file in enumerate(files, start=1):
        parsed = parse_filename(file.name)
        if not parsed:
            continue
        participant, camera, engine, emotion = parsed
        if PARTICIPANT_FILTER and participant not in PARTICIPANT_FILTER:
            continue
        if ENGINE_FILTER and engine not in ENGINE_FILTER:
            continue

        print(f"[{i}] 📖 {file.name}")
        df = pd.read_csv(file)
        df = df.rename(columns={"FrameMillis": "frameMillis", "Value": "value", "value": "value"})
        df = df[["frameMillis", "value"]]
        df["frameMillis"] = pd.to_numeric(df["frameMillis"], errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna()
        df["frameMillis"] = df["frameMillis"].astype("int32")
        df["value"] = df["value"].astype("float32")
        df = df.head(MAX_ROWS)
        feature_name = f"{camera}_{engine}_{emotion}"
        df = df.rename(columns={"value": feature_name})
        features.append(df)

    print(f"✅ Wczytano {len(features)} feature")
    return features


# ==========================================
# ŁĄCZENIE DATAFRAME
# ==========================================

def merge_features(feature_list):
    print("🔗 Łączenie danych")
    merged = feature_list[0]
    for df in feature_list[1:]:
        merged = pd.merge(merged, df, on="frameMillis", how="outer")
    merged = merged.sort_values("frameMillis")
    merged = merged.ffill()
    print("✅ Połączono dataset")
    return merged


# ==========================================
# DODANIE LAGÓW
# ==========================================

def add_lags(df):
    print("⏳ Dodawanie lagów")
    lagged = df.copy()
    for lag in LAGS:
        shifted = df.shift(-lag)
        shifted.columns = [f"{c}_t{lag}" if c != "frameMillis" else c for c in shifted.columns]
        lagged = pd.concat([lagged, shifted.drop(columns=["frameMillis"])], axis=1)
    lagged = lagged.dropna()
    print("Lags dodane")
    return lagged


# ==========================================
# BUDOWA DATASETU
# ==========================================

def build_dataset():
    features = load_all_csv(DATA_FOLDER)
    if len(features) == 0:
        print("❌ Brak danych")
        return None
    dataset = merge_features(features)
    dataset.to_csv(OUTPUT_DATASET, index=False)
    print(f"💾 Zapisano dataset: {OUTPUT_DATASET}")
    print("Shape:", dataset.shape)
    return dataset


# ==========================================
# BUDOWA DATASETU Z LAGAMI
# ==========================================

def build_lagged_dataset(dataset):
    dataset_lagged = add_lags(dataset)
    dataset_lagged.to_csv(OUTPUT_LAGGED, index=False)
    print(f"💾 Zapisano dataset z lagami: {OUTPUT_LAGGED}")
    print("Shape:", dataset_lagged.shape)
    return dataset_lagged


# ==========================================
# RULE MINING Z NiaARMTS
# ==========================================

def run_rule_mining(dataset):
    print("🧠 Uruchamianie rule mining")

    try:
        from niaarmts import NiaARMTS
    except ImportError:
        print("❌ Zainstaluj bibliotekę: pip install niaarmts")
        return

    # usuń kolumnę czasu
    data = dataset.drop(columns=["frameMillis"])
    transactions = data.values.astype(float)
    features = data.shape[1]

    dimension = features * 2
    lower = [0] * dimension
    upper = [1] * dimension
    interval = 10
    alpha = beta = gamma = delta = epsilon = 1

    # Tworzymy miner
    miner = NiaARMTS(
        dimension,
        lower,
        upper,
        features,
        transactions,
        interval,
        alpha, beta, gamma, delta, epsilon
    )

    print(dir(miner))

    # Wersja NiaARMTS w niektórych repo:
    # reguły są od razu w atrybutach: best_solution lub rules
    rules = getattr(miner, "best_solution", None)
    if rules is None:
        rules = getattr(miner, "rules", None)
    if rules is None:
        rules = "No found any rules by this NiaARMTS algorithm"

    print("\nOdkryte reguły:")
    print(rules)


# ==========================================
# MAIN
# ==========================================

def main():
    print("\n🚀 START PIPELINE\n")
    dataset = build_dataset()
    if dataset is None:
        return
    dataset_lagged = build_lagged_dataset(dataset)
    run_rule_mining(dataset_lagged)
    print("\n🎉 Pipeline zakończony")


if __name__ == "__main__":
    main()