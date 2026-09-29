import os
import numpy as np
import pandas as pd

INPUT_DIR = "csv_downloads"
OUTPUT_DIR = "float_downloads"

os.makedirs(OUTPUT_DIR, exist_ok=True)

for fname in os.listdir(INPUT_DIR):

    if not fname.endswith(".csv"):
        continue

    csv_path = os.path.join(INPUT_DIR, fname)

    try:
        df = pd.read_csv(csv_path)

        if "Value" not in df.columns:
            print("Brak kolumny Value:", fname)
            continue

        values = df["Value"].to_numpy(dtype=np.float32)

        out_name = os.path.splitext(fname)[0] + ".npy"
        out_path = os.path.join(OUTPUT_DIR, out_name)

        np.save(out_path, values)

        print("OK:", out_name, len(values))

    except Exception as e:
        print("BŁĄD:", fname, e)

print("KONIEC")