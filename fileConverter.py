import os
import numpy as np
import pandas as pd

input_dir = "csv_downloads"
output_dir = "binary_downloads"

os.makedirs(output_dir, exist_ok=True)


# --- 1. emotionRange (standardowe pliki) ---
def convert_emotion(path_in, path_out):
    df = pd.read_csv(path_in)

    emotion = df["emotionRange"].astype(str).str.strip()
    binary = (emotion != "<0.2").astype(np.uint8).to_numpy()

    np.save(path_out, binary)


# --- 2. arousal ---
# LVHA / LVLA → ostatnia litera A/H
def convert_arousal(path_in, path_out):
    df = pd.read_csv(path_in)

    quad = df["quadrant"].astype(str).str.strip()
    binary = np.array([1 if q[-1] == "H" else 0 for q in quad], dtype=np.uint8)

    np.save(path_out, binary)


# --- 3. valence ---
# LVLA / HVLA → pierwsza litera L/H
def convert_valence(path_in, path_out):
    df = pd.read_csv(path_in)

    quad = df["quadrant"].astype(str).str.strip()
    binary = np.array([1 if q[0] == "H" else 0 for q in quad], dtype=np.uint8)

    np.save(path_out, binary)


# --- main loop ---
for filename in os.listdir(input_dir):
    if not filename.endswith(".csv"):
        continue

    lower_name = filename.lower()
    input_path = os.path.join(input_dir, filename)

    base_name = os.path.splitext(filename)[0]
    output_path = os.path.join(output_dir, base_name + ".npy")

    # --- wybór typu pliku ---
    if "arousal" in lower_name:
        convert_arousal(input_path, output_path)
        print(f"[AROUSAL] zapisano: {output_path}")

    elif "valence" in lower_name:
        convert_valence(input_path, output_path)
        print(f"[VALENCE] zapisano: {output_path}")

    else:
        convert_emotion(input_path, output_path)
        print(f"[EMOTION] zapisano: {output_path}")