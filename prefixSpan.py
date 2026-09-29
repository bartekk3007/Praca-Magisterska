import os
import pandas as pd
from prefixspan import PrefixSpan

# CONFIG
DATA_DIR = "csv_downloads"
TARGET_PERSON = "P01"
TARGET_DEVICE = "LU"
THRESHOLD = 0.2

# PARSING FILENAME
def parse_filename(filename):
    name = filename.replace(".csv", "")
    person = name[:3]      # P01
    camera = name[3:5]     # DL
    device = name[5:7]     # LU
    emotion = name[7:]     # disgust / happiness etc.
    return person, camera, device, emotion

# LOAD SINGLE FILE
def load_sequence(file_path, emotion_name):
    df = pd.read_csv(file_path)
    sequence = []
    for _, row in df.iterrows():
        value = row["Value"]
        state = "HIGH" if value >= THRESHOLD else "LOW"
        sequence.append(f"{emotion_name}_{state}")

    cleaned = []
    for s in sequence:
        if not cleaned or cleaned[-1] != s:
            cleaned.append(s)
    return cleaned

# LOAD ONLY P01 + LU
def load_sequences_p01_lu(data_dir):
    sequences = []
    files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    print(f"Found {len(files)} files")
    print(f"Filtering: person={TARGET_PERSON}, device={TARGET_DEVICE}\n")
    used = 0
    skipped = 0
    for file in files:
        person, camera, device, emotion = parse_filename(file)
        if person != TARGET_PERSON or device != TARGET_DEVICE:
            skipped += 1
            continue
        path = os.path.join(data_dir, file)
        try:
            seq = load_sequence(path, emotion)
        except Exception as e:
            print(f"Error in {file}: {e}")
            continue
        sequences.append(seq)
        used += 1
        print(f"Loaded: {file} ({len(seq)} items)")

    print("\nSUMMARY")
    print(f"Used files: {used}")
    print(f"Skipped files: {skipped}")
    print(f"Sequences: {len(sequences)}")
    return sequences

# PREFIXSPAN
def run_prefixspan(sequences, minsup):
    ps = PrefixSpan(sequences)
    patterns = ps.frequent(minsup)
    patterns.sort(reverse=True, key=lambda x: x[0])
    return patterns

# MAIN
if __name__ == "__main__":
    print("Running PrefixSpan for P01 + LU only\n")
    sequences = load_sequences_p01_lu(DATA_DIR)
    print("\n========================")
    print("PREFIXSPAN ANALYSIS")
    print("========================\n")
    # dla jednej osoby: bardzo niskie minsup
    minsup = max(2, int(0.2 * len(sequences)))
    print(f"minsup = {minsup}\n")
    patterns = run_prefixspan(sequences, minsup)
    print("🔎 Top patterns:\n")
    for support, pattern in patterns[:30]:
        print(f"{support:4d}  {pattern}")