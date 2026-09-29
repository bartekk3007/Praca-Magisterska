import os
import re
import numpy as np
from collections import defaultdict

# =========================
# PARAMETRY
# =========================

DATA_DIR = "binary_downloads"

WINDOW_SIZE = 2500

WINDOW_THRESHOLD = 0.20      # próg aktywacji w jednym źródle
AGREEMENT_THRESHOLD = 0.50   # próg zgodności między źródłami

EMOTIONS = [
    "anger",
    "disgust",
    "happiness",
    "sadness",
    "surprise"
]

pattern = re.compile(
    r"(P\d{2})(DL|DR|UL|UR)(FR|LU|XP)(.+)\.npy",
    re.IGNORECASE
)

# =========================
# PARSER
# =========================

def parse_filename(fname):
    m = pattern.match(fname)
    if not m:
        return None

    return {
        "participant": m.group(1),
        "camera": m.group(2),
        "system": m.group(3),
        "emotion": m.group(4).lower()
    }

# =========================
# GRUPOWANIE
# =========================

participants = defaultdict(lambda: defaultdict(list))

for fname in os.listdir(DATA_DIR):
    if not fname.endswith(".npy"):
        continue

    info = parse_filename(fname)
    if info is None:
        continue

    if info["emotion"] not in EMOTIONS:
        continue

    participants[info["participant"]][info["emotion"]].append(
        os.path.join(DATA_DIR, fname)
    )

print("Participants:", len(participants))

# =========================
# MAPOWANIE EMOCJI
# =========================

emotion_to_id = {e: i + 1 for i, e in enumerate(EMOTIONS)}

print("\nEmotion mapping:")
for k, v in emotion_to_id.items():
    print(v, "=", k)

# =========================
# BUDOWA SEKWENCJI
# =========================

all_sequences = []

for participant, emotion_files in participants.items():

    print(f"\nProcessing {participant}")

    # znajdź dowolny plik do określenia długości
    sample_file = None
    for emo in EMOTIONS:
        if len(emotion_files[emo]) > 0:
            sample_file = emotion_files[emo][0]
            break

    if sample_file is None:
        continue

    length = len(np.load(sample_file, mmap_mode="r"))
    n_windows = length // WINDOW_SIZE

    sequence = []

    for w in range(n_windows):

        start = w * WINDOW_SIZE
        end = start + WINDOW_SIZE

        active_emotions = []

        for emotion in EMOTIONS:

            files = emotion_files[emotion]
            if len(files) == 0:
                continue

            votes = []

            for path in files:

                arr = np.load(path, mmap_mode="r")

                ratio = arr[start:end].mean()

                # 1. próg w pojedynczym źródle
                vote = 1 if ratio >= WINDOW_THRESHOLD else 0

                votes.append(vote)

            # 2. próg zgodności między źródłami
            agreement = np.mean(votes)

            if agreement >= AGREEMENT_THRESHOLD:
                active_emotions.append(emotion)

        sequence.append(tuple(sorted(active_emotions)))

    print("Sequence length:", len(sequence))

    if len(sequence) > 0:
        all_sequences.append(sequence)

print("\nTotal sequences:", len(all_sequences))

# =========================
# ZAPIS DO SPMF (SPADE)
# =========================

OUT_FILE = "spade_channel_no_neutral_agnostic2500.txt"

with open(OUT_FILE, "w") as f:

    for seq in all_sequences:

        line = []

        for itemset in seq:

            for emo in itemset:
                line.append(str(emotion_to_id[emo]))

            line.append("-1")

        line.append("-2")

        f.write(" ".join(line) + "\n")

print("\nSaved:", OUT_FILE)