import numpy as np
from pathlib import Path
from prefixspan import PrefixSpan

# =========================
# PARAMETRY
# =========================
DATA_DIR = Path("binary_downloads")

PARTICIPANT = "P01"
DEVICE = "LU"

WINDOW_SIZE = 50
MIN_SUPPORT = 5


# =========================
# 1. KONWERSJA SYGNAŁU → SEKWENCJA
# =========================
def signal_to_sequence(signal, window_size=50):
    """
    Zamienia sygnał na sekwencję zmian stanów.
    """

    compressed = []

    prev = None

    for i in range(0, len(signal), window_size):
        chunk = signal[i:i + window_size]

        if len(chunk) == 0:
            continue

        avg = np.mean(chunk)

        current = 1 if avg >= 0.5 else 0

        # zapisujemy tylko zmianę
        if current != prev:
            compressed.append(current)
            prev = current

    return compressed


# =========================
# 2. WCZYTANIE TYLKO P01 + LU
# =========================
def load_filtered_sequences():
    all_sequences = []

    # np. P01DLLUanger.npy
    pattern = f"{PARTICIPANT}*{DEVICE}*.npy"

    files = list(DATA_DIR.glob(pattern))

    print(f"Znaleziono plików: {len(files)}")

    for idx, file in enumerate(files, 1):
        signal = np.load(file)

        seq = signal_to_sequence(signal, WINDOW_SIZE)

        all_sequences.append(seq)

        print(f"\rZaładowano: {idx}/{len(files)} plików", end="")

    print("\nGotowe ładowanie.")

    return all_sequences


# =========================
# 3. PREFIXSPAN
# =========================
def run_prefixspan(sequences):
    ps = PrefixSpan(sequences)

    patterns = ps.frequent(MIN_SUPPORT)

    patterns = sorted(patterns, key=lambda x: -x[0])

    return patterns


# =========================
# 4. MAIN
# =========================
if __name__ == "__main__":
    print(f"Participant: {PARTICIPANT}")
    print(f"Device: {DEVICE}")

    print("\nŁadowanie danych...")
    sequences = load_filtered_sequences()

    print(f"\nLiczba sekwencji: {len(sequences)}")

    print("\nUruchamianie PrefixSpan...")
    patterns = run_prefixspan(sequences)

    print("\nTOP PATTERNS:\n")

    for support, pattern in patterns[:20]:
        print(f"Support={support} | Pattern={pattern}")