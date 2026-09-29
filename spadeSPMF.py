import os
import subprocess
import time

SPMF_JAR = "spmf.jar"
INPUT_FILE = "spade_input_no_fear.txt"
OUTPUT_FILE = "spade_output.txt"

ALGORITHM = "SPADE"
MINSUP = "0.1"

# ==========================================
# STATYSTYKI INPUTU
# ==========================================

print("\n=== INPUT STATISTICS ===")

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(INPUT_FILE)

file_size_mb = os.path.getsize(INPUT_FILE) / (1024 * 1024)

sequence_lengths = []

with open(INPUT_FILE, "r") as f:
    for line in f:
        tokens = line.split()

        # liczba itemsetów
        length = tokens.count("-1")

        if length > 0:
            sequence_lengths.append(length)

n_sequences = len(sequence_lengths)

print(f"Input file      : {INPUT_FILE}")
print(f"File size       : {file_size_mb:.2f} MB")
print(f"Sequences       : {n_sequences}")

if n_sequences:
    print(f"Avg length      : {sum(sequence_lengths)/n_sequences:.1f}")
    print(f"Min length      : {min(sequence_lengths)}")
    print(f"Max length      : {max(sequence_lengths)}")
    print(f"Total itemsets  : {sum(sequence_lengths)}")

print()

# ==========================================
# KOMENDA SPMF
# ==========================================

cmd = [
    "java",
    "-jar",
    SPMF_JAR,
    "run",
    ALGORITHM,
    INPUT_FILE,
    OUTPUT_FILE,
    MINSUP
]

print("=== COMMAND ===")
print(" ".join(cmd))
print()

# ==========================================
# START
# ==========================================

start = time.perf_counter()

try:

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )

    elapsed = time.perf_counter() - start

    print("\n=== FINISHED ===")
    print(f"Elapsed time : {elapsed:.2f} s")
    print(f"Return code  : {result.returncode}")

    print("\n=== STDOUT ===")
    print(result.stdout if result.stdout else "(empty)")

    print("\n=== STDERR ===")
    print(result.stderr if result.stderr else "(empty)")

    if os.path.exists(OUTPUT_FILE):

        size_kb = os.path.getsize(OUTPUT_FILE) / 1024

        print("\n=== OUTPUT FILE ===")
        print(f"Created : {OUTPUT_FILE}")
        print(f"Size    : {size_kb:.2f} KB")

        with open(OUTPUT_FILE, "r") as f:
            lines = sum(1 for _ in f)

        print(f"Patterns: {lines}")

    else:

        print("\n=== OUTPUT FILE ===")
        print("Output file was NOT created!")

except Exception as e:

    elapsed = time.perf_counter() - start

    print("\n=== ERROR ===")
    print(type(e).__name__)
    print(str(e))
    print(f"Elapsed before error: {elapsed:.2f} s")