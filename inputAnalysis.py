INPUT_FILE = "spade_input_neutral.txt"

with open(INPUT_FILE, "r") as f:
    lines = f.readlines()

total_windows = 0
total_multi = 0

max_percent = -1
max_seq = None

min_percent = 101
min_seq = None

print(
    "seq_id,all_windows,multi_windows,single_windows,multi_percent"
)

for seq_id, line in enumerate(lines):

    tokens = line.strip().split()

    itemset = []
    all_windows = 0
    multi_windows = 0

    for token in tokens:

        if token == "-1":

            if len(itemset) > 0:

                all_windows += 1

                if len(itemset) > 1:
                    multi_windows += 1

            itemset = []

        elif token == "-2":
            break

        else:
            itemset.append(int(token))

    single_windows = all_windows - multi_windows

    percent = (
        100 * multi_windows / all_windows
        if all_windows > 0
        else 0
    )

    total_windows += all_windows
    total_multi += multi_windows

    if percent > max_percent:
        max_percent = percent
        max_seq = seq_id

    if percent < min_percent:
        min_percent = percent
        min_seq = seq_id

    print(
        f"{seq_id},"
        f"{multi_windows},"
        f"{all_windows},"
        f"{percent:.2f}"
    )

print("\n===== GLOBAL =====")

global_percent = (
    100 * total_multi / total_windows
    if total_windows > 0
    else 0
)

print("multi_windows:", total_multi)
print("all_windows:", total_windows)
print("multi_percent:", round(global_percent, 2))

print("\n===== EXTREMES =====")

print(
    f"MAX multi-percent: {max_percent:.2f}% "
    f"(sequence {max_seq})"
)

print(
    f"MIN multi-percent: {min_percent:.2f}% "
    f"(sequence {min_seq})"
)