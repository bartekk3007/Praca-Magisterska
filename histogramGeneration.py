import os
import matplotlib.pyplot as plt

# Dane: wartość -> liczność
values = [0, 1, 2, 3, 4, 5, 6, 7]
counts = [1319, 1044, 828, 453, 213, 57, 7, 2]

# Katalog docelowy
output_dir = r"./wykresy"
os.makedirs(output_dir, exist_ok=True)

# Ścieżka pliku wynikowego
output_file = os.path.join(output_dir, "histogram.png")

# Tworzenie wykresu
plt.figure(figsize=(8, 5))
plt.bar(values, counts, width=0.8)
plt.title("Histogram")
plt.xlabel("W ilu nowych kombinacjach wykryto regułę")
plt.ylabel("Ilość wystąpień")
plt.xticks(values)
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Zapis do pliku
plt.tight_layout()
plt.savefig(output_file, dpi=300)
plt.close()

print(f"Histogram zapisano do: {output_file}")