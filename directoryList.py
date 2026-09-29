import os

folder = "binary_downloads"

files = os.listdir(folder)

for f in files:
    print(f)