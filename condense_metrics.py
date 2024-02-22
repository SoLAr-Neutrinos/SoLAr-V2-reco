import pickle
import sys
import os
import glob
from tqdm.auto import tqdm

condensed_metrics = {}

for file in tqdm(glob.glob("**/*.pkl")):
    folder = file.split("/")[0]
    tqdm.write(folder)
    with open(file, "rb") as f:
        condensed_metrics[folder] = pickle.load(f)

with open("condensed_metrics.pkl", "wb") as o:
    pickle.dump(condensed_metrics, o)

print("Done")