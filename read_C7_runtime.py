import glob
import numpy as np
import re
import sys

if (len(sys.argv) != 2):
    print("Usage: read_C7_runtime.py PATH")
    assert(False)

PATH = sys.argv[1]

times = []
for path in glob.glob(f"{PATH}/*.log"):
    with open(path, "r") as file:
        for line in file:
            if "TIME PER EVENT" in line:
                times.append(float(re.search(r'\d+(\.\d+)?', line).group()))
                break
        else:
            print(f"No time found for {path}")

print(f"Mean time for {PATH}: {np.mean(times)} sec")
