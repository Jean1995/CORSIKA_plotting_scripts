import numpy as np
import glob
import sys

from yaml import safe_load
import datetime
import time

if (len(sys.argv) != 2):
    print("Usage: read_C8_runtimes.py PATH")
    assert(False)

PATH = sys.argv[1]


print(f"Reading in C8 runtimes from folder {PATH}")
runtimes = []
for path in glob.glob(f"{PATH}/*/*/summary.yaml"):
    with open(path, 'r') as f:
        data = safe_load(f)
        time_object = data.get('runtime', None)
        num_showers = data.get('showers', None)
        if (type(time_object) is float):
            runtimes.append(time_object/num_showers)
        else:
            str_obj = time.strptime(time_object,'%H:%M:%S')
            str_obj_in_sec = datetime.timedelta(hours=str_obj.tm_hour,minutes=str_obj.tm_min,seconds=str_obj.tm_sec).total_seconds()
            runtimes.append(str_obj_in_sec/num_showers)

print(f"Average runtime of C8 in {PATH} of {np.mean(runtimes)} sec")
