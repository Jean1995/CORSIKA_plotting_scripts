import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
from corsikaio import CorsikaFile
import sys

from yaml import safe_load
import datetime
import time
from itertools import chain

from scripts import *

# PATH_TO_C7_FOLDER: Folder that conatins DATxxxx CORSIKA 7 files
# PATH_TO_C8_FOLDER: Folder structure 'shower_x/*/profile/profile.parquet', with shower_x being different showers

if (len(sys.argv) != 4):
    print("Usage: create_plots.py PATH_TO_C7_FOLDER PATH_TO_C8_FOLDER OUTPUT_NAME")
    assert(False)

C7_PATHS = [sys.argv[1]]
C8_PATHS = [sys.argv[2]]
OUTPUT_NAME = sys.argv[3]

labels = ["CORSIKA 7", "CORSIKA 8"]
colors = ['tab:orange', 'tab:blue']

NAME_PROFILE_FOLDER_C8 = "profile" # change name of folder where C8 profiles are stored
xmax_C8 = 1040 # upper limit of grammage for C8

assert(len(labels) == len(colors))

CREATE_CHARGE_EXCESS_PLOTS = True
CREATE_XMAX_PLOTS = True
CREATE_LONGITUDINAL_PLOTS = True

if not os.path.exists(OUTPUT_NAME):
    os.makedirs(OUTPUT_NAME)

### Read in C8

def X_C8(profiles, NUM_SHOWERS):
    'returns array of length n describing the X bins'
    return profiles['X'][:int(len(profiles['X']) / NUM_SHOWERS)]

def profile_C8(profiles, label, NUM_SHOWERS):
    'returns array of shape m * n, where m is the number of profiles. every row has n entries which describe the longitudinal profile '
    return profiles[label].to_numpy().reshape(NUM_SHOWERS, -1)


C8_DATA = []
C8_runtimes = []
C8_SHOWER_NUMBERS = []

for PATH in C8_PATHS:
    print(f"Reading in C8 profiles from folder {PATH}")
    profiles= []
    for path in glob.glob(f"{PATH}/*/*/{NAME_PROFILE_FOLDER_C8}/profile.parquet"):
        data_raw = pd.read_parquet(path)
        profiles.append(data_raw[data_raw['X'] <= xmax_C8]) # cut grammage above xmax_X8
    C8_DATA.append(pd.concat(profiles))
    num_showers = len(profiles)
    C8_SHOWER_NUMBERS.append(num_showers)
    runtimes = []
    for path in glob.glob(f"{PATH}/*/*/summary.yaml"):
        with open(path, 'r') as f:
            time_object = safe_load(f)['runtime']
            if (type(time_object) is float):
                runtimes.append(time_object)
            else:
                str_obj = time.strptime(time_object,'%H:%M:%S')
                str_obj_in_sec = datetime.timedelta(hours=str_obj.tm_hour,minutes=str_obj.tm_min,seconds=str_obj.tm_sec).total_seconds()
                runtimes.append(str_obj_in_sec)
    sum_runtimes = sum(runtimes)
    C8_runtimes.append(sum_runtimes)
    print(f"Sum of all runtimes for C8 simulation: {sum_runtimes}, which corresponds to a time per shower of {sum_runtimes/num_showers}")


### Read in C7

def X_C7(df, NUM_SHOWERS):
    'returns array of length n describing the X bins'
    return df['X'][:int(len(df['X']) / NUM_SHOWERS)]

def profile_C7(df, label, NUM_SHOWERS):
    'returns array of shape m * n, where m is the number of profiles. every row has n entries which describe the longitudinal profile '
    return df[label].to_numpy().reshape(NUM_SHOWERS, -1)

C7_DATA = []
C7_SHOWER_NUMBERS = []

for PATH in C7_PATHS:
    print(f"Reading in C7 profiles from file {PATH}")
    C7_df = pd.DataFrame()
    NUM_C7_SHOWERS = 0
    X = []
    electron = []
    positron = []
    photon = []
    charged = []
    hadron = []
    muplus = []
    muminus = []
    for path in glob.glob(f"{PATH}/DAT*"):
        print(f"read {path}")
        with CorsikaFile(path) as f:
            NUM_C7_SHOWERS+=int(f.run_header["n_showers"])
            for e in f:
                X.append(e.longitudinal['vertical_depth'])
                electron.append(e.longitudinal['n_e_minus'])
                positron.append(e.longitudinal['n_e_plus'])
                photon.append(e.longitudinal['n_photons'])
                charged.append(e.longitudinal['n_charged'])
                hadron.append(e.longitudinal['n_hadrons'])
                muplus.append(e.longitudinal['n_mu_plus'])
                muminus.append(e.longitudinal['n_mu_minus'])
    # list(chain.from_iterable(X)) flattens the 2d array 
    C7_df = pd.DataFrame({"X":list(chain.from_iterable(X)),
                     "electron":list(chain.from_iterable(electron)),
                     "positron":list(chain.from_iterable(positron)),
                     "photon":list(chain.from_iterable(photon)),
                     "charged":list(chain.from_iterable(charged)),
                     "hadron":list(chain.from_iterable(hadron)),
                     "muplus":list(chain.from_iterable(muplus)),
                     "muminus":list(chain.from_iterable(muminus))})
    C7_SHOWER_NUMBERS.append(NUM_C7_SHOWERS)
    C7_DATA.append(C7_df)

print("Finished reading C7")

# generate grammage

grammage = []
for df, NUM in zip(C7_DATA, C7_SHOWER_NUMBERS):
    grammage.append(X_C7(df, NUM))

for df, NUM in zip(C8_DATA, C8_SHOWER_NUMBERS):
    grammage.append(X_C8(df, NUM))

### plot charge excess

if (CREATE_CHARGE_EXCESS_PLOTS):
    print("Create charge excess plots")
    charge_excess = []

    for df, NUM in zip(C7_DATA, C7_SHOWER_NUMBERS):
        profile_electron = profile_C7(df, 'electron', NUM)
        profile_positron = profile_C7(df, 'positron', NUM)
        profile_charged = profile_electron + profile_positron
        # if there are no entries, replace by 1 to avoid division by zero. 
        # for the charge excesss, we will end up in the end with zero anyway, we just do 0/1 instead of 0/0
        profile_charged[profile_charged == 0] = 1 
        charge_excess.append((profile_electron - profile_positron) / (profile_charged))

    for df, NUM in zip(C8_DATA, C8_SHOWER_NUMBERS):
        profile_electron = profile_C8(df, 'electron', NUM)
        profile_positron = profile_C8(df, 'positron', NUM)
        profile_charged = profile_electron + profile_positron
        profile_charged[profile_charged == 0] = 1 
        charge_excess.append((profile_electron - profile_positron) / (profile_charged))

    plot_long_hist_ratio(grammage, charge_excess, labels, colors, f"Charge excess")
    plt.savefig(f"{OUTPUT_NAME}/charge_excess.png", dpi=300)

### create X_max profiles

if (CREATE_XMAX_PLOTS):
    print("Create Xmax plots")

    for p_type in ['electron', 'positron', 'photon', 'charged', 'muminus', 'muplus', 'muons', 'hadron']:
        profiles = []
        for df, NUM in zip(C7_DATA, C7_SHOWER_NUMBERS):
            if (p_type=='muons'):
                profiles.append(profile_C7(df, 'muminus', NUM) + profile_C7(df, 'muplus', NUM))
            else:
                profiles.append(profile_C7(df, p_type, NUM))
        for df, NUM in zip(C8_DATA, C8_SHOWER_NUMBERS):
            if (p_type=='muons'):
                profiles.append(profile_C8(df, 'muminus', NUM) + profile_C8(df, 'muplus', NUM))
            else:
                profiles.append(profile_C8(df, p_type, NUM))
        plot_Xmax_hist_ratio(grammage, profiles, labels, colors, f"Xmax distribution for {p_type}", NUM_BINS=20)
        plt.savefig(f"{OUTPUT_NAME}/Xmax_{p_type}.png", dpi=300)

### plot longitudinal profiles

if (CREATE_LONGITUDINAL_PLOTS):
    print("Create longitudinal plots")
    for p_type in ['electron', 'positron', 'photon', 'charged', 'muminus', 'muplus', 'muons', 'hadron']:
        profiles = []
        for df, NUM in zip(C7_DATA, C7_SHOWER_NUMBERS):
            if (p_type=='muons'):
                profiles.append(profile_C7(df, 'muminus', NUM) + profile_C7(df, 'muplus', NUM))
            else:
                profiles.append(profile_C7(df, p_type, NUM))
        for df, NUM in zip(C8_DATA, C8_SHOWER_NUMBERS):
            if (p_type=='muons'):
                profiles.append(profile_C8(df, 'muminus', NUM) + profile_C8(df, 'muplus', NUM))
            else:
                profiles.append(profile_C8(df, p_type, NUM))
        plot_long_hist_ratio(grammage, profiles, labels, colors, f"Longitudinal profile for {p_type}")
        plt.savefig(f"{OUTPUT_NAME}/long_{p_type}.png", dpi=300)

