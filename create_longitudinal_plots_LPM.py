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

if (len(sys.argv) != 6):
    print("Usage: create_plots.py PATH_TO_C7_FOLDER PATH_TO_C8_FOLDER PATH_TO_C7_FOLDER_LPM PATH_TO_C8_FOLDER_LPM OUTPUT_NAME")
    assert(False)

C7_PATHS = [sys.argv[1]]
C8_PATHS = [sys.argv[2]]

C7_PATHS_LPM = [sys.argv[3]]
C8_PATHS_LPM = [sys.argv[4]]

OUTPUT_NAME = sys.argv[5]

labels = ["CORSIKA 7", "CORSIKA 8"]
colors = ['tab:orange', 'tab:blue']

NAME_PROFILE_FOLDER_C8 = "profile" # change name of folder where C8 profiles are stored
xmax_C8 = 1040 # upper limit of grammage for C8

assert(len(labels) == len(colors))
assert(len(C8_PATHS) == len(C8_PATHS_LPM))
assert(len(C7_PATHS) == len(C7_PATHS_LPM))

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

C8_DATA_LPM = []
C8_SHOWER_NUMBERS_LPM = []

for PATH in C8_PATHS_LPM:
    print(f"Reading in C8 profiles from folder {PATH}")
    profiles= []
    for path in glob.glob(f"{PATH}/*/*/{NAME_PROFILE_FOLDER_C8}/profile.parquet"):
        data_raw = pd.read_parquet(path)
        profiles.append(data_raw[data_raw['X'] <= xmax_C8]) # cut grammage above xmax_X8
    C8_DATA_LPM.append(pd.concat(profiles))
    num_showers = len(profiles)
    C8_SHOWER_NUMBERS_LPM.append(num_showers)

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


C7_DATA_LPM = []
C7_SHOWER_NUMBERS_LPM = []

for PATH in C7_PATHS_LPM:
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
    C7_SHOWER_NUMBERS_LPM.append(NUM_C7_SHOWERS)
    C7_DATA_LPM.append(C7_df)

print("Finished reading C7")

# generate grammage

grammage = []
for df, NUM in zip(C7_DATA, C7_SHOWER_NUMBERS):
    grammage.append(X_C7(df, NUM))

for df, NUM in zip(C8_DATA, C8_SHOWER_NUMBERS):
    grammage.append(X_C8(df, NUM))

grammage_LPM = []
for df, NUM in zip(C7_DATA_LPM, C7_SHOWER_NUMBERS_LPM):
    grammage_LPM.append(X_C7(df, NUM))

for df, NUM in zip(C8_DATA_LPM, C8_SHOWER_NUMBERS_LPM):
    grammage_LPM.append(X_C8(df, NUM))

### plot longitudinal profiles

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

    profiles_LPM = []
    for df, NUM in zip(C7_DATA_LPM, C7_SHOWER_NUMBERS_LPM):
        if (p_type=='muons'):
            profiles_LPM.append(profile_C7(df, 'muminus', NUM) + profile_C7(df, 'muplus', NUM))
        else:
            profiles_LPM.append(profile_C7(df, p_type, NUM))
    for df, NUM in zip(C8_DATA_LPM, C8_SHOWER_NUMBERS_LPM):
        if (p_type=='muons'):
            profiles_LPM.append(profile_C8(df, 'muminus', NUM) + profile_C8(df, 'muplus', NUM))
        else:
            profiles_LPM.append(profile_C8(df, p_type, NUM))

    plot_long_hist_ratio_lpm(grammage, profiles, grammage_LPM, profiles_LPM, labels, colors, f"Longitudinal profile for {p_type}")
    plt.savefig(f"{OUTPUT_NAME}/long_{p_type}.png", dpi=300)

