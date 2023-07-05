import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
from corsikaio import CorsikaParticleFile
from particle import Corsika7ID
from particle import Particle
import sys

from scripts import *

# PATH_TO_C7_FOLDER: Folder that conatins DATxxxx CORSIKA 7 files
# PATH_TO_C8_FOLDER: Folder structure 'shower_x/*/NAME/profile.parquet', with shower_x being different showers

if (len(sys.argv) != 4):
    print("Usage: create_lateral_plots.py PATH_TO_C7_FOLDER PATH_TO_C8_FOLDER OUTPUT_NAME")
    assert(False)

C7_PATHS = [sys.argv[1]]
C8_PATHS = [sys.argv[2], sys.argv[2]]
OUTPUT_NAME = sys.argv[3]

labels = ["CORSIKA 7", "CORSIKA 8"]
colors = ['tab:orange', 'tab:blue']

NAME_PARTICLE_FOLDER_C8 = "particles" # change name of folder where C8 profiles are stored

INJECTION_HEIGHT_C8 = 112.75e3 # in m, necessary for correction time calculation
INJECTION_HEIGHT_C7 = 112.8e3 # in m
OBSLEVEL = 6798 # in m. if unknown, set to 0.

XLOG_TIME = True

# onyl read in the first X showers. can be used to create plots quicker for debug purposes
# might also be used if there are unequal numbers of equal showers in the individual simulations
LIMIT_INPUT = np.inf

# Start...

if not os.path.exists(OUTPUT_NAME):
    os.makedirs(OUTPUT_NAME)

# CONSTANTS

c = 299_792_458  # speed of light in m/s
time_bias_C7 = (INJECTION_HEIGHT_C7 - OBSLEVEL)/c * 1e9
time_bias_C8 = (INJECTION_HEIGHT_C8 - OBSLEVEL)/c * 1e9 
# NOTE: this is just set to take into account the differences for now, because there are particles in C8 which are quicker than the speed of light, apparently...
#time_bias_C7 = (INJECTION_HEIGHT_C7 - INJECTION_HEIGHT_C8)/c * 1e9
#time_bias_C8 = 0

# define bins for lateral profiles

t_min = np.inf
t_max = -np.inf
r_min = np.inf
r_max = -np.inf
E_min = np.inf
E_max = -np.inf

# iterate over the first X showers of each data set.
# if X is too big, this can take a long time.
# if X is too small, statistics are not big enough.
MAX_ITERATIONS = 5

print("Start determining histogram limits")

i = 0
for PATH in C8_PATHS:
    for path in glob.glob(f"{PATH}/*/*/{NAME_PARTICLE_FOLDER_C8}/particles.parquet"):
        i = i+1
        if (i >= MAX_ITERATIONS):
            break
        df = pd.read_parquet(path, columns=["time", "x", "y", "kinetic_energy"])
        t_min = min(t_min, min(df['time'] * 1e9 - time_bias_C8))
        t_max = max(t_max, max(df['time'] * 1e9 - time_bias_C8))
        E_min = min(E_min, min(df['kinetic_energy']))
        E_max = max(E_max, max(df['kinetic_energy']))
        r_min = min(r_min, min(np.sqrt(df['x']**2 + df['y']**2)))
        r_max = max(r_max, max(np.sqrt(df['x']**2 + df['y']**2)))
i = 0
for PATH in C7_PATHS:
    for path in glob.glob(f"{PATH}/DAT*[!.long]"):
        with CorsikaParticleFile(path) as f:
            for e in f:
                i = i+1
                if (i >= MAX_ITERATIONS):
                    break
                t_min = min(t_min, min(e.particles['t']-time_bias_C7))
                t_max = max(t_max, max(e.particles['t']-time_bias_C7))
            else:
                continue
            break
NUM_HIST_BINS_T = 25
MIN_HIST_BINS_T = 1e-3 # TODO: Weird artifact for CORSIKA 7 at small times compared to CORSIKA 8
#MIN_HIST_BINS_T = t_min
MAX_HIST_BINS_T = t_max
if (XLOG_TIME):
    BINS_T = np.geomspace(MIN_HIST_BINS_T, MAX_HIST_BINS_T, NUM_HIST_BINS_T)
else:
    BINS_T = np.linspace(MIN_HIST_BINS_T, MAX_HIST_BINS_T, NUM_HIST_BINS_T)

NUM_HIST_BINS_R = 25
MIN_HIST_BINS_R = r_min
MAX_HIST_BINS_R = r_max
BINS_R = np.geomspace(MIN_HIST_BINS_R, MAX_HIST_BINS_R, NUM_HIST_BINS_R)

NUM_HIST_BINS_E = 25
MIN_HIST_BINS_E = E_min
MAX_HIST_BINS_E = E_max
BINS_E = np.geomspace(MIN_HIST_BINS_E, MAX_HIST_BINS_E, NUM_HIST_BINS_E)


# read C8 particle files

print("Reading in C8 particle files")

particles = [[11, -11], [22]]
particles_names = ['Charged', 'Photon']
assert(len(particles) == len(particles_names))

C8_r_hists = []

C8_E_hists = []

C8_t_hists = []

C8_2d_hists = []

NOTIFICATION_INTERVAL = 100 # get message for every Xth shower

for PATH in C8_PATHS:
    i = 0
    print("Read C8 showers from path", PATH)
    r_hists = [ [] for _ in range(len(particles)) ] # create empty list for every particle type
    E_hists = [ [] for _ in range(len(particles)) ] # create empty list for every particle type
    t_hists = [ [] for _ in range(len(particles)) ] # create empty list for every particle type
    hists_2d = [ [] for _ in range(len(particles)) ] # create empty list for every particle type

    for path in glob.glob(f"{PATH}/*/*/{NAME_PARTICLE_FOLDER_C8}/particles.parquet"):
        if (i > LIMIT_INPUT):
            print(f"Reached maximum input of {LIMIT_INPUT} showers, contiunue...")
            break
        i = i+1
        if(i%NOTIFICATION_INTERVAL==0):
            print("reading shower number", i)
        df = pd.read_parquet(path, columns=["pdg", "kinetic_energy", "x", "y", "time"])

        for count, particle in enumerate(particles):
            r_list = []
            E_list = []
            t_list = []
            for p in particle:
                # loop over all particles in particle group
                r_list.extend(np.sqrt(df.query(f'pdg == {p}')['x']**2 + df.query(f'pdg == {p}')['y']**2))
                E_list.extend(df.query(f'pdg == {p}')['kinetic_energy'])
                t_list.extend(df.query(f'pdg == {p}')['time'] * 1e9 - time_bias_C8)

            r_hist, _ = np.histogram(r_list, BINS_R)
            r_hists[count].append(r_hist)

            E_hist, _ = np.histogram(E_list, BINS_E)
            E_hists[count].append(E_hist)
    
            t_hist, _ = np.histogram(t_list, BINS_T)
            t_hists[count].append(t_hist)

            hist_2d, _, _ = np.histogram2d(r_list, E_list, bins=[BINS_R, BINS_E]) 
            hists_2d[count].append(hist_2d)

    C8_r_hists.append(r_hists)
    C8_E_hists.append(E_hists)
    C8_t_hists.append(t_hists)
    C8_2d_hists.append(hists_2d)

# read C7 particle files

print("Reading in C7 particle files")

C7_r_hists = []

C7_E_hists = []

C7_t_hists = []

C7_2d_hists = []

for PATH in C7_PATHS:
    i = 0
    print("Read C7 showers from path", PATH)
    r_hists = [ [] for _ in range(len(particles)) ] # create empty list for every particle type
    E_hists = [ [] for _ in range(len(particles)) ] # create empty list for every particle type
    t_hists = [ [] for _ in range(len(particles)) ] # create empty list for every particle type
    hists_2d = [ [] for _ in range(len(particles)) ] # create empty list for every particle type

    for path in glob.glob(f"{PATH}/DAT*[!.long]"):
        with CorsikaParticleFile(path) as f:
            for e in f:
                if (i > LIMIT_INPUT):
                    print(f"Reached maximum input of {i} showers, continue...")
                    break
                i = i+1
                for count, particle in enumerate(particles):
                    r_list = []
                    E_list = []
                    t_list = []
                    for p in particle:
                        # loop over all particles in particle group
                        # within C7, the particle ID is stored as CORSIKA_ID*1000, so we need to account for that
                        particle_filter = (np.int32(e.particles['particle_description']/1000) == Corsika7ID.from_pdgid(p))
                        mass = Particle.from_pdgid(p).mass / 1000 # mass from Particle framework is in MeV, convert to GeV

                        r_list.extend(np.sqrt(e.particles['x'][particle_filter]**2 + e.particles['y'][particle_filter]**2) / 100) # cm to m in C7!!!
                        E_list.extend(np.sqrt(e.particles['px'][particle_filter]**2 + e.particles['py'][particle_filter]**2 + e.particles['pz'][particle_filter]**2 + mass**2) - mass)
                        t_list.extend(e.particles['t'][particle_filter] - time_bias_C7)

                    r_hist, _ = np.histogram(r_list, BINS_R)
                    r_hists[count].append(r_hist)

                    E_hist, _ = np.histogram(E_list, BINS_E)
                    E_hists[count].append(E_hist)
            
                    t_hist, _ = np.histogram(t_list, BINS_T)
                    t_hists[count].append(t_hist)

                    hist_2d, _, _ = np.histogram2d(r_list, E_list, bins=[BINS_R, BINS_E]) 
                    hists_2d[count].append(hist_2d)

    C7_r_hists.append(r_hists)
    C7_E_hists.append(E_hists)
    C7_t_hists.append(t_hists)
    C7_2d_hists.append(hists_2d)

print("Plot lateral profiles")

# r distribution

for count, p_name in enumerate(particles_names):
    plot_lateral_hist_ratio(BINS_R, [sublist[count] for sublist in C7_r_hists+C8_r_hists], labels, colors, f'{p_name}', r"distance to shower axis / m", ratio_lim=(-0.25, 0.25), add_watermark=True)
    plt.savefig(f"{OUTPUT_NAME}/lateral_{p_name}_r.pdf", dpi=300)

# E distribution

for count, p_name in enumerate(particles_names):
    plot_lateral_hist_ratio(BINS_E, [sublist[count] for sublist in C7_E_hists+C8_E_hists], labels, colors, f'{p_name}', r"energy on observation plane / GeV", ratio_lim=(-0.25, 0.25), add_watermark=True)
    plt.savefig(f"{OUTPUT_NAME}/lateral_{p_name}_E.pdf", dpi=300)

# t distribution

for count, p_name in enumerate(particles_names):
    plot_lateral_hist_ratio(BINS_T, [sublist[count] for sublist in C7_t_hists+C8_t_hists], labels, colors, f'{p_name}', r"arrival time delay / ns", xlog=XLOG_TIME, ratio_lim=(-0.25, 0,25), add_watermark=True)
    plt.savefig(f"{OUTPUT_NAME}/lateral_{p_name}_t.pdf", dpi=300)

# 2d distribution in r-E

for count, p_name in enumerate(particles_names):
    plot_lateral_2d(BINS_R, BINS_E, [sublist[count] for sublist in C7_2d_hists+C8_2d_hists], labels, 'r / m', 'E / GeV', add_watermark=True)
    plt.savefig(f"{OUTPUT_NAME}/lateral_{p_name}_2d_r_E.pdf", dpi=300)

# 2d distribution in r-E

for count, p_name in enumerate(particles_names):
    plot_lateral_2d_ratio(BINS_R, BINS_E, [sublist[count] for sublist in C7_2d_hists+C8_2d_hists], labels, 'r / m', 'E / GeV', add_watermark=True)
    plt.savefig(f"{OUTPUT_NAME}/lateral_{p_name}_2d_r_E_ratios.pdf", dpi=300)