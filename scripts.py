import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from corsikaio import CorsikaFile

def plot_long_hist(X, profile, title=''):    
    '''
    X: len n, which describes the X bins
    profile: array of shape m * n, where m is the number of profiles. every row has n entries which describe the longitudinal profile
    '''
    plt.fill_between(X, np.percentile(profile.T, q=25, axis=1), np.percentile(profile.T, q=75, axis=1), step='pre', alpha=0.5, color='blue')
    plt.step(X, np.percentile(profile.T, q=50, axis=1), color='blue', label='CORSIKA 8 (07ec65d5)')
    
    plt.grid()
    plt.legend(fontsize=12)
    plt.xlabel('grammage', fontsize=12)
    plt.ylabel('number particles', fontsize=12)
    plt.xlim(0, 1000)
    plt.title(title, fontsize=14)
    plt.tight_layout()

def plot_long_hist_multiple(X_list, profile_list, label_list, color_list, title=''):    
    '''
    X: len n, which describes the X bins
    profile: array of shape m * n, where m is the number of profiles. every row has n entries which describe the longitudinal profile
    '''
    for X, profile, label, color in zip(X_list, profile_list, label_list, color_list):
        plt.fill_between(X, np.percentile(profile.T, q=25, axis=1), np.percentile(profile.T, q=75, axis=1), step='pre', alpha=0.5, color=color)
        plt.step(X, np.percentile(profile.T, q=50, axis=1), label=label, color=color)
    
    plt.grid()
    plt.legend(fontsize=12)
    plt.xlabel('grammage', fontsize=12)
    plt.ylabel('number particles', fontsize=12)
    plt.xlim(0, 1000)
    plt.title(title, fontsize=14)    
    plt.tight_layout()

def plot_long_hist_ratio(X_list, profile_list, label_list, color_list, title='', add_watermark=False):    
    '''
    X: len n, which describes the X bins
    profile: array of shape m * n, where m is the number of profiles. every row has n entries which describe the longitudinal profile
    '''
    assert(len(X_list) > 1)

    fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]}, figsize=(15,10),sharex=True)
    
    ax[0].tick_params('x', labelbottom=False) # only for last plot
    for X, profile, label, color in zip(X_list, profile_list, label_list, color_list):
        ax[0].fill_between(X, np.percentile(profile.T, q=25, axis=1), np.percentile(profile.T, q=75, axis=1), step='pre', alpha=0.5, color=color)
        ax[0].step(X, np.percentile(profile.T, q=50, axis=1), label=label, color=color)
    ax[0].grid(which='both')
    ax[0].set_ylabel('# particles', fontsize=14)
    
    ax[0].legend(bbox_to_anchor=(0, 1, 1, 0), loc="lower left", ncol=2, fontsize=14)

    if (add_watermark):
        ax[0].text(0.85, 0.85, 'C8 - ICRC2023', horizontalalignment='center', verticalalignment='center', transform = ax[0].transAxes, fontsize=18, alpha=0.5, color='gray')


    for X, profile, label, color in zip(X_list[1:], profile_list[1:], label_list[1:], color_list[1:]):
        # find common X values for base and compare
        X_common, indices_base, indices_compare = np.intersect1d(X_list[0], X, return_indices=True)
        vals_base = np.percentile(profile_list[0].T[indices_base], q=50, axis=1)
        vals_compare = np.percentile(profile.T[indices_compare], q=50, axis=1)
        ax[1].step(X_common, (vals_compare - vals_base)/vals_base, label=label, color=color)   
    ax[1].grid(which='both')
    fig.subplots_adjust(hspace=0.05)
    ax[1].set_xlabel(r"grammage / g/cm²", fontsize=14)
    ax[1].set_ylabel(f"ratio to {label_list[0]}", fontsize=14)
    ax[1].set_ylim(-0.2, 0.2)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
        

def plot_Xmax_hist(X_list, profile_list, label_list, color_list, title='', NUM_BINS=15):
    for X, profile, label, color in zip(X_list, profile_list, label_list, color_list):
        max_list = []
        for prof in profile:
            max_list.append(X[np.argmax(prof)])
        plt.hist(max_list, bins=NUM_BINS, histtype='step', label=label, color=color)
    plt.legend(fontsize=12)
    plt.xlabel('X_max / grammage', fontsize=12)
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

def plot_Xmax_hist_ratio(X_list, profile_list, label_list, color_list, title='', NUM_BINS=15, add_watermark=False):
    assert(len(X_list) > 1)
    fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]}, figsize=(15,10),sharex=True)
    
    ax[0].tick_params('x', labelbottom=False) # only for last plot
    
    X_max_lists = []
   
    X_max_mean_list = []

    min_bin = +np.inf
    max_bin = -np.inf
   

    for X, profile in zip(X_list, profile_list):
        X_max = []
        for prof in profile:
            X_max.append(X[np.argmax(prof)])
        min_bin = min(min_bin, min(X))
        max_bin = max(max_bin, max(X))
        X_max_lists.append(X_max)
        X_max_mean_list.append(np.mean(X_max))
        
    bins = np.linspace(min_bin, max_bin, NUM_BINS)
    
    n_list = []
    
    for X_max_list, label, color, X_max_mean in zip(X_max_lists, label_list, color_list, X_max_mean_list):
        n, _, _ = ax[0].hist(X_max_list, bins=bins, histtype='step', color=color, label=f"{label}, mean(X_max) = {X_max_mean:.2f} g/cm²")
        n_list.append(n)
        
    ax[0].grid(which='both')
    ax[0].set_ylabel('# showers', fontsize=14)
    ax[0].legend(bbox_to_anchor=(0, 1, 1, 0), loc="lower left", ncol=2, fontsize=14)
    
    if (add_watermark):
        ax[0].text(0.85, 0.85, 'C8 - ICRC2023', horizontalalignment='center', verticalalignment='center', transform = ax[0].transAxes, fontsize=18, alpha=0.5, color='gray')

    for n, label, color in zip(n_list[1:], label_list[1:], color_list[1:]):
        ax[1].step(bins[1:], (n - n_list[0]) / n_list[0], color=color)
    ax[1].grid(which='both')
    fig.subplots_adjust(hspace=0.05)
    ax[1].set_xlabel(r"grammage / g/cm²", fontsize=14)
    ax[1].set_ylabel(f"ratio to {label_list[0]}", fontsize=14)
    ax[1].set_ylim(-0.2, 0.2)

    plt.legend(fontsize=12)
    plt.xlabel('X_max / grammage', fontsize=12)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()

def plot_lateral_hist_ratio(bins, hist_list, label_list, color_list, title='', xaxis='', ratio_lim=(-1, 1), xlog=True, add_watermark=False):    
    '''
    bins: location of bins (length n)
    hist: array of shape m * n, where m is the number of profiles. each row has n entires, where n describes the histogram bin entry
    '''

    fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]}, figsize=(15,10),sharex=True)
    
    ax[0].tick_params('x', labelbottom=False) # only for last plot
    for hist, label, color in zip(hist_list, label_list, color_list):
        ax[0].fill_between(bins[1:], np.percentile(hist, q=25, axis=0), np.percentile(hist, q=75, axis=0), step='pre', alpha=0.5, color=color)
        ax[0].step(bins[1:], np.percentile(hist, q=50, axis=0), label=label, color=color)
    ax[0].grid(which='major')
    ax[0].set_ylabel('# particles', fontsize=18)
    if (xlog):
        ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[0].legend(bbox_to_anchor=(0, 1, 1, 0), loc="lower left", ncol=len(label_list), fontsize=18)

    if (add_watermark):
        ax[0].text(0.85, 0.85, 'C8 - ICRC2023', horizontalalignment='center', verticalalignment='center', transform = ax[0].transAxes, fontsize=18, alpha=0.5, color='gray')

    for hist, label, color in zip(hist_list[1:], label_list[1:], color_list[1:]):
        # find common X values for base and compare
        vals_base = np.percentile(hist_list[0], q=50, axis=0)
        vals_compare = np.percentile(hist, q=50, axis=0)
        ax[1].step(bins[1:], (vals_compare - vals_base)/vals_base, label=label, color=color)   
    ax[1].grid(which='major')
    fig.subplots_adjust(hspace=0.05)
    ax[1].set_xlabel(xaxis, fontsize=18)
    ax[1].set_ylabel(f"ratio to {label_list[0]}", fontsize=18)
    ax[1].set_ylim(*ratio_lim)
    ax[0].tick_params(axis='both', labelsize=16)
    ax[1].tick_params(axis='both', labelsize=16)
    #fig.suptitle(title, fontsize=18)
    fig.tight_layout()
       


def plot_long_hist_ratio_lpm(X_list, profile_list, X_list_LPM, profile_list_LPM, label_list, color_list, title='', add_watermark=False):    
    '''
    X: len n, which describes the X bins
    profile: array of shape m * n, where m is the number of profiles. every row has n entries which describe the longitudinal profile
    '''
    assert(len(X_list) > 1)

    fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]}, figsize=(15,10),sharex=True)
    
    ax[0].tick_params('x', labelbottom=False) # only for last plot
    for X, profile, label, color in zip(X_list, profile_list, label_list, color_list):
        ax[0].fill_between(X, np.percentile(profile.T, q=25, axis=1), np.percentile(profile.T, q=75, axis=1), step='pre', alpha=0.5, color=color)
        ax[0].step(X, np.percentile(profile.T, q=50, axis=1), label=label, color=color)

    for X, profile, label, color in zip(X_list_LPM, profile_list_LPM, label_list, color_list):
        ax[0].fill_between(X, np.percentile(profile.T, q=25, axis=1), np.percentile(profile.T, q=75, axis=1), step='pre', alpha=0.5, color=color)
        ax[0].step(X, np.percentile(profile.T, q=50, axis=1), label=f"{label} LPM", color=color, linestyle='dashed')

    ax[0].grid(which='both')
    ax[0].set_ylabel('# particles', fontsize=14)
    
    ax[0].legend(bbox_to_anchor=(0, 1, 1, 0), loc="lower left", ncol=2, fontsize=14)

    if (add_watermark):
        ax[0].text(0.85, 0.85, 'C8 - ICRC2023', horizontalalignment='center', verticalalignment='center', transform = ax[0].transAxes, fontsize=18, alpha=0.5, color='gray')

    for X, profile, label, color in zip(X_list[1:], profile_list[1:], label_list[1:], color_list[1:]):
        # find common X values for base and compare
        X_common, indices_base, indices_compare = np.intersect1d(X_list[0], X, return_indices=True)
        vals_base = np.percentile(profile_list[0].T[indices_base], q=50, axis=1)
        vals_compare = np.percentile(profile.T[indices_compare], q=50, axis=1)
        ax[1].step(X_common, (vals_compare - vals_base)/vals_base, label=label, color=color)   


    for X, profile, label, color in zip(X_list_LPM[1:], profile_list_LPM[1:], label_list[1:], color_list[1:]):
        # find common X values for base and compare
        X_common, indices_base, indices_compare = np.intersect1d(X_list_LPM[0], X, return_indices=True)
        vals_base = np.percentile(profile_list_LPM[0].T[indices_base], q=50, axis=1)
        vals_compare = np.percentile(profile.T[indices_compare], q=50, axis=1)
        ax[1].step(X_common, (vals_compare - vals_base)/vals_base, label=label, color=color, linestyle='dashed')   

    ax[1].grid(which='both')
    fig.subplots_adjust(hspace=0.05)
    ax[1].set_xlabel(r"grammage / g/cm²", fontsize=14)
    ax[1].set_ylabel(f"ratio to {label_list[0]}", fontsize=14)
    ax[1].set_ylim(-0.2, 0.2)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
