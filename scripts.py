import numpy as np
import matplotlib as mpl
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

    fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]}, figsize=(8,6),sharex=True)
    
    ax[0].tick_params('x', labelbottom=False) # only for last plot
    for X, profile, label, color in zip(X_list, profile_list, label_list, color_list):
        ax[0].fill_between(X, np.percentile(profile.T, q=25, axis=1), np.percentile(profile.T, q=75, axis=1), step='pre', alpha=0.5, color=color)
        ax[0].step(X, np.percentile(profile.T, q=50, axis=1), label=label, color=color)
    ax[0].grid(which='both')
    ax[0].set_ylabel('# particles', fontsize=14)
    
    ax[0].legend(bbox_to_anchor=(0, 1, 1, 0), loc="lower left", ncol=2, fontsize=18)

    if (add_watermark):
        ax[0].text(0.85, 0.85, 'C8 - ICRC2023', horizontalalignment='center', verticalalignment='center', transform = ax[0].transAxes, fontsize=18, alpha=0.5, color='gray')


    for X, profile, label, color in zip(X_list[1:], profile_list[1:], label_list[1:], color_list[1:]):
        # find common X values for base and compare
        X_common, indices_base, indices_compare = np.intersect1d(X_list[0], X, return_indices=True)
        vals_base = np.percentile(profile_list[0].T[indices_base], q=50, axis=1)
        vals_compare = np.percentile(profile.T[indices_compare], q=50, axis=1)
        ax[1].step(X_common, vals_compare/vals_base, label=label, color=color)   
    ax[1].grid(which='both')
    fig.subplots_adjust(hspace=0.05)
    ax[1].set_xlabel(r"grammage / g/cm²", fontsize=14)
    ax[1].set_ylabel(f"ratio to {label_list[0]}", fontsize=14)
    ax[1].set_ylim(0.8, 1.2)
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
    fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]}, figsize=(8,6), sharex=True)
    
    ax[0].tick_params('x', labelbottom=False) # only for last plot
    
    X_max_lists = []
   
    X_max_mean_list = []

    min_bin = +np.inf
    max_bin = -np.inf
   

    for X, profile in zip(X_list, profile_list):
        X_max = []
        for prof in profile:
            X_max.append(X[np.argmax(prof)])
        min_bin = min(min_bin, min(X_max))
        max_bin = max(max_bin, max(X_max))
        X_max_lists.append(X_max)
        X_max_mean_list.append(np.mean(X_max))
        
    bins = np.linspace(min_bin, max_bin, NUM_BINS)
    n_list = []
    
    for X_max_list, label, color, X_max_mean in zip(X_max_lists, label_list, color_list, X_max_mean_list):
        n, _, _ = ax[0].hist(X_max_list, bins=bins, histtype='step', color=color, label=r"{0}, $\overline{{X_\mathrm{{max}}}}$ = {1:.2f} g/cm²".format(label, X_max_mean))
        n_list.append(n)
        
    ax[0].grid(which='both')
    ax[0].set_ylabel('# showers', fontsize=14)
    ax[0].legend(bbox_to_anchor=(0, 1, 1, 0), loc="lower left", ncol=1, fontsize=18)
    
    if (add_watermark):
        ax[0].text(0.85, 0.85, 'C8 - ICRC2023', horizontalalignment='center', verticalalignment='center', transform = ax[0].transAxes, fontsize=18, alpha=0.5, color='gray')

    for n, label, color in zip(n_list[1:], label_list[1:], color_list[1:]):
        ax[1].step(bins[1:], n / n_list[0], color=color)
    ax[1].grid(which='both')
    fig.subplots_adjust(hspace=0.05)
    ax[1].set_xlabel(r"grammage / g/cm²", fontsize=14)
    ax[1].set_ylabel(f"ratio to {label_list[0]}", fontsize=14)
    ax[1].set_ylim(0.8, 1.2)

    plt.xlabel(r'$X_\mathrm{max}$ / g/cm²', fontsize=14)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()

def plot_lateral_hist_ratio(bins, hist_list, label_list, color_list, title='', xaxis='', ratio_lim=(0, 2), xlog=True, add_watermark=False):    
    '''
    bins: location of bins (length n)
    hist: array of shape m * n, where m is the number of profiles. each row has n entires, where n describes the histogram bin entry
    '''

    fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]}, figsize=(8,6),sharex=True)
    
    ax[0].tick_params('x', labelbottom=False) # only for last plot
    for hist, label, color in zip(hist_list, label_list, color_list):
        ax[0].fill_between(bins[1:], np.percentile(hist, q=25, axis=0), np.percentile(hist, q=75, axis=0), step='pre', alpha=0.5, color=color)
        ax[0].step(bins[1:], np.percentile(hist, q=50, axis=0), label=label, color=color)
    ax[0].grid(which='major')
    ax[0].set_ylabel('# particles', fontsize=14)
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
        ax[1].step(bins[1:], vals_compare/vals_base, label=label, color=color)   
    ax[1].grid(which='major')
    fig.subplots_adjust(hspace=0.05)
    ax[1].set_xlabel(xaxis, fontsize=14)
    ax[1].set_ylabel(f"ratio to {label_list[0]}", fontsize=14)
    ax[1].set_ylim(*ratio_lim)
    #ax[0].tick_params(axis='both', labelsize=14)
    #ax[1].tick_params(axis='both', labelsize=14)
    #fig.suptitle(title, fontsize=18)
    fig.tight_layout()
       
def plot_lateral_2d(bins_x, bins_y, hist_raw, label_list, xlabel, ylabel, add_watermark=False, contour=True):
    NUM_PLOTS = len(label_list) # how many subplots?
    fig, axes = plt.subplots(nrows=1, ncols=NUM_PLOTS, sharey=True, figsize=(4 * NUM_PLOTS, 4))

    medians = []
    for hist in hist_raw:
        medians.append(np.percentile(hist, q=50, axis=0)) # calculate medians out of all 2d histograms for every bin

    max_bin = 0 # calculate biggest entry of histograms
    for hist in medians:
        max_bin = max(max_bin, max(hist.flatten()))

    for ax, hist, label in zip(axes, medians, label_list):
        im = ax.pcolormesh(bins_x, bins_y, hist.T, norm=mpl.colors.LogNorm(vmin=1e-0, vmax=max_bin))
        if (contour):
            levels = np.logspace(1, np.floor(np.log10(max_bin)), int(np.floor(np.log10(max_bin))))
            #fmt = mpl.ticker.LogFormatterMathtext() # scientific notation in clabel
            #fmt.create_dummy_axis()
            CS = ax.contour(10**((np.log10(bins_x)[:-1] + np.log10(bins_x)[1:]) / 2), 
                       10**((np.log10(bins_y)[:-1] + np.log10(bins_y)[1:]) / 2), 
                       hist.T, linewidths=1, levels=levels, alpha=0.7, colors='black')
            #plt.clabel(CS, inline=1, fontsize=10, fmt=fmt) # this doesn't work as intended...
        ax.set_title(label, fontsize=18)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.xaxis.set_minor_locator(mpl.ticker.LogLocator(numticks=999, subs="auto"))
        ax.yaxis.set_minor_locator(mpl.ticker.LogLocator(numticks=999, subs="auto"))
        ax.set_facecolor('silver')
        if (add_watermark):
            ax.text(0.71, 0.95, 'C8 - ICRC2023', horizontalalignment='center', verticalalignment='center', transform = ax.transAxes, fontsize=14, alpha=0.5, color='gray')

    axes[0].set_ylabel(ylabel, fontsize=12) # shared y axis, so label only for first axis

    fig.subplots_adjust(right=0.8, wspace=0.1) # make space for colorbar
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)

def plot_lateral_2d_ratio(bins_x, bins_y, hist_raw, label_list, xlabel, ylabel, add_watermark=False):
    NUM_PLOTS = len(label_list) - 1 # how many subplots?
    fig, axes = plt.subplots(nrows=1, ncols=NUM_PLOTS, sharey=True, figsize=(4 * NUM_PLOTS, 4))

    if (NUM_PLOTS==1):
        axes = [axes] # if we only create one plot, axes is a single object. but this script expects a list of axes.

    medians = []
    for hist in hist_raw:
        medians.append(np.percentile(hist, q=50, axis=0)) # calculate medians out of all 2d histograms for every bin

    baseline = medians[0]

    ratios = []
    for median in medians[1:]:
        print(np.shape(median))
        ratios.append((median - baseline)/baseline)


    for ax, ratio, label in zip(axes, ratios, label_list[1:]):
        im = ax.pcolormesh(bins_x, bins_y, ratio.T, vmin=-1, vmax=1, cmap='seismic')
        ax.set_title(f'{label_list[0]} vs {label}', fontsize=14)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.xaxis.set_minor_locator(mpl.ticker.LogLocator(numticks=999, subs="auto"))
        ax.yaxis.set_minor_locator(mpl.ticker.LogLocator(numticks=999, subs="auto"))
        ax.set_facecolor('silver')
        if (add_watermark):
            ax.text(0.71, 0.95, 'C8 - ICRC2023', horizontalalignment='center', verticalalignment='center', transform = ax.transAxes, fontsize=14, alpha=0.5, color='gray')


    axes[0].set_ylabel(ylabel, fontsize=12) # shared y axis, so label only for first axis

    fig.subplots_adjust(right=0.75, wspace=0.10) # make space for colorbar
    cbar_ax = fig.add_axes([0.8, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)




def plot_long_hist_ratio_lpm(X_list, profile_list, X_list_LPM, profile_list_LPM, label_list, color_list, title='', add_watermark=False):    
    '''
    X: len n, which describes the X bins
    profile: array of shape m * n, where m is the number of profiles. every row has n entries which describe the longitudinal profile
    '''
    assert(len(X_list) > 1)

    fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]}, figsize=(8,6),sharex=True)
    
    ax[0].tick_params('x', labelbottom=False) # only for last plot
    for X, profile, label, color in zip(X_list, profile_list, label_list, color_list):
        ax[0].fill_between(X, np.percentile(profile.T, q=25, axis=1), np.percentile(profile.T, q=75, axis=1), step='pre', alpha=0.5, color=color)
        ax[0].step(X, np.percentile(profile.T, q=50, axis=1), label=label, color=color)

    for X, profile, label, color in zip(X_list_LPM, profile_list_LPM, label_list, color_list):
        ax[0].fill_between(X, np.percentile(profile.T, q=25, axis=1), np.percentile(profile.T, q=75, axis=1), step='pre', alpha=0.5, color=color)
        ax[0].step(X, np.percentile(profile.T, q=50, axis=1), label=f"{label} LPM", color=color, linestyle='dashed')

    ax[0].grid(which='both')
    ax[0].set_ylabel('# particles', fontsize=14)
    
    ax[0].legend(bbox_to_anchor=(0, 1, 1, 0), loc="lower left", ncol=2, fontsize=18)

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
        ax[1].step(X_common, vals_compare/vals_base, label=label, color=color, linestyle='dashed')   

    ax[1].grid(which='both')
    fig.subplots_adjust(hspace=0.05)
    ax[1].set_xlabel(r"grammage / g/cm²", fontsize=14)
    ax[1].set_ylabel(f"ratio to {label_list[0]}", fontsize=14)
    ax[1].set_ylim(0.8, 1.2)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
