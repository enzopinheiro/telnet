from collections import Counter
from copy import deepcopy
from datetime import datetime
import os
from random import choices, sample
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import xarray as xr
from modules.submodules import PMI
from typing import Dict, List
from utilities.plots import plot_feat_freq
from utils import DataManager, get_search_matrix, make_dir, read_obs_data, set_seed
from utils import exp_data_dir, exp_results_dir, DEVICE
from itertools import combinations, repeat
import pandas as pd
from multiprocessing import Pool


def preprocess_data(
        Xdm: Dict[str, DataManager], 
        Ydm: DataManager, 
        train_yrs: np.ndarray
    ):
    stat_yrs = np.asarray([i for i in train_yrs if i>=1971 and i<=2020])
    Xdm['auto']['var_seas'] = Xdm['auto']['var']
    Xdm['auto'].monthly2seasonal('var_seas', 'sum', True)
    Xdm['auto'].compute_statistics(stat_yrs, ['mean', 'std'], 'var_seas')
    Xdm['auto'].to_anomalies('var_seas', stdized=True)
    Xdm['auto'].apply_detrend()
    
    # Xdm['cov'].compute_statistics(stat_yrs, ['mean', 'std'])
    # Xdm['cov'].to_anomalies('var', stdized=True)
    Xdm['cov'].apply_detrend()

    Ydm['var_seas'] = Ydm['var']
    Ydm.monthly2seasonal('var_seas', 'sum', True)
    Ydm.compute_statistics(stat_yrs, ['mean', 'std'], 'var_seas')
    Ydm.to_anomalies('var_seas', stdized=True)
    Ydm.apply_detrend()
    Ydm.compute_statistics(stat_yrs, ['terciles'], 'var_seas') 

    return Xdm, Ydm


def split_sample(samples, test_samples=None):

    if test_samples is None:
        test_samples = np.array(sample(range(1982, 2024), 20))
        # np.savetxt(f'{exp_data_dir}/test_years.txt', test_samples, fmt='%i')
    train_samples = np.array([i for i in samples if (i not in test_samples)])
    train_samples, val_samples = train_test_split(train_samples, test_size=0.2)
    train_samples = np.sort(train_samples)
    val_samples = np.sort(np.append(val_samples, [2002]))
    test_samples = choices(test_samples, k=len(test_samples))
    # np.savetxt(f'{exp_data_dir}/val_years.txt', val_samples, fmt='%i')
    # np.savetxt(f'{exp_data_dir}/train_years.txt', train_samples, fmt='%i')
        
    return train_samples, val_samples, test_samples

def exec_selection(arguments):
    """
    Run the input variable selection process.
    """
    Xsel, Ysel, index_names, Xval, Yval = arguments
    
    if np.isnan(Ysel).all():
        return list(index_names)

    # ivs = PartialCorrelation(DEVICE)
    ivs = PMI(DEVICE)
    ivs.fit(Xsel, Ysel, index_names, Xval=Xval, yval=Yval)
    feat_order = list(ivs.scores.keys())

    return feat_order

def variable_selection(Xdm: Dict[str, DataManager],
                       Ydm: DataManager,
                       train_yrs: np.ndarray,
                       val_yrs: np.ndarray):

    # Feature selection
    time_range = pd.date_range(Xdm['auto']['var'].time.values[0], 
                                Ydm['var_seas'].time.values[-2], freq='MS')
    X_train_samples = [i.to_datetime64() for i in time_range if i.year in train_yrs
                        if (i+pd.DateOffset(months=4)).to_datetime64() in Ydm['var_seas']['time'].values]
    Y_train_samples = [(i+pd.DateOffset(months=4)).to_datetime64() for i in time_range if i.year in train_yrs
                        if (i+pd.DateOffset(months=4)).to_datetime64() in Ydm['var_seas']['time'].values]

    X_val_samples = [i.to_datetime64() for i in time_range if i.year in val_yrs
                        if (i+pd.DateOffset(months=4)).to_datetime64() in Ydm['var_seas']['time'].values]
    Y_val_samples = [(i+pd.DateOffset(months=4)).to_datetime64() for i in time_range if i.year in val_yrs
                        if (i+pd.DateOffset(months=4)).to_datetime64() in Ydm['var_seas']['time'].values]

    Xsel = Xdm['cov']['var'].sel(time=X_train_samples).values  # time, indices
    Ysel = Ydm['var_seas'].sel(time=Y_train_samples).mean(('lat', 'lon')).values  # time

    Xval = Xdm['cov']['var'].sel(time=X_val_samples).values  # time, indices
    Yval = Ydm['var_seas'].sel(time=Y_val_samples).mean(('lat', 'lon')).values  # time
    
    index_names = Xdm['cov']['var']['indices'].values

    feat_order = exec_selection((Xsel, Ysel, index_names, Xval, Yval))
    
    return feat_order

def main(nsamples, reproduce_paper=True):


    """
    seed_n: int, 
    X: Dict[str, DataManager], 
    Y: DataManager, 
    search_arr: np.ndarray,
    seeds: np.ndarray, 
    nmodels: int
    """

    torch.multiprocessing.set_start_method('spawn')
    init_time = datetime.now()

    result_dir = f'{exp_results_dir}/selection/'
    make_dir(result_dir)
    checkpoints_dir = f'{exp_data_dir}/checkpoints_sel/'
    make_dir(checkpoints_dir)
    print ('Reading data ...')
    X, Y, idcs_list = read_obs_data()

    if os.path.exists(os.path.join(exp_data_dir, 'seeds_pmi.txt')):
        # Read the existing seeds file
        seeds = np.loadtxt(os.path.join(exp_data_dir, 'seeds_pmi.txt'), delimiter=',', dtype=int)
    else:
        if reproduce_paper:
            seeds = np.arange(nsamples)
        else:
            seeds = sample(range(100000), nsamples)
        # Save seeds as a text file
        np.savetxt(os.path.join(exp_data_dir, 'seeds_pmi.txt'), seeds, delimiter=',', fmt='%i')

    full_feat_order = []

    for seed_n in seeds:

        seed_pos = np.argwhere(seeds == seed_n).flatten()[0]
        set_seed(seed_n)

        print ('Sampling years ...')
        test_yrs = np.arange(2003, 2023)
        train_yrs, val_yrs, test_yrs = split_sample(np.arange(1941, 2002), test_yrs)

        print (f'Preprocessing data n={seed_pos} ...')
        Xin, Yin = preprocess_data(deepcopy(X), deepcopy(Y), train_yrs)
        print (f'Ranking variables n={seed_pos} ...')
        feat_order = variable_selection(Xin, Yin, train_yrs, val_yrs)

        full_feat_order.append(feat_order)
    
    full_feat_order = np.stack(full_feat_order, axis=0)  # (nsamples, indices)
    plot_feat_freq(full_feat_order, result_dir)
    
    final_feats = []

    for l in range(full_feat_order.shape[1]):
        feat_freq = Counter(full_feat_order[:, l]).most_common(full_feat_order.shape[1])
        print (feat_freq)
        cond = False
        k = 0
        while cond == False:
            if feat_freq[k][0] not in final_feats:
                final_feats.append(feat_freq[k][0])  # Most frequent feature at position l
                cond = True
            else:
                k += 1
    print (final_feats)

    np.save(os.path.join(f'{exp_data_dir}/models/', f'final_feats.npy'), final_feats, allow_pickle=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Feature pre-selection script.')
    parser.add_argument('-n','--number', help='Sample size', required=True, default=1000)

    args = parser.parse_args()

    main(
        nsamples=int(args.number),
    )