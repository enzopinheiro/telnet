# Same as model_selection.py but trains and evaluated the model on fixed train and validation years to retrieve the best initial weights
import os
import torch
import h5py
import xarray as xr
import numpy as np
import pandas as pd
import torch.nn as nn
from random import sample, choices
from typing import Dict, List
from copy import deepcopy
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from models.model import TelNet
from modules.submodules import PMI
from utils import DataManager, CreateDataset, EvalMetrics
from utils import DEVICE, exp_data_dir, exp_results_dir
from utils import month2onehot, make_dir, compute_anomalies, scalar2categ, set_seed, printProgressBar
from utilities.plots import plot_error_maps, plot_obs_ypred_maps

def save_torch_model(model, config, n, model_dir, init_weights=False):

    if init_weights:
        torch.save(model.state_dict(), f'{model_dir}/telnet{n:04d}.init.pt')
    else:
        torch.save(model.state_dict(), f'{model_dir}/telnet{n:04d}.pt')

    del model

def load_torch_model(config, H, W, n):
     
    nunits, drop, weight_scale, epochs, lr, clip, batch_size, nfeats, time_steps, lead, nmembs = config

    telnet = TelNet(nmembs, H, W, nfeats, nunits, time_steps, lead, drop, weight_scale)
    telnet.load_state_dict(torch.load(f'{exp_data_dir}/pretrained/telnet{n:04d}.pt'))
    
    return telnet.to(DEVICE)

def identify_checkpoint(arr, seed_n, result_dir):
 
    n = 0
    checkpoint_file = [i for i in os.listdir(result_dir) if i.startswith(f'checkpoint_{seed_n}_')]
    if len(checkpoint_file) != 0:
        # print ('Restarting the search from the last checkpoint ...', end = "\r")
        checkpoint_file = checkpoint_file[-1]
        n = int(checkpoint_file.split('_')[-1].split('.')[0])
        with h5py.File(f'{result_dir}/checkpoint_{seed_n}_{n}.h5', 'r') as hf:
            data = hf['dataset'][:]
            arr[:] = data
        n+=1
   
    return arr, n
 
def update_checkpoint(arr, n, seed_n, result_dir):
 
    with h5py.File(f'{result_dir}/checkpoint_{seed_n}_{n}.h5', 'w') as hf:
        hf.create_dataset('dataset', data=arr)
   
    if os.path.exists(f'{result_dir}/checkpoint_{seed_n}_{n-1}.h5'):
        os.remove(os.path.join(result_dir, f'checkpoint_{seed_n}_{n-1}.h5'))

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

    Xdm['cov'].compute_statistics(stat_yrs, ['mean', 'std'])
    Xdm['cov'].to_anomalies('var', stdized=True)
    
    Ydm['var_seas'] = Ydm['var']
    Ydm.monthly2seasonal('var_seas', 'sum', True)
    Ydm.compute_statistics(stat_yrs, ['mean', 'std'], 'var_seas')
    Ydm.to_anomalies('var_seas', stdized=True)
    Ydm.compute_statistics(stat_yrs, ['terciles'], 'var_seas') 

    return Xdm, Ydm

def split_dataset(
        Xdm: DataManager, 
        Ydm: DataManager, 
        train_yrs: np.ndarray, 
        val_yrs: np.ndarray, 
        nfeats: int,
        time_steps: int, 
        lead: int
    ):

    Xdm['cov']['var'] = Xdm['cov']['var'].isel(indices=slice(0, nfeats))

    Xdm['auto'].add_seq_dim('var_seas', time_steps)
    Xdm['auto'].replace_nan('var_seas', -999.)

    Xdm['cov'].add_seq_dim('var', time_steps)
    Xdm['cov'].replace_nan('var', -999.)

    Ydm.add_seq_dim('var_seas', lead, 'lead')
    Ydm.replace_nan('var_seas', -999.)

    time_range = pd.date_range(Xdm['auto']['var_seas'].time.values[0], 
                               Ydm['var_seas'].time.values[-2], freq='MS')
    X_train_samples = [i.to_datetime64() for i in time_range if i.year in train_yrs]
    Y_train_samples = [(i+pd.DateOffset(months=1)).to_datetime64() for i in time_range if i.year in train_yrs]
    X_val_samples = [i.to_datetime64() for i in time_range if i.year in val_yrs]
    Y_val_samples = [(i+pd.DateOffset(months=1)).to_datetime64() for i in time_range if i.year in val_yrs]
    
    # Making sure no season is in both train and val
    X_train_samples = [i for i in X_train_samples 
                       if not np.isin(pd.date_range(start=i, periods=lead, freq='MS'), X_val_samples).any()
                          and
                          not np.isin(pd.date_range(end=i, periods=lead, freq='MS'), X_val_samples).any()]
    Y_train_samples = [i for i in Y_train_samples 
                       if not np.isin(pd.date_range(start=i, periods=lead, freq='MS'), Y_val_samples).any()
                          and
                          not np.isin(pd.date_range(end=i, periods=lead, freq='MS'), Y_val_samples).any()]

    Xdm['auto'].create_subsamples(['train', 'val'], [X_train_samples, X_val_samples], 'var_seas')
    Xdm['cov'].create_subsamples(['train', 'val'], [X_train_samples, X_val_samples], 'var')
    Ydm.create_subsamples(['train', 'val'], [Y_train_samples, Y_val_samples], 'var_seas')

    Xdm = {'auto': Xdm['auto'], 'cov': Xdm['cov']}

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

def read_sample_files(seed_n, train_yrs=None, val_yrs=None, test_yrs=None):

    if os.path.exists(f'{exp_data_dir}/test_years_{seed_n}.txt'):
        test_yrs = np.sort(np.loadtxt(f'{exp_data_dir}/test_years_{seed_n}.txt', dtype=int))
    else:
        np.savetxt(f'{exp_data_dir}/test_years_{seed_n}.txt', test_yrs, fmt='%i')
    if os.path.exists(f'{exp_data_dir}/val_years_{seed_n}.txt'):
        val_yrs = np.sort(np.loadtxt(f'{exp_data_dir}/val_years_{seed_n}.txt', dtype=int))
    else:
        np.savetxt(f'{exp_data_dir}/val_years_{seed_n}.txt', val_yrs, fmt='%i')
    if os.path.exists(f'{exp_data_dir}/train_years_{seed_n}.txt'):
        train_yrs = np.sort(np.loadtxt(f'{exp_data_dir}/train_years_{seed_n}.txt', dtype=int))
    else:
        np.savetxt(f'{exp_data_dir}/train_years_{seed_n}.txt', train_yrs, fmt='%i')
    return train_yrs, val_yrs, test_yrs

def training(X: Dict[str, DataManager], 
             Y: DataManager, 
             model_config:list,
             seed_n:int):
    
    nunits, drop, weight_scale, epochs, lr, clip, batch_size, nfeats, time_steps, lead, nmembs = model_config
    B = Y['train'].shape[0]
    H = X['auto']['train'].shape[-2]
    W = X['auto']['train'].shape[-1]
    I = nfeats
    D = nunits
    T = time_steps
    L = lead
    
    lats2d = Y.lat[:, None].repeat(W, 1)
    lat_wgts = torch.as_tensor(np.cos(np.deg2rad(lats2d)), device=DEVICE)
    lat_wgts = lat_wgts / lat_wgts.mean()
    lat_wgts = lat_wgts[None, None].tile((1, L, 1, 1))

    Xstatic = month2onehot(Y['train']['time.month'].values)
    Xtrain = [X['auto']['train'].values, X['cov']['train'].values, Xstatic]
    Ytrain = Y['train'].values
    train_dataset = CreateDataset(Xtrain, Ytrain)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    telnet = TelNet(nmembs, H, W, I, D, T, L, drop, weight_scale).to(DEVICE)
    save_torch_model(telnet, model_config, seed_n, f'{exp_data_dir}/models', init_weights=True)
    telnet.train_model(train_dataloader, epochs, clip, lr=lr, lat_wgts=lat_wgts)
    
    return telnet

def inference(X: Dict[str, DataManager], 
              Y: DataManager,
              telnet: nn.ModuleList):
    
    Xstatic_val = month2onehot(Y['val']['time.month'].values)
    Xval = [
        torch.as_tensor(X['auto']['val'].values.astype(np.float32), device=DEVICE), 
        torch.as_tensor(X['cov']['val'].values.astype(np.float32), device=DEVICE),
        torch.as_tensor(Xstatic_val.astype(np.float32), device=DEVICE)
    ]
    xmask = torch.where(Xval[0][0:1, 0:1] == -999., 0., 1.).to(DEVICE)
    ypred, wgts = telnet.inference(Xval, xmask)  # B, L, N, H, W
    Ypred = ypred.detach().cpu().numpy()
    Wgts = wgts.detach().cpu().numpy()

    return Ypred, Wgts

def evaluate(Y: DataManager, 
             Ypred: np.ndarray):
    
    init_months = [1, 4, 7, 10]  # time step 3 is the second non-overlapping season [DJF, MAM, JJA, SON]
    lead = 6
    nmembs = Ypred.shape[2]

    RPS = np.full((4, 6), np.nan)

    for n, i in enumerate(init_months):
        idcs = np.where((Y['val']['time.month'].values == i))[0]
        Yval_i = Y['val'][idcs]
        Yval_i = xr.concat(
            [Yval_i.sel(time=j).drop('time').squeeze().assign_coords({'time_seq': pd.date_range(j, periods=lead, freq='MS')}).rename(time_seq='time') 
             for j in Yval_i.time.values], dim='time'
        )
        Yval_i = xr.where(Yval_i==-999., np.nan, Yval_i)

        Ypred_i = Ypred[idcs].reshape(-1, Ypred.shape[-3], Ypred.shape[-2], Ypred.shape[-1])
        Yq33 = Y.q33.sel(time=Yval_i.time.values).values
        Yq66 = Y.q66.sel(time=Yval_i.time.values).values
        Ymn = Y.mn.sel(time=Yval_i.time.values).values
        Ystd = Y.std.sel(time=Yval_i.time.values).values
        
        Yval_i_categ, _, _ = scalar2categ(Yval_i.values, 3, 'one-hot', Yq33, Yq66)
        Ypred_i_probs, _, _ = scalar2categ(Ypred_i, 3, 'one-hot', Yq33, Yq66, count=True)

        for l in range(lead):
            RPS[n, l] = EvalMetrics.RPS(Yval_i_categ[l::lead].transpose(1, 0, 2, 3), Ypred_i_probs[l::lead].transpose(1, 0, 2, 3), ax=np.s_[0, 1, 2])

    return RPS

def plot_error_maps_fcts(Y: DataManager, 
                         Ypred: np.ndarray, 
                         varsel_wgts: np.ndarray,
                         var_names: List[str],
                         result_dir: str,
                         plot_ypred: bool=False):
    
    init_months = [1, 10]  # time step 3 is the second non-overlapping season [DJF, MAM] (lead 2 in seasonal forecasting)
    lead = 6
    nmembs = Ypred.shape[2]
    x, y = np.meshgrid(Y.lon, Y.lat)

    for n, i in enumerate(init_months):
        idcs = np.where((Y['val']['time.month'].values == i))[0]
        Yval_i = Y['val'][idcs]
        Yval_i = xr.concat(
            [Yval_i.sel(time=j).drop('time').squeeze().assign_coords({'time_seq': pd.date_range(j, periods=lead, freq='MS')}).rename(time_seq='time') 
             for j in Yval_i.time.values], dim='time'
        )
        Yval_i = xr.where(Yval_i==-999., np.nan, Yval_i)
        years = Yval_i[0::lead]['time.year'].values

        vsel_wgts_i = varsel_wgts[idcs].reshape(-1, varsel_wgts.shape[-1])

        Ypred_i = Ypred[idcs].reshape(-1, Ypred.shape[-3], Ypred.shape[-2], Ypred.shape[-1])
        Yq33 = Y.q33.sel(time=Yval_i.time.values).values
        Yq66 = Y.q66.sel(time=Yval_i.time.values).values
        Ymn = Y.mn.sel(time=Yval_i.time.values).values
        Ystd = Y.std.sel(time=Yval_i.time.values).values
        
        Yval_i_categ, _, _ = scalar2categ(Yval_i.values, 3, 'one-hot', Yq33, Yq66)
        Ypred_i_probs, _, _ = scalar2categ(Ypred_i, 3, 'one-hot', Yq33, Yq66, count=True)
        Yval_i_total = compute_anomalies(Yval_i.values, Ymn, Ystd, reverse=True)
        Ypred_i_total = compute_anomalies(Ypred_i, np.tile(Ymn[:, None], (1, nmembs, 1, 1)), np.tile(Ystd[:, None], (1, nmembs, 1, 1)), reverse=True)
        
        rmse = EvalMetrics.RMSE(Yval_i.values[3::lead], Ypred_i.mean(1)[3::lead]) 
        rps = EvalMetrics.RPS(Yval_i_categ[3::lead].transpose(1, 0, 2, 3), Ypred_i_probs[3::lead].transpose(1, 0, 2, 3))

        if i == 10:
            title = 'Nov-DJF'
        if i == 1:
            title = 'Feb-MAM'
        
        plot_error_maps([[x, y]], [rmse], ['TelNet'], f'{result_dir}/RMSE_{title}.png', '', 1, 1, (5, 10), 'RMSE', True, cbar_orientation='vertical')
        plot_error_maps([[x, y]], [rps], ['TelNet'], f'{result_dir}/RPS_{title}.png', '', 1, 1, (5, 10), 'RPS', cbar_orientation='vertical')
        
        if plot_ypred:
            if i == 10 or i ==1:
                make_dir(f'{result_dir}/forecasts')
                for n, yr in enumerate(years):
                    plot_obs_ypred_maps(x, y, [yr], 
                                    Yval_i_total[3::lead][n:n+1], Ypred_i_total.mean(1)[3::lead][n:n+1], 
                                    Yval_i.values[3::lead][n:n+1], Ypred_i.mean(1)[3::lead][n:n+1], 
                                    Yval_i_categ[3::lead][n:n+1], Ypred_i_probs[3::lead][n:n+1],
                                    vsel_wgts_i[3::lead][n], var_names, '', 
                                    f'forecast_{yr}_{title}.png', f'{result_dir}/forecasts', (13, 7))

def split_validation(
        X: Dict[str, DataManager], 
        Y: DataManager, 
        train_samples, 
        val_samples,
        model_config: list,
        seed_n: int
    ):
    
    nfeats = model_config[-4]
    time_steps = model_config[-3]
    lead = model_config[-2]
    
    Xdm, Ydm = split_dataset(X, Y, train_samples, val_samples, nfeats, time_steps, lead)

    telnet = training(Xdm, Ydm, model_config, seed_n)
    Ypred, Wgts = inference(Xdm, Ydm, telnet)
    RPS = evaluate(Ydm, Ypred)

    del telnet
    torch.cuda.empty_cache()

    return RPS

def main(arguments):

    """
    seed_n: int, 
    X: Dict[str, DataManager], 
    Y: DataManager, 
    config: np.ndarray,
    seeds: np.ndarray, 
    feat_order: np.ndarray
    """

    seed_n, X, Y, config, seeds, feat_order = arguments
    
    seed_pos = np.argwhere(seeds == seed_n).flatten()[0]
    set_seed(seed_n)

    print ('Sampling years ...', end = "\r")
    test_yrs = np.arange(2003, 2023)
    train_yrs, val_yrs, test_yrs = split_sample(np.arange(1941, 2002), test_yrs)
    train_yrs, val_yrs, test_yrs = read_sample_files(0, train_yrs, val_yrs, test_yrs)  # uses seed 0 train and validation years files
    print (f'Preprocessing data n={seed_pos} ...', end = "\r")
    X, Y = preprocess_data(X, Y, train_yrs)
    X['cov']['var'] = X['cov']['var'].sel(indices=feat_order)
    
    checkpoints_dir = f'{exp_data_dir}/checkpoints_init_weights'
    ERRORS = np.zeros((1, 4, 6))  # 1 error, 4 init_months, 6 lead times
    ERRORS, n = identify_checkpoint(ERRORS, seed_n, checkpoints_dir)
    if n == 1:
        return ERRORS
    rps = split_validation(deepcopy(X), deepcopy(Y), train_yrs, val_yrs, config, seed_n)
    ERRORS[0] = rps
    update_checkpoint(ERRORS, n, seed_n, checkpoints_dir)

    return ERRORS

if __name__ == '__main__':

    main()
