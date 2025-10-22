import os
from sklearn.decomposition import PCA
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
from utils import DataManager, CreateDataset, EvalMetrics
from utils import DEVICE, exp_data_dir, exp_results_dir
from utils import month2onehot, make_dir, compute_anomalies, scalar2categ, set_seed, printProgressBar
from utilities.plots import plot_error_maps, plot_obs_ypred_maps


def save_varsel_wgts(VSweights, init_month, lead, feat_order_dict, nfeats, outdir, fname):
    dims_vs = ['init_month', 'lead', 'index']
    dvars = {'VSweights': (dims_vs, VSweights)}
    coords = {'init_month': (['init_month'], init_month),
              'lead': (['lead'], np.arange(lead)),
              'index': (['index'], np.append(['ylag'], feat_order_dict[:nfeats]))}
    ds = xr.Dataset(data_vars=dvars, coords=coords)
    ds.to_netcdf(f'{outdir}/{fname}.nc')

def identify_checkpoint(metric_name, seed_n, result_dir):
 
    n = 0
    checkpoint_file = [i for i in os.listdir(result_dir) if i.startswith(f'{metric_name}_{seed_n:04d}_')]
    if len(checkpoint_file) != 0:
        checkpoint_file = checkpoint_file[-1]
        n = int(checkpoint_file.split('_')[-1].split('.')[0])
        arr = np.load(os.path.join(result_dir, checkpoint_file), allow_pickle=True)
        n+=1
    else:
        arr = None
   
    return arr, n
 
def update_checkpoint(arr, metric_name, n, seed_n, result_dir):
        
    np.save(f'{result_dir}/{metric_name}_{seed_n:04d}_{n:04d}.npy', arr)

    if os.path.exists(f'{result_dir}/{metric_name}_{seed_n:04d}_{n-1:04d}.npy'):
        os.remove(os.path.join(result_dir, f'{metric_name}_{seed_n:04d}_{n-1:04d}.npy'))

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

def split_dataset(
        Xdm: DataManager, 
        Ydm: DataManager, 
        train_yrs: np.ndarray, 
        val_yrs: np.ndarray, 
        model_predictors: np.ndarray,
        nfeats: int,
        time_steps: int, 
        lead: int
    ):

    Xdm['cov']['var'] = Xdm['cov']['var'].sel(indices=model_predictors).isel(indices=slice(0, nfeats))

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
    yrs = [i for i in samples if (i not in test_samples)]
    val_samples = sample(yrs, k=13)
    train_samples = [i for i in yrs if i not in val_samples]
    train_samples = np.sort(np.append(train_samples, [1941]))
    val_samples = np.sort(np.append(val_samples, [2002]))
    test_samples = choices(test_samples, k=len(test_samples))
    # np.savetxt(f'{exp_data_dir}/val_years.txt', val_samples, fmt='%i')
    # np.savetxt(f'{exp_data_dir}/train_years.txt', train_samples, fmt='%i')
        
    return train_samples, val_samples, test_samples

def training(X: Dict[str, DataManager], 
             Y: DataManager, 
             model_config:list,
             DEVICE:str):
    
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
    train_dataset = CreateDataset(Xtrain, Ytrain, DEVICE)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    Xstatic_val = month2onehot(Y['val']['time.month'].values)
    Xval = [X['auto']['val'].values, X['cov']['val'].values, Xstatic_val]
    Yval = Y['val'].values
    val_dataset = CreateDataset(Xval, Yval, DEVICE)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    telnet = TelNet(nmembs, H, W, I, D, T, L, drop, weight_scale).to(DEVICE)
    telnet.train_model(train_dataloader, epochs, clip, lr=lr, lat_wgts=lat_wgts, val_dataloader=val_dataloader)
    
    return telnet

def inference(X: Dict[str, DataManager], 
              Y: DataManager,
              telnet: nn.ModuleList,
              DEVICE:str):
    
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
             Ypred: np.ndarray,
             vswgts: np.ndarray,
             init_months: list=[1, 4, 7, 10],
             lead: int=6):

    lats = Y['val'].lat.values
    lons = Y['val'].lon.values
    nfeats = vswgts.shape[-1]
    nyrs = Y['val'].shape[0]//12
    nvalid_points = np.where(Y['val'].sel(lat=lats, lon=lons).values[0, 0].reshape(-1)!=-999.)[0].shape[0]
    nmembs = Ypred.shape[2]
    x, y = np.meshgrid(lons, lats)
    
    RMSE = np.full((len(init_months), lead, len(lats), len(lons)), np.nan)
    RPS = np.full((len(init_months), lead, len(lats), len(lons)), np.nan)
    SSR = np.full((len(init_months), lead, len(lats), len(lons)), np.nan)
    # Rank histogram
    ranks = np.full((len(init_months), lead, nyrs*nvalid_points), np.nan)
    ranks_ext = np.full((len(init_months), lead, nyrs*nvalid_points), np.nan)
    # Reliability and sharpness diagrams
    ncategs = 3
    # PCA
    npcs = 2
    # bins = np.linspace(0., 1., 11)
    bins = np.linspace(0., 1., 6)
    obs_freq = np.full((len(init_months), lead, ncategs, len(bins)-1), np.nan)
    prob_avg = np.full((len(init_months), lead, ncategs, len(bins)-1), np.nan)
    pred_marginal = np.full((len(init_months), lead, ncategs, len(bins)-1), np.nan)
    rel = np.full((len(init_months), lead, ncategs), np.nan)
    res = np.full((len(init_months), lead, ncategs), np.nan)
    pc_coefs = np.full((2, len(init_months), lead, npcs), np.nan)
    pc_loadings = np.full((2, len(init_months), lead, npcs, len(lats), len(lons)), np.nan)
    vs_wgts = np.full((len(init_months), lead, nfeats), np.nan)
    
    for n, i in enumerate(init_months):
        idcs = np.where((Y['val']['time.month'].values == i))[0]
        Yval_i = Y['val'].sel(lat=lats, lon=lons)[idcs]
        time_axis = np.concatenate([pd.date_range(j, periods=lead, freq='MS') for j in Yval_i.time.values])
        Yval_i = Yval_i.stack(time_stacked=('time', 'time_seq'), create_index=False).drop(('time', 'time_seq')).squeeze().rename(time_stacked='time').assign_coords({'time': time_axis}).transpose('time', 'lat', 'lon')
        Yval_i = xr.where(Yval_i==-999., np.nan, Yval_i)

        vs_wgts[n] = vswgts[idcs].mean(0)

        Ypred_i = Ypred[idcs].reshape(-1, nmembs, len(lats), len(lons))
        Yq33 = Y.q33.sel(time=time_axis, lat=lats, lon=lons).values
        Yq66 = Y.q66.sel(time=time_axis, lat=lats, lon=lons).values
        Ymn = Y.mn.sel(time=time_axis, lat=lats, lon=lons).values
        Ystd = Y.std.sel(time=time_axis, lat=lats, lon=lons).values
        
        Yval_i_categ, _, _ = scalar2categ(Yval_i.values, ncategs, 'one-hot', Yq33, Yq66)
        Ypred_i_probs, _, _ = scalar2categ(Ypred_i, ncategs, 'one-hot', Yq33, Yq66, count=True)
        Yval_i_total = compute_anomalies(Yval_i.values, Ymn, Ystd, reverse=True)
        Ypred_i_total = compute_anomalies(Ypred_i, np.tile(Ymn[:, None], (1, nmembs, 1, 1)), np.tile(Ystd[:, None], (1, nmembs, 1, 1)), reverse=True)
        for l in range(lead):
            rmse_i_l = EvalMetrics.RMSE(Yval_i.values[l::lead], Ypred_i.mean(1)[l::lead])  # H, W
            RMSE[n, l] = rmse_i_l
            # PCA applit to observations
            pca = PCA(npcs)
            nan_mask = np.isnan(Yval_i.values[l::lead])
            pca_data_in = np.where(nan_mask, -999, Yval_i.values[l::lead]).reshape(-1, Yval_i.values.shape[-2]*Yval_i.values.shape[-1])
            pca.fit(pca_data_in)
            pc_coefs[0, n, l] = pca.explained_variance_ratio_
            pc_loadings[0, n, l] = pca.components_.reshape(2, Yval_i.values.shape[-2], Yval_i.values.shape[-1])
            # PCA applit to TelNet predictions
            pca = PCA(npcs)
            nan_mask = np.isnan(Ypred_i.mean(1)[l::lead])
            pca_data_in = np.where(nan_mask, -999, Ypred_i.mean(1)[l::lead]).reshape(-1, Ypred_i.shape[-2]*Ypred_i.shape[-1])
            pca.fit(pca_data_in)
            pc_coefs[1, n, l] = pca.explained_variance_ratio_
            pc_loadings[1, n, l] = pca.components_.reshape(2, Ypred_i.mean(1).shape[-2], Ypred_i.shape[-1])
            rps_i_l = EvalMetrics.RPS(Yval_i_categ[l::lead].transpose(1, 0, 2, 3), Ypred_i_probs[l::lead].transpose(1, 0, 2, 3))  # H, W
            RPS[n, l] = rps_i_l
            ssr_i_l = EvalMetrics.SSR(Yval_i_total[l::lead], Ypred_i_total[l::lead])
            SSR[n, l] = ssr_i_l
            ranks_i_l = EvalMetrics.obs_rank(Yval_i.values[l::lead], Ypred_i[l::lead])
            ranks[n, l] = ranks_i_l
            ranks_ext_i_l = EvalMetrics.obs_rank(Yval_i.values[l::lead], Ypred_i[l::lead], 1)
            ranks_ext[n, l] = ranks_ext_i_l
            obs_freq_i_l, prob_avg_i_l, pred_marginal_i_l, rel_i_l, res_i_l = EvalMetrics.calibration_refinement_functions(Yval_i_categ[l::lead].transpose(1, 0, 2, 3), 
                                                                                                                           Ypred_i_probs[l::lead].transpose(1, 0, 2, 3), 
                                                                                                                           bins)
            obs_freq[n, l] = obs_freq_i_l
            prob_avg[n, l] = prob_avg_i_l
            pred_marginal[n, l] = pred_marginal_i_l
            rel[n, l] = rel_i_l
            res[n, l] = res_i_l

    return RMSE, RPS, SSR, ranks, ranks_ext, obs_freq, prob_avg, pred_marginal, rel, res, pc_coefs, pc_loadings, vs_wgts

def split_validation(
        X: Dict[str, DataManager], 
        Y: DataManager, 
        train_samples, 
        val_samples,
        model_predictors,
        model_config: list,
        DEVICE: str
    ):
    
    nfeats = model_config[-4]
    time_steps = model_config[-3]
    lead = model_config[-2]
    
    Xdm, Ydm = split_dataset(X, Y, train_samples, val_samples, model_predictors, nfeats, time_steps, lead)

    telnet = training(Xdm, Ydm, model_config, DEVICE)
    Ypred, Wgts = inference(Xdm, Ydm, telnet, DEVICE)
    RMSE, RPS, SSR, ranks, ranks_ext, obs_freq, prob_avg, pred_marginal, rel, res, pc_coefs, pc_loadings, vs_wgts = evaluate(Ydm, Ypred, Wgts)

    del telnet
    torch.cuda.empty_cache()

    return Ydm, RMSE, RPS, SSR, ranks, ranks_ext, obs_freq, prob_avg, pred_marginal, rel, res, pc_coefs, pc_loadings, vs_wgts

def main(arguments):

    """
    seed_n: int, 
    X: Dict[str, DataManager], 
    Y: DataManager, 
    search_arr: np.ndarray,
    seeds: np.ndarray, 
    """

    seed_n, X, Y, model_predictors, search_arr, seeds, device = arguments
    
    seed_pos = np.argwhere(seeds == seed_n).flatten()[0]
    set_seed(seed_n)
    if device is None:
        device = DEVICE
    nconfigs = len(search_arr)
    init_months = [1, 4, 7, 10]

    checkpoints_dir = f'{exp_data_dir}/checkpoints_sel'
    metric_names = ['RMSE', 'RPS', 'SSR', 'ranks', 'ranks_ext', 'obs_freq', 'prob_avg', 'pred_marginal', 'rel', 'res', 'PCA_coefs', 'PCA_loadings']
    metrics = [None]*len(metric_names)
    for i, metric in enumerate(metric_names):
        metrics[i], n = identify_checkpoint(metric, seed_n, checkpoints_dir)
    if all([True if metric is not None else False for metric in metrics]) and n == nconfigs:
        print (f'All metrics already computed for seed {seed_n}')
        return

    print ('Sampling years ...')
    test_yrs = np.arange(2003, 2023)
    train_yrs, val_yrs, test_yrs = split_sample(np.arange(1942, 2002), test_yrs)
    print (f'Preprocessing data n={seed_pos} ...')
    X, Y = preprocess_data(X, Y, train_yrs)
    
    for config in search_arr[n:]:
        leads = config[-2]
        Ydm, rmse, rps, ssr, ranks, ranks_ext, obs_freq, prob_avg, pred_marginal, rel, res, pc_coefs, pc_loadings, wgts = split_validation(deepcopy(X), deepcopy(Y), train_yrs, val_yrs, model_predictors, config, device) 
        if n == 0:
            for i, metric in enumerate([rmse, rps, ssr, ranks, ranks_ext, obs_freq, prob_avg, pred_marginal, rel, res, pc_coefs, pc_loadings]):
                metrics[i] = np.full((nconfigs, *metric.shape), np.nan)
        metrics[0][n] = rmse
        metrics[1][n] = rps
        metrics[2][n] = ssr
        metrics[3][n] = ranks
        metrics[4][n] = ranks_ext
        metrics[5][n] = obs_freq
        metrics[6][n] = prob_avg
        metrics[7][n] = pred_marginal
        metrics[8][n] = rel
        metrics[9][n] = res
        metrics[10][n] = pc_coefs
        metrics[11][n] = pc_loadings
        for i, metric in enumerate(metric_names):
            update_checkpoint(metrics[i], metric, n, seed_n, checkpoints_dir)
        n += 1
        nfeats = config[-4]
        if not os.path.exists(os.path.join(f'{checkpoints_dir}/varsel_{seed_n:04d}_{n:04d}.nc')):
            save_varsel_wgts(wgts, init_months, leads, model_predictors, nfeats, checkpoints_dir, f'varsel_{seed_n:04d}_{n:04d}')

if __name__ == '__main__':

    main()
