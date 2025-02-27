import os
from math import ceil
import xarray as xr
import numpy as np
import pandas as pd
from typing import Dict, List
from random import sample
from copy import deepcopy
from sklearn.model_selection import train_test_split
from utils import DataManager, EvalMetrics
from utils import exp_data_dir, exp_results_dir, points_ce
from utils import make_dir, prepare_X_data, prepare_Y_data, read_era5_data, read_indices_data, compute_anomalies, scalar2categ
from utilities.plots import plot_error_maps, plot_heatmap, plot_obs_ypred_maps, plot_prob_diags, plot_rank_histogram, plot_extremes_rank_histogram, plot_mn_varsel_wgts


def read_obs_data():

    root_datadir = os.getenv('TELNET_DATADIR')
    if root_datadir is None:
        raise ValueError('Environment variable TELNET_DATADIR is not set.')

    idcs_list = ['oni', 'mei', 'atn-sst', 'ats-sst', 'atl-sst', 'iod', 'iobw', 'nao', 'pna', 'qbo', 'aao', 'ao']
    indices = read_indices_data('1941-01-01', '2023-12-01', root_datadir, idcs_list, '_1941-2023')
    pcp = read_era5_data('pr', root_datadir, points_ce)

    cov_date_s = ('1942-01-01', '2022-12-01')
    auto_date_s = ('1941-12-01', '2023-01-01')
    pred_date_s = ('1941-12-01', '2023-01-01')
    ce_bounds = ((-1.75, -8.5), (-42.75, -36.))  # ERA5
    
    X = {'auto': deepcopy(pcp['pr']), 'cov': deepcopy(indices)}
    Y = deepcopy(pcp['pr'])

    Xdm = prepare_X_data(X, auto_date_s, cov_date_s, 
                         cov_bounds=((None, None), (None, None)), 
                         auto_bounds=ce_bounds)

    Ydm = prepare_Y_data(Y, pred_date_s, region_bounds=ce_bounds)

    return Xdm, Ydm, idcs_list

def read_telnet_darray(fname):

    ypred_ds = xr.open_dataset(f'{exp_data_dir}/{fname}_ypred.nc')
    vs_ds = xr.open_dataset(f'{exp_data_dir}/{fname}_vsweights.nc')

    return ypred_ds, vs_ds

def read_num_models_data(test_yrs, init_month, Ylat, Ylon, Ymn_obs, Ystd_obs):

    months_str = {1: 'jan', 2: 'feb', 3: 'mar', 4: 'apr', 5: 'may', 6: 'jun', 
                  7: 'jul', 8: 'aug', 9: 'sep', 10: 'oct', 11: 'nov', 12: 'dec'}
    
    root_datadir = os.getenv('TELNET_DATADIR')
    if root_datadir is None:
        raise ValueError('Environment variable TELNET_DATADIR is not set.')
    dynmodel_dir = f'{root_datadir}/numerical_models_data/{months_str[init_month]}/'
    
    models = ['cola-rsmas-ccsm4', 'cancm4i', 'gem-nemo', 'gfdl-spear', 'seas5']
    models_totals = []
    for model in models:
        model_name = [i for i in os.listdir(dynmodel_dir) if i.startswith(f'pr_seasonal_{model}')][0]
        ds = xr.open_dataset(f'{dynmodel_dir}/{model_name}')
        pcp = ds['Ypred'].reindex(lat=ds.lat[::-1])
        pcp = compute_num_models_std_anom(pcp, test_yrs)
        pcp = flatten_lead_dim(pcp, init_month)
        pcp = pcp.interp(lat=Ylat, lon=Ylon, method='linear')
        pcp = compute_num_models_totals(pcp, Ymn_obs, Ystd_obs)
        models_totals.append(pcp)
    return models_totals, models

def flatten_lead_dim(darr, init_month):

    # +1 to center the seasonal window
    time_axis = [np.datetime64(f'{i}-{((init_month+ceil(j))%12)+1:02d}-01') 
                 for i in darr.time.values for j in darr.leads.values]

    darr = darr.stack({'time_flatten': ['time', 'leads']}).transpose('time_flatten', 'nmembs', 'lat', 'lon').drop(['time', 'leads'])
    darr = darr.rename({'time_flatten': 'time'})
    
    darr['time'] = time_axis
    
    return darr

def compute_num_models_std_anom(darr, test_yrs):

    yrs = darr.time.values
    clim_yrs = np.array([i for i in yrs if i not in test_yrs])
    mn = darr.sel(time=clim_yrs).mean('nmembs').mean('time')
    std = darr.sel(time=clim_yrs).mean('nmembs').std('time')
    Ypred_anom = ((darr - mn)/std)

    return Ypred_anom

def compute_num_models_totals(darr, mn_obs, std_obs):

    darr*=std_obs.sel(time=darr.time.values)
    darr+=mn_obs.sel(time=darr.time.values)
    return darr

def preprocess_data(
        Xdm: Dict[str, DataManager], 
        Ydm: DataManager, 
        train_yrs: np.ndarray, 
        val_yrs: np.ndarray, 
        feat_order: np.ndarray,
        nfeats: int,
        time_steps: int = 2,
        lead: int = 6
    ):
    
    stat_yrs = np.asarray([i for i in train_yrs if i>=1971 and i<=2020])
    Xdm['auto']['var_seas'] = deepcopy(Xdm['auto']['var'])
    Xdm['auto'].monthly2seasonal('var_seas', 'sum', True)
    Xdm['auto'].compute_statistics(stat_yrs, ['mean', 'std'], 'var_seas')
    Xdm['auto'].to_anomalies('var_seas', stdized=True)
    Xdm['auto'].add_seq_dim('var_seas', time_steps)
    Xdm['auto'].replace_nan('var_seas', -999.)

    Xdm['cov']['var'] = Xdm['cov']['var'].sel(indices=feat_order).isel(indices=slice(0, nfeats))
    Xdm['cov'].compute_statistics(stat_yrs, ['mean', 'std'])
    Xdm['cov'].to_anomalies('var', stdized=True)
    Xdm['cov'].add_seq_dim('var', time_steps)
    Xdm['cov'].replace_nan('var', -999.)
    
    Ydm['var_seas'] = deepcopy(Ydm['var'])
    Ydm.monthly2seasonal('var_seas', 'sum', True)
    Ydm.compute_statistics(stat_yrs, ['mean', 'std'], 'var_seas') 
    Ydm.to_anomalies('var_seas', stdized=True)
    Ydm.compute_statistics(stat_yrs, ['terciles'], 'var_seas') 
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

def read_sample_and_feature_files():

    if os.path.exists(f'{exp_data_dir}/test_years.txt'):
        test_yrs = np.sort(np.loadtxt(f'{exp_data_dir}/test_years.txt', dtype=int))
    else:
        test_yrs = None
    if os.path.exists(f'{exp_data_dir}/train_years.txt'):
        train_yrs = np.sort(np.loadtxt(f'{exp_data_dir}/train_years.txt', dtype=int))
    else:
        train_yrs = None
    if os.path.exists(f'{exp_data_dir}/val_years.txt'):
        val_yrs = np.sort(np.loadtxt(f'{exp_data_dir}/val_years.txt', dtype=int))
    else:
        val_yrs = None
    if os.path.exists(f'{exp_data_dir}/feat_order.txt'):
        feat_order = np.loadtxt(f'{exp_data_dir}/feat_order.txt', dtype=str, delimiter=',')
    else:
        feat_order = None

    return train_yrs, val_yrs, test_yrs, feat_order

def split_sample(samples, train_samples=None, val_samples=None, test_samples=None):

    if test_samples is None:
        test_samples = np.array(sample(range(1982, 2021), 20))
        np.savetxt(f'{exp_data_dir}/test_years.txt', test_samples, fmt='%i')
    if train_samples is None and val_samples is None:
        train_samples = np.array([i for i in samples if (i not in test_samples)])
        train_samples, val_samples = train_test_split(train_samples, test_size=0.20, random_state=4)
        train_samples = np.sort(train_samples)
        val_samples = np.sort(val_samples)
        np.savetxt(f'{exp_data_dir}/val_years.txt', val_samples, fmt='%i')
        np.savetxt(f'{exp_data_dir}/train_years.txt', train_samples, fmt='%i')
        
    return train_samples, val_samples, test_samples

def evaluate(Y: DataManager, 
             Ypred: np.ndarray,
             num_models: dict,
             varsel_wgts: np.ndarray,
             var_names: List[str],
             result_dir: str,
             plot_ypred: bool=False):
    
    init_months = [1, 4, 7, 10]  # time step 3 will the second non-overlapping season [MAM, JJA, SON, DJF]
    lead = 6
    nmembs = Ypred.shape[2]
    x, y = np.meshgrid(Y.lon, Y.lat)
    models_names = ['TelNet', 'CCSM4', 'CanCM4i', 'GEM-NEMO', 'GFDL', 'SEAS5']
    
    RMSE = np.full((len(num_models[11])+1, 4, 6), np.nan)
    RPS = np.full((len(num_models[11])+1, 4, 6), np.nan)

    for n, i in enumerate(init_months):
        Yval_categ_l3 = []
        Ypred_probs_l3 = []
        Yval_anom_l3 = []
        Ypred_anom_l3 = []
        RMSE_l3 = []
        RPS_l3 = []
        coords_i = []
        num_models_i = num_models[i+1]
        idcs = np.where((Y['val']['time.month'].values == i))[0]

        vsel_wgts_i = varsel_wgts[idcs].reshape(-1, varsel_wgts.shape[-1])

        Yval_i = Y['val'][idcs]
        Yval_i = xr.concat(
            [Yval_i.sel(time=j).drop('time').squeeze().assign_coords({'time_seq': pd.date_range(j, periods=lead, freq='MS')}).rename(time_seq='time') 
             for j in Yval_i.time.values], dim='time'
        )
        Yval_i = xr.where(Yval_i==-999., np.nan, Yval_i)
        years = Yval_i[0::lead]['time.year'].values
        Yval_i_time = Yval_i.time.values
        Ypred_i = Ypred[idcs].reshape(-1, Ypred.shape[-3], Ypred.shape[-2], Ypred.shape[-1])
        Yq33 = Y.q33.sel(time=Yval_i.time.values).values
        Yq66 = Y.q66.sel(time=Yval_i.time.values).values
        Ymn = Y.mn.sel(time=Yval_i.time.values).values
        Ystd = Y.std.sel(time=Yval_i.time.values).values

        # Yq33 = norm.ppf(1/3, loc=0, scale=1)
        # Yq66 = norm.ppf(2/3, loc=0, scale=1)
        
        Yval_i_categ, _, _ = scalar2categ(Yval_i.values, 3, 'one-hot', Yq33, Yq66)
        Ypred_i_probs, _, _ = scalar2categ(Ypred_i, 3, 'one-hot', Yq33, Yq66, count=True)
        Yval_i_total = compute_anomalies(Yval_i.values, Ymn, Ystd, reverse=True)
        Ypred_i_total = compute_anomalies(Ypred_i, np.tile(Ymn[:, None], (1, nmembs, 1, 1)), np.tile(Ystd[:, None], (1, nmembs, 1, 1)), reverse=True)
        
        for l in range(lead):
            rmse_i_l = EvalMetrics.RMSE(Yval_i.values[l::lead], Ypred_i.mean(1)[l::lead])
            RMSE[0, n, l] = np.nanmean(rmse_i_l)
            rps_i_l = EvalMetrics.RPS(Yval_i_categ[l::lead].transpose(1, 0, 2, 3), Ypred_i_probs[l::lead].transpose(1, 0, 2, 3))
            RPS[0, n, l] = np.nanmean(rps_i_l)
            if l == 3:
                Yval_anom_l3.append(Yval_i.values[l::lead])
                Ypred_anom_l3.append(Ypred_i[l::lead])
                Yval_categ_l3.append(Yval_i_categ[l::lead])
                Ypred_probs_l3.append(Ypred_i_probs[l::lead])
                RMSE_l3.append(rmse_i_l)
                RPS_l3.append(rps_i_l)
                coords_i.append((x, y))

        for m, model in enumerate(num_models_i):
            model_time = [i for i in model.time.values if i in Yval_i_time]
            Ypred_num_i = model.sel(time=model_time).values
            nmembs_num = Ypred_num_i.shape[1]
            Ymn = Y.mn.sel(time=model_time).values
            Ystd = Y.std.sel(time=model_time).values
            Yq33 = Y.q33.sel(time=model_time).values
            Yq66 = Y.q66.sel(time=model_time).values
            # Yq33 = norm.ppf(1/3, loc=0, scale=1)
            # Yq66 = norm.ppf(2/3, loc=0, scale=1)
            Ypred_num_anom_i = (Ypred_num_i - np.tile(Ymn[:, None], (1, nmembs_num, 1, 1)))/np.tile(Ystd[:, None], (1, nmembs_num, 1, 1))
            Ypred_num_i_probs, _, _ = scalar2categ(Ypred_num_anom_i, 3, 'one-hot', Yq33, Yq66, count=True)
            for l in range(3, lead):
                rmse_i_l = EvalMetrics.RMSE(Yval_i.values[l::lead], Ypred_num_anom_i.mean(1)[l-3::3])
                RMSE[m+1, n, l] = np.nanmean(rmse_i_l)
                rps_i_l = EvalMetrics.RPS(Yval_i_categ[l::lead].transpose(1, 0, 2, 3), Ypred_num_i_probs[l-3::3].transpose(1, 0, 2, 3))
                RPS[m+1, n, l] = np.nanmean(rps_i_l)
                if l == 3:
                    Ypred_anom_l3.append(Ypred_num_anom_i[l-3::3])
                    Ypred_probs_l3.append(Ypred_num_i_probs[l-3::3])
                    RMSE_l3.append(rmse_i_l)
                    RPS_l3.append(rps_i_l)
                    coords_i.append((x, y))
        if i == 10:
            title = 'Nov-DJF'
        elif i == 1:
            title = 'Feb-MAM'
        elif i == 4:
            title = 'May-JJA'
        elif i == 7:
            title = 'Aug-SON'
        plot_rank_histogram(Yval_anom_l3[0], Ypred_anom_l3, models_names, f'{result_dir}/rank_histogram_{title}.png', 2, 3, (15, 10))
        plot_extremes_rank_histogram(Yval_anom_l3[0], Ypred_anom_l3, models_names, f'{result_dir}/rank_histogram_ev_{title}.png', 2, 3, 1, (15, 10))
        plot_prob_diags(Yval_categ_l3[0], Ypred_probs_l3, models_names, np.linspace(0., 1., 11), f'{result_dir}/rel_diagram_{title}.png', 2, 3)
        plot_error_maps(coords_i, RMSE_l3, models_names, f'{result_dir}/RMSE_{title}.png', '', 2, 3, (12, 8), 'RMSE', True, cbar_orientation='vertical')
        plot_error_maps(coords_i, RPS_l3, models_names, f'{result_dir}/RPS_{title}.png', '', 2, 3, (12, 8), 'RPS', cbar_orientation='vertical')
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
                    
    RMSESS = np.concatenate([((1-(RMSE[0]/RMSE[i]))*100)[None] for i in range(1, RMSE.shape[0])], 0)
    RPSS = np.concatenate([((1-(RPS[0]/RPS[i]))*100)[None] for i in range(1, RPS.shape[0])], 0)
    plot_config_comparison(RMSESS[:, :, -3:], 'RMSESS', result_dir)
    plot_config_comparison(RPSS[:, :, -3:], 'RPSS', result_dir)
    plot_mn_varsel_wgts(Y, varsel_wgts, var_names, 'mn_varsel_wgts.png', result_dir)

    return RMSE, RPS

def plot_config_comparison(metric_arr: np.ndarray, metric_name: str, result_dir: str):
    
    init_months = ['Feb', 'May', 'Aug', 'Nov']
    ylabels = ['CCSM4', 'CanCM4i', 'GEM-NEMO', 'GFDL', 'SEAS5']
    yticks = np.arange(0.5, len(ylabels)+0.5)
    xlabels = {'Feb': ['MAM', 'AMJ', 'MJJ'],
               'May': ['JJA', 'JAS', 'ASO'],
               'Aug': ['SON', 'OND', 'NDJ'],
               'Nov': ['DJF', 'JFM', 'FMA']}
    if metric_name == 'RMSE':
        lvls = np.arange(0., 350., 50.)
        cmap = 'viridis'
        extend = 'max'
    elif metric_name == 'RPS':
        lvls = np.arange(0., 1.1, 0.1)
        cmap = 'viridis'
        extend = 'max'
    else:
        lvls = np.arange(-30, 35, 5)
        cmap = 'RdBu'
        extend= 'both'
    plot_heatmap(metric_arr, ylabels, yticks, xlabels, lvls, init_months, metric_name, result_dir, extend=extend, cmap=cmap, cbar_orientation='vertical')

def main(model_n):
    print ("Evaluating test set")
    X, Y, idcs_list = read_obs_data()
    train_yrs, val_yrs, test_yrs, feat_order = read_sample_and_feature_files()
    train_yrs, val_yrs, test_yrs = split_sample(np.arange(1942, 2023), train_yrs, val_yrs, test_yrs)
    model_config = pd.read_csv(f'{exp_data_dir}/search_matrix.csv', index_col=0).loc[model_n]
    model_config = [int(x) if x % 1 == 0 else float(x) for x in model_config.values]
    init_months = [1, 4, 7, 10]
    nfeats = model_config[-4]
    time_steps = model_config[-3]
    lead = model_config[-2]
    result_dir = f'{exp_results_dir}/test'
    make_dir(result_dir)
    Xdm, Ydm = preprocess_data(X, Y, train_yrs, test_yrs, feat_order, nfeats, time_steps, lead)
    dyn_models = {}
    for i in init_months:
        dyn_models[i+1], models_name = read_num_models_data(test_yrs, i+1, Y.lat, Y.lon, Ydm['mn'], Ydm['std'])
    Ypred_ds, Wgts_ds = read_telnet_darray('telnet_v1')
    Ypred = Ypred_ds['Ypred'].values
    Wgts = Wgts_ds['VSweights'].values
    RMSE, RPS = evaluate(Ydm, Ypred, dyn_models, Wgts, ['ylag']+list(feat_order[:nfeats]), result_dir, True)
    print (f"Test set results are saved at {result_dir}")

if __name__ == '__main__':

    main()
