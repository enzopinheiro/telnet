import os
import torch
import numpy as np
import argparse
from copy import deepcopy
from datetime import datetime
from itertools import repeat
from multiprocessing import Pool
from random import sample
from utils import exp_data_dir, exp_results_dir, make_dir, read_obs_data, read_num_models_data, read_dl_models_data, get_search_matrix
from test import main as test_main
from utilities.plots import plot_error_maps, plot_mn_varsel_wgts, plot_rank_histogram, plot_prob_diags, scorebars


def main(nsamples, reproduce_paper=True):
    torch.multiprocessing.set_start_method('spawn')
    init_time = datetime.now()
    checkpoints_dir = f'{exp_data_dir}/checkpoints_model'
    make_dir(checkpoints_dir)
    result_dir = f'{exp_results_dir}/test'
    make_dir(result_dir)
    print ('Reading data ...')
    X, Y, idcs_list = read_obs_data()
    search_arr, search_df = get_search_matrix()
    
    init_months = [1, 4, 7, 10]
    baseline_models = {}
    for i in init_months:
        dyn_models, _ = read_num_models_data(i+1, Y.lat, Y.lon)
        dl_models, _ = read_dl_models_data(i+1, Y.lat, Y.lon)
        baseline_models[i+1] = {'dyn': dyn_models, 'dl': dl_models}

    models_names = ['TelNet', 'CCSM4', 'CanCM4i', 'GEM-NEMO', 'GFDL', 'SEAS5', 'ClimaX']
    x, y = np.meshgrid(Y.lon, Y.lat)
    coords = [(x, y) for i in range(len(models_names))]

    model_predictors = np.loadtxt(os.path.join(f'{exp_data_dir}/models/', 'final_feats.txt'), dtype=str)
    model_config = np.loadtxt(os.path.join(f'{exp_data_dir}/models/', 'model_config.txt'), dtype=float)
    model_config = [int(x) if x % 1 == 0 else float(x) for x in model_config]
    X['cov']['var'] = X['cov']['var'].sel(indices=model_predictors)

    if os.path.exists(os.path.join(exp_data_dir, 'seeds_model.txt')):
        # Read the existing seeds file
        seeds = np.loadtxt(os.path.join(exp_data_dir, 'seeds_model.txt'), delimiter=',', dtype=int)
    else:
        if reproduce_paper:
            seeds = np.arange(nsamples)
        else:
            seeds = sample(range(100000), nsamples)
        # Save seeds as a text file
        np.savetxt(os.path.join(exp_data_dir, 'seeds_model.txt'), seeds, delimiter=',', fmt='%i')   
    
    Yval_RPS, RMSE, RPS, ranks, ranks_ext, obs_freq, prob_avg, pred_marginal, rel, res, RMSESS, RPSS = [], [], [], [], [], [], [], [], [], [], [], []
    varsel_dict = {1:[], 4:[], 7:[], 10:[]}
    for seed in seeds:
        i = test_main((seed, deepcopy(X), deepcopy(Y), deepcopy(model_config), deepcopy(model_predictors), seeds, deepcopy(baseline_models), init_months))
        Yval_RPS.append(i[0])
        RMSE.append(i[1][None, :, :, 3:])
        RPS.append(i[2][None, :, :, 3:])
        ranks.append(i[3][None, :, :, 3:])
        ranks_ext.append(i[4][None, :, :, 3:])
        obs_freq.append(i[5][None, :, :, 3:])
        prob_avg.append(i[6][None, :, :, 3:])
        pred_marginal.append(i[7][None, :, :, 3:])
        rel.append(i[8][None, :, :, 3:])
        res.append(i[9][None, :, :, 3:])
        RMSESS.append(i[10][None, :, :, 3:])
        RPSS.append(i[11][None, :, :, 3:])
        for j in init_months:
            varsel_dict[j].append(i[12].sel(time=i[12]['time.month']==j).values[:, 3:].mean(0)[None])

    RMSE = np.concatenate(RMSE, axis=0)  # (nsample, nmodel, ninit, nlead, nlat, nlon)
    RPS = np.concatenate(RPS, axis=0)  # (nsample, nmodel-1, ninit, nlead, nlat, nlon)
    ranks = np.concatenate(ranks, axis=0)  # (nsample, nmodel-1, ninit, nlead, nvalidpoints*nyrs)
    ranks_ext = np.concatenate(ranks_ext, axis=0)  # (nsample, nmodel-1, ninit, nlead, nvalidpoints*nyrs)
    obs_freq = np.concatenate(obs_freq, axis=0)  # (nsample, nmodel-1, ninit, nlead, ncategs, nbins)
    prob_avg = np.concatenate(prob_avg, axis=0) # (nsample, nmodel-1, ninit, nlead,, ncategs, nbins)
    pred_marginal = np.concatenate(pred_marginal, axis=0)  # (nsample, nmodel-1, ninit, nlead,, ncategs, nbins)
    rel = np.concatenate(rel, axis=0)  # (nsample, nmodel-1, ninit, nlead, ncategs)
    res = np.concatenate(res, axis=0)  # (nsample, nmodel-1, ninit, nlead, ncategs)
    RMSESS = np.concatenate(RMSESS, axis=0)  # (nsample, nmodel-1, ninits, nleads)
    RPSS = np.concatenate(RPSS, axis=0)  # (nsample, nmodel-2, ninits, nleads)
    varsel_dict = {j:np.concatenate(varsel_dict[j], axis=0) for j in init_months}  # dict values have shape (nsamples, nleads, nfeats)
    
    gs_rmse = {'plot_specs': [2, 1, 0.2, 0.1], 'subplot_specs': [[1, 3, None, None], [1, 4, None, None]], 'fig_size': (12, 8)}
    gs_rps = {'plot_specs': [2, 1, 0.2, 0.1], 'subplot_specs': [[1, 3, None, None], [1, 3, None, None]], 'fig_size': (12, 8)}

    # Plots
    for ii, i in enumerate(init_months):
        for jj, j in enumerate([3, 4, 5]):
            RMSE_ = np.nanmean(RMSE[:, :, ii, jj], axis=0)  # (nmodel, nlat, nlon)
            plot_error_maps(coords, RMSE_, models_names, f'{result_dir}/RMSE_{i}_{j}', '', gs_rmse, 'RMSE', True, cbar_orientation='vertical')
            RPS_ = np.nanmean(RPS[:, :, ii, jj], axis=0)  # (nmodel-1, nlat, nlon)
            plot_error_maps(coords, RPS_, models_names[0:-1], f'{result_dir}/RPS_{i}_{j}', '', gs_rps, 'RPS', True, cbar_orientation='vertical')
            ranks_ = ranks[:, :, ii, jj]  # (nsample, nmodel-1, nvalidpoints*nyrs)
            plot_rank_histogram(ranks_, models_names[0:-1], f'{result_dir}/rank_histogram_{i}_{j}', 2, 3, (15, 10))
            ranks_ext_ = ranks_ext[:, :, ii, jj]  # (nsample, nmodel-1, nvalidpoints*nyrs)
            plot_rank_histogram(ranks_ext_, models_names[0:-1], f'{result_dir}/rank_histogram_ev_{i}_{j}', 2, 3, (15, 10))
            obs_freq_ = obs_freq[:, :, ii, jj]  # (nsample, nmodel-1, ncategs, nbins)
            prob_avg_ = prob_avg[:, :, ii, jj]  # (nsample, nmodel-1, ncategs, nbins)
            pred_marginal_ = pred_marginal[:, :, ii, jj]  # (nsample, nmodel-1, ncategs, nbins)
            rel_ = rel[:, :, ii, jj]  # (nsample, nmodel-1, ncategs)
            res_ = res[:, :, ii, jj]  # (nsample, nmodel-1, ncategs)
            plot_prob_diags(prob_avg_, obs_freq_, pred_marginal_, rel_, res_, models_names[0:-1], f'{result_dir}/prob_diags_{i}_{j}', 2, 3, (22, 10))

    init_months_str = ['Feb', 'May', 'Aug', 'Nov']
    xlabels = {'Feb': ['MAM', 'AMJ', 'MJJ'],
               'May': ['JJA', 'JAS', 'ASO'],
               'Aug': ['SON', 'OND', 'NDJ'],
               'Nov': ['DJF', 'JFM', 'FMA']}
    scorebars(RMSESS, xlabels, models_names[1:], init_months_str, 'RMSESS', f'{result_dir}')
    scorebars(RPSS, xlabels, models_names[1:-1], init_months_str, 'RPSS', f'{result_dir}')
    plot_mn_varsel_wgts(varsel_dict, np.append(['ypred'], model_predictors), xlabels, init_months, init_months_str, 'varsel_weights_mn', f'{result_dir}')

    print (f'Time elapsed: {datetime.now()-init_time}')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Model test sampling')
    parser.add_argument('-n','--number', help='Sample size', required=True, default=1000)
    args = vars(parser.parse_args())
    n = int(args['number'])
    main(n, reproduce_paper=True)
