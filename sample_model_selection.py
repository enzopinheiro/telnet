import argparse
from collections import Counter
import os
import shutil
import torch
import xarray as xr
import numpy as np
from copy import deepcopy
from itertools import repeat
from random import sample
from datetime import datetime
from multiprocessing import Pool
from model_selection import main as selection_main
from utils import exp_data_dir, exp_results_dir, make_dir, read_obs_data, get_search_matrix
from utilities.plots import plot_boxplot, plot_error_maps, plot_mn_varsel_wgts, plot_model_freq, plot_feat_freq, plot_pca_maps, plot_prob_diags, plot_rank_histogram


def main(nsamples, iconfig=0, fconfig=-1, which_gpu=None):
    torch.multiprocessing.set_start_method('spawn')
    init_time = datetime.now()
    checkpoints_dir = f'{exp_data_dir}/checkpoints_sel'
    make_dir(checkpoints_dir)
    result_dir = f'{exp_results_dir}/selection/'
    make_dir(result_dir)
    print ('Reading data ...')
    X, Y, idcs_list = read_obs_data()
    model_predictors = np.loadtxt(os.path.join(f'{exp_data_dir}/models/', f'final_feats.txt'), dtype=str, delimiter=' ')
    search_arr, search_df = get_search_matrix()
    init_months = [1, 4, 7, 10]
    if os.path.exists(os.path.join(exp_data_dir, 'seeds_sel.txt')):
        # Read the existing seeds file
        seeds = np.loadtxt(os.path.join(exp_data_dir, 'seeds_sel.txt'), delimiter=',', dtype=int)
    else:
        seeds = np.arange(nsamples)
        # Save seeds as a text file
        np.savetxt(os.path.join(exp_data_dir, 'seeds_sel.txt'), seeds, delimiter=',', fmt='%i')

    if fconfig == -1:
        fconfig = nsamples

    seed_slice = seeds[iconfig:fconfig]
    if which_gpu is not None:
        DEVICE = torch.device(f"cuda:{which_gpu}" if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(DEVICE)
    
    # nproc = 1  # Number of processes to run in parallel within the selected device. Not recommended to run in serial mode because it will take a long time
    # with Pool(nproc) as p:
    #     p.map(selection_main,
    #                 zip(seed_slice,
    #                     repeat(deepcopy(X)),
    #                     repeat(deepcopy(Y)),
    #                     repeat(deepcopy(model_predictors)),
    #                     repeat(search_arr),
    #                     repeat(seeds),
    #                     repeat(DEVICE)
    #                     ),
    #         chunksize=1
    #     )
    for seed in seed_slice:
        selection_main((seed, deepcopy(X), deepcopy(Y), deepcopy(model_predictors), search_arr, seeds, DEVICE))
    
    if len([i for i in os.listdir(checkpoints_dir) if i.startswith('RPS_') and i.endswith('.npy')]) == len(seeds):

        lats = Y['var'].lat.values
        lons = Y['var'].lon.values
        x, y = np.meshgrid(lons, lats)
        coords = [(x, y)]
        # Plot validation results for each of the top_confgs
        avg_rps = []
        avg_ssr = []
        for i in range(len(search_df)):
            varsel_dict = {1:[], 4:[], 7:[], 10:[]}
            
            result_dir_config = f'{exp_results_dir}/selection/config_{i:03d}'
            make_dir(result_dir_config)
            # RMSE = [np.load(os.path.join(checkpoints_dir, j)) for j in os.listdir(checkpoints_dir) if j.startswith('RMSE') and j.endswith(f'{i:04d}_{ntelnet:04d}.npy')][0]
            RMSE = [np.load(os.path.join(checkpoints_dir, f'RMSE_{seed:04d}_{len(search_df)-1:04d}.npy')) for seed in seeds]
            RMSE = np.stack(RMSE, axis=0)[:, i]  # (nsamples, nconfig, ninit, nlead, nlat, nlon)
            RPS = [np.load(os.path.join(checkpoints_dir, f'RPS_{seed:04d}_{len(search_df)-1:04d}.npy')) for seed in seeds]
            RPS = np.stack(RPS, axis=0)[:, i]  # (nsamples, nconfig, ninit, nlead, nlat, nlon)
            SSR = [np.load(os.path.join(checkpoints_dir, f'SSR_{seed:04d}_{len(search_df)-1:04d}.npy')) for seed in seeds]
            SSR = np.stack(SSR, axis=0)[:, i]  # (nsamples, nconfig, ninit, nlead, nlat, nlon)
            ranks = [np.load(os.path.join(checkpoints_dir, f'ranks_{seed:04d}_{len(search_df)-1:04d}.npy')) for seed in seeds]
            ranks = np.stack(ranks, axis=0)[:, i]  # (nsamples, nconfig, ninit, nlead, nvalidpoints*nyrs)
            ranks_ext = [np.load(os.path.join(checkpoints_dir, f'ranks_ext_{seed:04d}_{len(search_df)-1:04d}.npy')) for seed in seeds]
            ranks_ext = np.stack(ranks_ext, axis=0)[:, i]  # (nsamples, nconfig, ninit, nlead, nvalidpoints*nyrs)
            obs_freq = [np.load(os.path.join(checkpoints_dir, f'obs_freq_{seed:04d}_{len(search_df)-1:04d}.npy')) for seed in seeds]
            obs_freq = np.stack(obs_freq, axis=0)[:, i]  # (nsamples, nconfig, ninit, nlead, ncategs, nbins)
            prob_avg = [np.load(os.path.join(checkpoints_dir, f'prob_avg_{seed:04d}_{len(search_df)-1:04d}.npy')) for seed in seeds]
            prob_avg = np.stack(prob_avg, axis=0)[:, i]  # (nsamples, nconfig, ninit, nlead, ncategs, nbins)
            pred_marginal = [np.load(os.path.join(checkpoints_dir, f'pred_marginal_{seed:04d}_{len(search_df)-1:04d}.npy')) for seed in seeds]
            pred_marginal = np.stack(pred_marginal, axis=0)[:, i]  # (nsamples, nconfig, ninit, nlead, ncategs, nbins)
            rel = [np.load(os.path.join(checkpoints_dir, f'rel_{seed:04d}_{len(search_df)-1:04d}.npy')) for seed in seeds]
            rel = np.stack(rel, axis=0)[:, i]  # (nsamples, nconfig, ninit, nlead, ncategs)
            res = [np.load(os.path.join (checkpoints_dir, f'res_{seed:04d}_{len(search_df)-1:04d}.npy')) for seed in seeds]
            res = np.stack(res, axis=0)[:, i]  # (nsamples, nconfig, ninit, nlead, ncategs)
            pc_coefs = [np.load(os.path.join(checkpoints_dir, f'PCA_coefs_{seed:04d}_{len(search_df)-1:04d}.npy')) for seed in seeds]
            pc_coefs = np.stack(pc_coefs, axis=0)[:, i]  # (nsamples, nconfig, npcs)
            pc_loadings = [np.load(os.path.join(checkpoints_dir, f'PCA_loadings_{seed:04d}_{len(search_df)-1:04d}.npy')) for seed in seeds]
            pc_loadings = np.stack(pc_loadings, axis=0)[:, i]  # (nsamples, nconfig, npcs, nlat, nlon)
            
            vs_wgts = [xr.open_dataset(f'{checkpoints_dir}/varsel_{seed:04d}_{(i+1):04d}.nc')['VSweights'] for seed in seeds]
            vs_wgts = xr.concat(vs_wgts, dim='samples')  # (nsamples, ninit, lead, nfeats)
            for k in init_months:
                varsel_dict[k] = vs_wgts.sel(init_month=k).values[:, 3:]  # dict values have shape (nsamples, nleads, nfeats)
            avg_rps.append(np.nanmean(RPS, axis=np.s_[1, 2, 3, 4]))  # average RPS over all init months, leads, lats and lons
            avg_ssr.append(np.nanmean(SSR, axis=np.s_[1, 2, 3, 4]))  # average SSR over all init months, leads, lats and lons
            gs_rmse = {'plot_specs': [1, 1, 0.2, 0.2], 'subplot_specs': [[1, 1, None, None]], 'fig_size': (10, 10)}
            gs_rps = {'plot_specs': [1, 1, 0.2, 0.2], 'subplot_specs': [[1, 1, None, None]], 'fig_size': (10, 10)}
            gs_pca = {'plot_specs': [1, 1, 0.2, 0.2], 'subplot_specs': [[1, 2, None, None]], 'fig_size': (15, 8)}

            # Plots
            for ii, init in enumerate(init_months):
                for jj, lead in enumerate([3, 4, 5]):
                    RMSE_ = np.nanmean(RMSE[:, ii, jj], axis=0)[None]  # (nmodel, nlat, nlon)
                    plot_error_maps(coords, RMSE_, ['TelNet'], f'{result_dir_config}/RMSE_{init}_{lead}', '', gs_rmse, 'RMSE', True, cbar_orientation='vertical')
                    RPS_ = np.nanmean(RPS[:, ii, jj], axis=0)[None]  # (nmodel, nlat, nlon)
                    plot_error_maps(coords, RPS_, ['TelNet'], f'{result_dir_config}/RPS_{init}_{lead}', '', gs_rps, 'RPS', True, cbar_orientation='vertical')
                    ranks_ = ranks[:, ii, jj][:, None]  # (nsample, nmodel, nvalidpoints*nyrs)
                    plot_rank_histogram(ranks_, ['TelNet'], f'{result_dir_config}/rank_histogram_{init}_{lead}', 2, 3, (15, 10))
                    ranks_ext_ = ranks_ext[:, ii, jj][:, None]  # (nsample, nmodel, nvalidpoints*nyrs)
                    plot_rank_histogram(ranks_ext_, ['TelNet'], f'{result_dir_config}/rank_histogram_ev_{init}_{lead}', 2, 3, (15, 10))
                    obs_freq_ = obs_freq[:, ii, jj][:, None]  # (nsample, nmodel, ncategs, nbins)
                    prob_avg_ = prob_avg[:, ii, jj][:, None]  # (nsample, nmodel, ncategs, nbins)
                    pred_marginal_ = pred_marginal[:, ii, jj][:, None]  # (nsample, nmodel, ncategs, nbins)
                    rel_ = rel[:, ii, jj][:, None]  # (nsample, nmodel, ncategs)
                    res_ = res[:, ii, jj][:, None]  # (nsample, nmodel, ncategs)
                    plot_prob_diags(prob_avg_, obs_freq_, pred_marginal_, rel_, res_, ['TelNet'], f'{result_dir_config}/prob_diags_{init}_{lead}', 2, 3, (22, 10))
                    pca_coefs_ = np.median(pc_coefs[:, :, ii, jj], 0)  # (nmodel+1, npcs)
                    pca_loadings_ = np.median(pc_loadings[:, :, ii, jj], 0)  # (nmodel+1, npcs, nlat, nlon)
                    plot_pca_maps(pca_loadings_, pca_coefs_, [coords[0]]+coords, ['Observation']+['TelNet'], result_dir_config, f'pca_{init}_{lead}', gs_pca, True, cbar_orientation='vertical')

            init_months_str = ['Feb', 'May', 'Aug', 'Nov']
            xlabels = {'Feb': ['MAM', 'AMJ', 'MJJ'],
                       'May': ['JJA', 'JAS', 'ASO'],
                       'Aug': ['SON', 'OND', 'NDJ'],
                       'Nov': ['DJF', 'JFM', 'FMA']}
            
            plot_mn_varsel_wgts(varsel_dict, vs_wgts.index.values, xlabels, init_months, init_months_str, 'varsel_weights_mn', f'{result_dir_config}')

        plot_boxplot(avg_rps, f'{exp_results_dir}/selection/', 'RPS')
        plot_boxplot(avg_ssr, f'{exp_results_dir}/selection/', 'SSR')

    print (f'Time elapsed: {datetime.now()-init_time}')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Model selection sampling')
    parser.add_argument('-n','--number', help='Sample size', required=True, default=1000)
    parser.add_argument('-i','--init_seed', help='Initial config', required=False, default=0)
    parser.add_argument('-f','--final_seed', help='Final config', required=False, default=-1)
    parser.add_argument('-gpu','--which_gpu', help='Run on a specific GPU', required=False, default=0)
    args = vars(parser.parse_args())
    n = int(args['number'])
    i = int(args['init_seed'])
    f = int(args['final_seed'])
    gpu = int(args['which_gpu'])
    main(n, i, f, gpu)
