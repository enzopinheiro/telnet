import argparse
from collections import Counter
import os
import shutil
import torch
import numpy as np
from copy import deepcopy
from itertools import repeat
from random import sample
from datetime import datetime
from multiprocessing import Pool
from model_selection import main as selection_main
from gen_init_weights import main as init_weights_main
from utils import exp_data_dir, exp_results_dir, make_dir, read_obs_data, get_search_matrix
from utilities.plots import plot_model_freq, plot_feat_freq


def main(nsamples, reproduce_paper=True):
    torch.multiprocessing.set_start_method('spawn')
    init_time = datetime.now()
    checkpoints_dir = f'{exp_data_dir}/checkpoints_sel'
    make_dir(checkpoints_dir)
    print ('Reading data ...')
    X, Y, idcs_list = read_obs_data()
    search_arr, search_df = get_search_matrix()
    if os.path.exists(os.path.join(exp_data_dir, 'seeds_sel.txt')):
        # Read the existing seeds file
        seeds = np.loadtxt(os.path.join(exp_data_dir, 'seeds_sel.txt'), delimiter=',', dtype=int)
    else:
        if reproduce_paper:
            seeds = np.arange(nsamples)
        else:
            seeds = sample(range(100000), nsamples)
        # Save seeds as a text file
        np.savetxt(os.path.join(exp_data_dir, 'seeds_sel.txt'), seeds, delimiter=',', fmt='%i')

    nproc = 4  # Number of processes to run in parallel. Not recommended to run in serial mode because it will take a long time
    with Pool(nproc) as p:
        out = p.map(selection_main,
                    zip(seeds,
                        repeat(deepcopy(X)),
                        repeat(deepcopy(Y)),
                        repeat(search_arr),
                        repeat(seeds)
                        )
                    )
    top_idcs = []
    feat_order = []
    for i in out:
        top_idcs.append(i[0])
        feat_order.append(i[1])
    
    top_idcs = np.array(top_idcs)  # (nsamples)
    feat_order = np.stack(feat_order, axis=0)  # (nsamples, nfeatures)
    print()
    # Plot frequency of top 1 models
    top_idx = plot_model_freq(top_idcs, exp_results_dir)
    # Plot features frequency conditioned on the most frequent top 1 model
    model_config = search_df.iloc[top_idx]
    print(f'Most frequent model {top_idx}:\n {model_config}')
    nfeats = int(model_config['nfeats'])
    feat_pos = np.where(top_idcs == top_idx)[0]
    feat_order = feat_order[feat_pos, :nfeats]
    plot_feat_freq(feat_order, exp_results_dir)
    feat_order_list = [tuple(np.sort(row)) for row in feat_order]
    most_common_feats = Counter(feat_order_list).most_common(1)[0][0]
    final_feats = [i for i in idcs_list if i in most_common_feats]
    model_config = [int(x) if x % 1 == 0 else float(x) for x in model_config.values]
    print (f'Most frequent features: {final_feats}')
    # Sample init weigthts and selects the best one based on fixed validation RPS
    if not os.path.exists(os.path.join(f'{exp_data_dir}/models/', 'telnet_init.pt')):
        checkpoints_dir = f'{exp_data_dir}/checkpoints_init_weights'
        make_dir(checkpoints_dir)
        with Pool(nproc) as p:
            RPS = p.map(init_weights_main,
                        zip(seeds,
                            repeat(deepcopy(X)),
                            repeat(deepcopy(Y)),
                            repeat(model_config),
                            repeat(seeds),
                            repeat(final_feats),
                            )
                        )
        RPS = np.concatenate(RPS, axis=0)  # nsamples, 4 init_months, 6 lead times
        RPS_mn = np.mean(RPS[:, :, 3:], np.s_[1, 2])
        min_idx = np.argmin(RPS_mn)
        for i in os.listdir(f'{exp_data_dir}/models/'):
            if i.endswith('.pt') and int(i[6:10]) != min_idx:
                os.remove(os.path.join(f'{exp_data_dir}/models/', i))
            else:
                os.rename(os.path.join(f'{exp_data_dir}/models/', i), os.path.join(f'{exp_data_dir}/models/', 'telnet_init.pt'))
        shutil.rmtree(checkpoints_dir)
    np.savetxt(os.path.join(f'{exp_data_dir}/models/', 'final_feats.txt'), final_feats, delimiter=',', fmt='%s')
    np.savetxt(os.path.join(f'{exp_data_dir}/models/', 'model_config.txt'), model_config, delimiter=',', fmt='%.5f')
    print (f'Time elapsed: {datetime.now()-init_time}')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Model selection sampling')
    parser.add_argument('-n','--number', help='Sample size', required=True, default=1000)
    args = vars(parser.parse_args())
    n = int(args['number'])
    main(n, reproduce_paper=True)
