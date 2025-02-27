import os
import matplotlib
matplotlib.rcParams['backend'] = 'Agg'
import numpy as np
import cartopy.crs as ccrs
from typing import List
from collections import Counter
from matplotlib import pyplot as plt
from matplotlib import colors as clr
from matplotlib.colors import BoundaryNorm
from .plotting_utils import MakePlot
from utils import shp_ce_feature, points_ce

def plot_model_freq(top_idcs, savedir):  

    nsamples = top_idcs.shape[0]
    model_freq = Counter(top_idcs)
    model_freq = {k: v for k,v in model_freq.items()}
    model_freq = dict(sorted(model_freq.items()))
    top_model = max(model_freq, key=model_freq.get)
    plot = MakePlot((20, 5))
    plot.create_fig_instance(1, 1)
    plot.create_subplot_instance(1, 1, 0)
    plot.create_axis_instance(0, 1, 0, 1)
    plt.bar(model_freq.keys(), model_freq.values())
    plt.xlabel('Model ID')
    plt.ylabel('Frequency')
    plot.ax.text(top_model, model_freq[top_model], f'{top_model}', fontdict={'size':8})
    plt.savefig(os.path.join(savedir, 'model_freq.png'), dpi=300, bbox_inches='tight')
    plt.close()

    return top_model

def plot_feat_freq(feat_order, savedir):
    
    nsamples = feat_order.shape[0]
    nfeats = feat_order.shape[1]
    plot = MakePlot((20, 5))
    plot.create_fig_instance(1, nfeats)
    for i in range(nfeats):
        plot.create_subplot_instance(1, 1, i)
        plot.create_axis_instance(0, 1, 0, 1)
        feat_freq = Counter(feat_order[:, i])
        feat_freq = {k: v/nsamples for k,v in feat_freq.items()}
        plt.bar(feat_freq.keys(), feat_freq.values())
        plt.xlabel('Feature ID')
        plt.ylabel('Frequency')
    
    plt.savefig(os.path.join(savedir, 'feat_freq.png'), dpi=300, bbox_inches='tight')
    plt.close()

def reliability_diag(bins, calibration_dist, refinement_dist, rel, res, plt_obj, ngrid, title, fsize=10):

    colors = ['red', 'green', 'blue'][::-1]
    
    plt_obj.create_subplot_instance(3, 3, ngrid, 0.25, 0.2)
    plt_obj.create_axis_instance(0, 3, 0, 2)
    ax1 = plt_obj.ax
    nbins = bins.shape[1]
    bins1 = np.linspace(0, 1, nbins+1)
    for i in range(3):
        calibration_dist_med = np.nanmedian(calibration_dist[i], -1)
        upper_bar = np.nanpercentile(calibration_dist[i], 95., -1) - calibration_dist_med
        lower_bar = calibration_dist_med - np.nanpercentile(calibration_dist[i], 5., -1)
        ax1.errorbar(np.nanmedian(bins[i], -1), calibration_dist_med, yerr=np.concatenate((lower_bar[np.newaxis], upper_bar[np.newaxis]), 0), c=colors[i], capsize=4.0)
        ax1.plot(bins1, bins1, 'k--')
        # if ngrid in np.arange(3, 6):
        if ngrid in [3, 4, 5]:
            ax1.set_xlabel('Forecast probability')
        if ngrid in [0, 3]:
            ax1.set_ylabel('Observed frequency')
        ax1.set_xlim((0, 1))
        ax1.set_title(title)
        
        plt_obj.create_axis_instance(i, i+1, 2, 3)
        ax2 = plt_obj.ax
        density_med = np.nanmedian(refinement_dist[i], -1)
        density_lower = np.nanpercentile(refinement_dist[i], 5., -1)
        density_upper = np.nanpercentile(refinement_dist[i], 95., -1)
        ax2.bar((bins1+((bins1[1]-bins1[0])/2))[0:-1], density_med, color=colors[i], width=0.1,
                yerr=[density_med-density_lower, density_upper-density_med], ecolor='black', capsize=4)
        ax2.set_ylim((0, 0.6))
        xticks = np.round(bins1+((bins1[1]-bins1[0])/2), 2)[0:-1][::2]
        ax2.set_xticks(xticks)
        ax2.set_xticklabels(xticks, rotation=45)
        if i == 2 and ngrid in [3, 4, 5]:
            ax2.set_xlabel('Forecast bin')
            ax2.set_xticks(xticks)
            ax2.set_xticklabels(xticks, rotation=45)
        elif i == 2:
            ax2.set_xticks(xticks)
            ax2.set_xticklabels(xticks, rotation=45)
        else:
            ax2.set_xticks(xticks)
            ax2.set_xticklabels([], rotation=45)

def plot_prob_diags(prob_avg, obs_freq, pred_marginal, rel, res, models_name, figname, nrows, ncols, fsize=(22, 10)):

    plt_obj = MakePlot(fsize)
    plt_obj.create_fig_instance(nrows, ncols)

    prob_avg = prob_avg.transpose((1, 2, 3, 0))[:, ::-1]  # [nmodel, ncateg, nbin, nsamples]  # AN is the first categ
    obs_freq = obs_freq.transpose((1, 2, 3, 0))[:, ::-1]  # [nmodel, ncateg, nbin, nsamples]  # AN is the first categ
    pred_marginal = pred_marginal.transpose((1, 2, 3, 0))[:, ::-1]  # [nmodel, ncateg, nbin, nsamples]  # AN is the first categ
    rel = rel.transpose((1, 2, 0))[:, ::-1]  # [nmodel, ncateg, nsamples]  # AN is the first categ
    res = res.transpose((1, 2, 0))[:, ::-1]  # [nmodel, ncateg, nsamples]  # AN is the first categ

    n = 0
    for prob_m, freq_o_m, marg_m, rel_m, res_m, name in zip(prob_avg, obs_freq, pred_marginal, rel, res, models_name):
        reliability_diag(prob_m, freq_o_m, marg_m, rel_m, res_m, plt_obj, n, name)
        n+=1
    plt.savefig(figname, bbox_inches='tight', dpi=300)
    plt.close()

def plot_rank_histogram(model_ranks, models_name, figname, nrows, ncols, fsize=(22, 10)):

    plt_obj = MakePlot(fsize)
    nsamples = model_ranks.shape[0]
    model_ranks = model_ranks.transpose((1, 0, 2))
    plt_obj.create_fig_instance(nrows, ncols)
    for i, (name, ranks) in enumerate(zip(models_name, model_ranks)):
        plt_obj.create_subplot_instance(1, 1, i)
        plt_obj.create_axis_instance(0, 1, 0, 1)
        densities_per_sample = []
        for j in range(nsamples):
            y, binEdges = np.histogram(ranks[j][~np.isnan(ranks[j])], bins=np.arange(0, 1.1, 0.1), density=True)
            densities_per_sample.append(y[None])
        densities_per_sample = np.concatenate(densities_per_sample, 0)  # (nsample, nbins)
        density_med = np.nanmedian(densities_per_sample, 0)
        density_lower = np.percentile(densities_per_sample, 5.0, 0)
        density_upper = np.percentile(densities_per_sample, 95.0, 0)
        bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
        width = binEdges[1] - binEdges[0]
        plt_obj.ax.bar(bincenters, density_med, width=width, color='b', yerr=[density_med-density_lower, density_upper-density_med], ecolor='black', capsize=4)
        plt_obj.ax.axhline(1, color='black', linestyle='--')
        plt_obj.ax.set_title(name)
        plt_obj.ax.set_xlabel('Rank [Normalized]')
        plt_obj.ax.set_ylabel('Frequency')
        plt_obj.ax.set_ylim(0., 3.)
    
    plt.savefig(figname, bbox_inches='tight', dpi=300)
    plt.close()

def plot_prob_maps(fig, x, y, z, title='', probs=False):

    fig.ax.set_extent([np.min(x[0]), np.max(x[0]), np.min(y[:, 0]), np.max(y[:, 0])])
    fig.ax.add_feature(shp_ce_feature, edgecolor='black', facecolor='none')
    fig.mask_shape_outside(points_ce, 'grey', ccrs.PlateCarree())
    
    if probs == False:
        colors = ('#DF252B', '#28DF25', '#2539DF')
        cmap = clr.ListedColormap(colors)
        lvls = np.array([0.5, 1.5, 2.5, 3.5])
        norm = BoundaryNorm(lvls, ncolors=cmap.N, clip=False)
        z = z+1.
                
        c = fig.ax.pcolormesh(x, y, z, cmap=cmap, norm=norm)
        fig.ax.set_title(title)
        return c

    else:
        lvls = np.array([0.40, 0.55, 0.70, 0.85])
        cats = np.argmax(z, axis=0)

        z[0] = np.where(cats==0, z[0], 0)
        z[1] = np.where(cats==1, z[1], 0)
        z[2] = np.where(cats==2, z[2], 0)

        colors_BN = ('#FFFB00', '#FF8F00', '#FF0000')
        cmap_BN = clr.ListedColormap(colors_BN)
        cmap_BN.set_over('#8D0000')
        cmap_BN.set_under('#ffffff')
        norm_BN = BoundaryNorm(lvls, ncolors=cmap_BN.N)

        colors_NN = ('#9FFC9D', '#58C255', '#1CC617')
        cmap_NN = clr.ListedColormap(colors_NN)
        cmap_NN.set_over('#068102')
        cmap_NN.set_under('#ffffff')
        norm_NN = BoundaryNorm(lvls, ncolors=cmap_NN.N)

        colors_AN = ('#73FFF5', '#73BDFF', '#0086FD')
        cmap_AN = clr.ListedColormap(colors_AN)
        cmap_AN.set_over('#003665')
        cmap_AN.set_under('#ffffff')
        norm_AN = BoundaryNorm(lvls, ncolors=cmap_AN.N)

        c1 = fig.ax.pcolormesh(x, y, z[0], cmap=cmap_BN, norm=norm_BN, alpha=np.where(cats!=0, 0, 1))
        c2 = fig.ax.pcolormesh(x, y, z[1], cmap=cmap_NN, norm=norm_NN, alpha=np.where(cats!=1, 0, 1))
        c3 = fig.ax.pcolormesh(x, y, z[2], cmap=cmap_AN, norm=norm_AN, alpha=np.where(cats!=2, 0, 1))
        fig.ax.set_title(title)

        return c1, c2, c3

def plot_determ_maps(fig, x, y, z, 
                     lvls=np.arange(-3, 3.5, 0.5), 
                     colors=('#2372C9', '#3498ED', '#4BA7EF', '#76BBF3', '#93D3F6', '#B0F0F7', '#D6FFFF', '#FFFFFF', '#FFFFC5', '#FBE78A', '#FF9D37', '#FF5F26', '#FF2E1B', '#AE000C', '#340003')[::-1],
                     extend='both', 
                     title='',
                     cbar=True):
    
    fig.ax.set_extent([np.min(x[0]), np.max(x[0]), np.min(y[:, 0]), np.max(y[:, 0])])
    fig.ax.add_feature(shp_ce_feature, edgecolor='black', facecolor='none')
    fig.mask_shape_outside(points_ce, 'grey', ccrs.PlateCarree())

    cmap = clr.ListedColormap(colors)
    norm = BoundaryNorm(lvls, ncolors=cmap.N, extend=extend)

    c = fig.ax.pcolormesh(x, y, z, cmap=cmap, norm=norm)
    if cbar:
        cbar = fig.add_colorbar(c, lvls, extend, 'horizontal', cax=fig.ax)
    fig.ax.set_title(title)
    
    return c, cbar

def plot_obs_ypred_maps(x, y, time, Yobs, Ypred, Yobs_anom, Ypred_anom, Yobs_categs, Ypred_probs, varsel_weights, var_names, title, filename, savedir, figsize=(12, 8)):
    
    """
    x: meshed lat lon 2d array 
    y: meshed lat lon 2d array 
    """

    Ypred_categs = np.argmax(Ypred_probs, axis=1)
    Yobs_categs = np.argmax(Yobs_categs, axis=1)
    n = Ypred.shape[0]
    fig = MakePlot(figsize)
    fig.create_fig_instance(n, 4, hspace=0.3)
    k = 0
        
    for i in range(n):
        # Totals
        fig.create_subplot_instance(2, 1, k)
        lvl = [0, 50, 100, 150, 200, 300, 400, 500, 600, 700, 800]
        fig.create_axis_instance(0, 1, 0, 1, ccrs.PlateCarree())
        c, cbar = plot_determ_maps(fig, x, y, Yobs[i], 
                 colors=('#FFFFFF', '#F20000', '#FDB826', '#EDF937', '#A8DD36', '#65C13E', '#28A54B', '#1E8079', '#3254BD', '#6B55FF', '#7B007A'),  
                 extend='max', lvls=lvl, cbar=False)
        fig.add_gridlines()
        fig.ax.set_title(f'Observed totals')
        fig.create_axis_instance(1, 2, 0, 1, ccrs.PlateCarree())
        c, cbar = plot_determ_maps(fig, x, y, Ypred[i], 
                 colors=('#FFFFFF', '#F20000', '#FDB826', '#EDF937', '#A8DD36', '#65C13E', '#28A54B', '#1E8079', '#3254BD', '#6B55FF', '#7B007A'),  
                 extend='max', lvls=lvl, cbar=True)
        fig.add_gridlines()
        cbar.ax.set_xticks(lvl[::2])
        cbar.ax.tick_params(labelsize=5)
        fig.ax.set_title(f'Forecasted totals')

        # Anomalies
        fig.create_subplot_instance(2, 1, k+n)
        lvl = [-1., -0.75, -0.5, -0.25, -0.1, 0.1, 0.25, 0.5, 0.75, 1.]
        fig.create_axis_instance(0, 1, 0, 1, ccrs.PlateCarree())
        c, cbar = plot_determ_maps(fig, x, y, Yobs_anom[i], lvls=lvl, cbar=False)
        fig.add_gridlines()
        fig.ax.set_title(f'Observed std anomalies')
        fig.create_axis_instance(1, 2, 0, 1, ccrs.PlateCarree())
        c, cbar = plot_determ_maps(fig, x, y, Ypred_anom[i], lvls=lvl)
        fig.add_gridlines()
        cbar.ax.tick_params(labelsize=5)
        fig.ax.set_title(f'Forecasted std anomalies')

        # Categories
        fig.create_subplot_instance(2, 1, k+(2*n))
        fig.create_axis_instance(0, 1, 0, 1, ccrs.PlateCarree())
        c = plot_prob_maps(fig, x, y, Yobs_categs[i])
        fig.add_gridlines()
        fig.ax.set_title(f'Observed categories')
        fig.create_axis_instance(1, 2, 0, 1, ccrs.PlateCarree())
        c = plot_prob_maps(fig, x, y, Ypred_categs[i])
        fig.add_gridlines()
        fig.ax.set_title(f'Forecasted categories')

        # Probabilities and variable selection weights
        fig.create_subplot_instance(2, 1, k+(3*n))
        fig.create_axis_instance(0, 1, 0, 1)
        plot_varsel_wgts(fig.ax, varsel_weights, var_names)
        fig.create_axis_instance(1, 2, 0, 1, ccrs.PlateCarree())
        c1, c2, c3 = plot_prob_maps(fig, x, y, Ypred_probs[i], probs=True)
        fig.add_gridlines()
        fig.ax.set_title(f'Forecasted probabilities')

        if i == (n-1):
            ticks = np.array([0.40, 0.55, 0.70, 0.85])
            cbar1 = fig.fig.colorbar(c1, cax=fig.fig.add_axes([0.7325, 0.075, 0.045, 0.026]), orientation='horizontal', ticks=ticks, extend='max', extendfrac=0.2)
            cbar2 = fig.fig.colorbar(c2, cax=fig.fig.add_axes([0.7925, 0.075, 0.045, 0.026]), orientation='horizontal', ticks=ticks, extend='max', extendfrac=0.2)
            cbar3 = fig.fig.colorbar(c3, cax=fig.fig.add_axes([0.8525, 0.075, 0.045, 0.026]), orientation='horizontal', ticks=ticks, extend='max', extendfrac=0.2)
            for cbar in [cbar1, cbar2, cbar3]:
                cbar.ax.set_xticklabels(['40', '55', '70', '85'])
                cbar.ax.tick_params(labelsize=5, direction='in')
        k+=1
    plt.suptitle(title)
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    plt.savefig(f'{savedir}/{filename}', dpi=300, bbox_inches='tight')
    plt.close()

def scorebars(data, xlabels, models_name, init_months, metric_name, result_dir):
    fig = MakePlot((12, 10))
    fig.create_fig_instance(2, 2, wspace=0.1, hspace=0.15)
    x = np.arange(data.shape[3])  # nleads
    width = 0.15
    for i in range(4):
        multiplier = 0
        fig.create_subplot_instance(1, 1, i)
        fig.create_axis_instance(0, 1, 0, 1)
        for j in np.arange(data.shape[1]):  # loop on the number of models
            offset = width * multiplier
            med = np.nanmedian(data[:, j, i, :], axis=0)  # (nleads)
            lower_bar = np.percentile(data[:, j, i, :], 5., axis=0)  # (nleads)
            upper_bar = np.percentile(data[:, j, i, :], 95., axis=0)  # (nleads)
            fig.ax.bar(x+offset, med, width=width, label=models_name[j], yerr=[med-lower_bar, upper_bar-med], ecolor='black', capsize=4)
            multiplier += 1
        fig.ax.set_title(init_months[i])
        fig.ax.set_xticks(x+0.3, xlabels[init_months[i]])
        fig.ax.set_ylim(-50, 50)
        if i == 1:
            fig.ax.legend(bbox_to_anchor=(1.03, 1))
    plt.savefig(f'{result_dir}/{metric_name}_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_mn_varsel_wgts(wgts, var_list, xlabels, init_months, init_months_str, filename, savedir, figsize=(8, 10)):

    fig = MakePlot(figsize)
    n_imons = len(init_months)
    bar_width = 0.75
    fig.create_fig_instance(n_imons, 1)
    for ii, i in enumerate(init_months):
        fig.create_subplot_instance(1, 1, ii)
        fig.create_axis_instance(0, 1, 0, 1)
        wgts_i_j = wgts[i]  # (nsamples, nleads, nvars)
        n_leads = wgts_i_j.shape[1]
        n_vars = wgts_i_j.shape[-1]
        x = np.arange(n_leads)
        bar_width_per_var = bar_width/n_vars
        multiplier = 0
        for k in range(n_vars):
            med = np.median(wgts_i_j[:, :, k], axis=0)
            lower_bar = np.percentile(wgts_i_j[:, :, k], 5.0, axis=0)
            upper_bar = np.percentile(wgts_i_j[:, :, k], 95.0, axis=0)
            fig.ax.bar(x+bar_width_per_var*multiplier, med*100, width=bar_width_per_var, label=var_list[k], yerr=[(med-lower_bar)*100, (upper_bar-med)*100], ecolor='black', capsize=4)
            multiplier += 1
        fig.ax.set_xticks(x+bar_width/2, xlabels[init_months_str[ii]])
        if ii == 0:
            fig.ax.legend(bbox_to_anchor=(1.03, 1))
        fig.ax.text(-0.09, 0.5, init_months_str[ii], rotation=90, horizontalalignment='center', verticalalignment='center', transform=fig.ax.transAxes)
    plt.savefig(f'{savedir}/{filename}', dpi=300, bbox_inches='tight')
    plt.close()

def plot_varsel_wgts(plt_ax, wgts, var_names):

    plt_ax.bar(var_names, wgts*100)
    plt_ax.set_title('Variable selection weights')
    plt_ax.set_xticks(range(len(var_names)))
    plt_ax.set_xticklabels(var_names, fontdict={'fontsize': 10, 'rotation': 20})

def plot_error_maps(coords, errors, titles_errors, figname, suptitle, gridspecs, metric_name, anom=True, cbar_orientation='horizontal', plot_cbar=True, savefig=True):

    """
    gridspecs: dict {'plot_specs': [nrows, ncols, wspace, hspace], 
                     'subplots_specs': [[row1, col1, wspace1, hspace1], [row2, col2, wspace2, hspace2], ...], 
                     'fig_size': (width, height)}
    """
    fsize = gridspecs['fig_size']
    nrows_plot, ncols_plot, wspace_plot, hspace_plot = gridspecs['plot_specs']
    n_plots = len(errors)
    max_rows = max([i[0] for i in gridspecs['subplot_specs']] + [nrows_plot])
    max_cols = max([i[1] for i in gridspecs['subplot_specs']]+ [ncols_plot])
    plots_dict = {}
    next_plot = True
    n = 0
    for i in range(n_plots):
        if next_plot:
            irow = 0
            icol = 0
            nrows_subplot, ncols_subplot, wspace_subplot, hspace_subplot = gridspecs['subplot_specs'][n]
            next_plot = False
        plots_dict[i] = [n, irow, icol, nrows_subplot, ncols_subplot, wspace_subplot, hspace_subplot]
        if irow+1 == nrows_subplot and icol+1 != ncols_subplot:
            irow = 0
            icol+=1
        elif irow+1 != nrows_subplot and icol+1 == ncols_subplot:
            irow+=1
            icol = 0
        elif irow+1 == nrows_subplot and icol+1 == ncols_subplot:
            n+=1
            next_plot = True
        
    if metric_name == 'RMSE' or metric_name == 'MSE':
        if anom:
            lvls = np.arange(0.3, 1.4, 0.1)
        else:
            lvls = np.arange(0., 550., 50.)
        cmap = plt.get_cmap('YlOrRd')
        norm = BoundaryNorm(lvls, ncolors=cmap.N, clip=False)
        cmap.set_over('#000000')
        cmap.set_under('#FFFFFF')
        extend = 'both'
    elif metric_name == 'RPS' or metric_name == 'CRPS':
        lvls = np.arange(0., 1.1, 0.1)
        cmap = plt.get_cmap('hot_r')
        norm = BoundaryNorm(lvls, ncolors=cmap.N, clip=False, extend='max')
        extend = 'max'
    elif metric_name == 'Ignorance':
        lvls = np.arange(0., 4.4, 0.4)
        cmap = plt.get_cmap('hot_r')
        norm = BoundaryNorm(lvls, ncolors=cmap.N-1, clip=False)
        extend = 'max'
    elif metric_name == 'Bias':
        if anom:
           lvls = np.arange(-1., 1.2, 0.2)
        else:
            lvls = np.arange(-400., 440., 40.)
        cmap = plt.get_cmap('RdBu')
        norm = BoundaryNorm(lvls, ncolors=cmap.N, clip=False)
        cmap.set_under('#000000')
        cmap.set_over('#000044')
        extend = 'both'
    else:
        lvls = np.arange(-1., 1.25, 0.25)        
        colors = ('#0033FF', '#007FFF', '#00CCFF', '#CCCCCC',
                  '#CCCCCC', '#FF9900', '#FF3300', '#A50000')
        cmap = clr.ListedColormap(colors)
        norm = BoundaryNorm(lvls, ncolors=cmap.N, clip=False)
        extend = 'neither'
    
    plt_obj = MakePlot(fsize)
    plt_obj.create_fig_instance(nrows_plot, ncols_plot, wspace_plot, hspace_plot)
    k = 0
    for coord, i, t in zip(coords, errors, titles_errors):
        x, y = coord[0], coord[1]
        n, irow, icol, nrows, ncols, wspace, hspace = plots_dict[k]
        plt_obj.create_subplot_instance(nrows, ncols, n, wspace, hspace)
        plt_obj.create_axis_instance(irow, irow+1, icol, icol+1, ccrs.PlateCarree())
        plt_obj.ax.set_extent([np.min(x[0]), np.max(x[0]), np.min(y[:, 0]), np.max(y[:, 0])], crs=ccrs.PlateCarree())
        plt_obj.add_gridlines()
        c = plt_obj.ax.pcolormesh(x, y, i, cmap=cmap, norm=norm)
        plt_obj.mask_shape_outside(points_ce, 'grey', ccrs.PlateCarree())
        plt_obj.ax.set_title(t)
        k+=1

    if plot_cbar:
        plt_obj.add_colorbar(c, lvls, extend, cbar_orientation)

    if savefig:
        plt.suptitle(suptitle)
        plt.savefig(figname, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        return c, plt_obj
