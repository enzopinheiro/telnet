import os
import cartopy
import matplotlib
matplotlib.rcParams['backend'] = 'Agg'
import numpy as np
import cartopy.crs as ccrs
from typing import List
from collections import Counter
from matplotlib import pyplot as plt
from matplotlib import colors as clr
from matplotlib import patches
from matplotlib.colors import BoundaryNorm
from .plotting_utils import MakePlot
# from utils import points_ce, shp_ce_feature


def plot_model_freq(model_freq, top_model, savedir):  

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
    # plot.create_fig_instance(1, 1)
    plot = MakePlot((25, 10))
    plot.create_fig_instance(1, 1)
    plot.create_subplot_instance(1, nfeats, 0)
    for i in range(nfeats):
        plot.create_axis_instance(0, 1, i, i+1)
        feat_freq = Counter(feat_order[:, i])
        feat_freq = {k: v/nsamples for k, v in feat_freq.items()}
        plt.bar(feat_freq.keys(), feat_freq.values())
        plt.xlabel('Feature ID')
        plt.ylabel('Frequency')

    plt.savefig(os.path.join(savedir, f'feat_freq.png'), dpi=300, bbox_inches='tight')
    plt.close()

def reliability_diag(bins, calibration_dist, refinement_dist, rel, res, plt_obj, ngrid, title, fsize=10):

    colors = ['red', 'green', 'blue'][::-1]
    
    plt_obj.create_subplot_instance(3, 3, ngrid, 0.25, 0.2)
    plt_obj.create_axis_instance(0, 3, 0, 2)
    ax1 = plt_obj.ax
    nbins = bins.shape[1]
    bins1 = np.linspace(0, 1, nbins+1)
    for i in range(3):
        calibration_dist_med = np.nanmedian(calibration_dist[i], -1)  # (nbin)
        upper_bar = np.nanpercentile(calibration_dist[i], 95., -1) - calibration_dist_med
        lower_bar = calibration_dist_med - np.nanpercentile(calibration_dist[i], 5., -1)
        ax1.errorbar(np.nanmedian(bins[i], -1), calibration_dist_med, yerr=np.concatenate((lower_bar[np.newaxis], upper_bar[np.newaxis]), 0), c=colors[i], capsize=4.0)
        ax1.plot(bins1, bins1, 'k--')
        # if ngrid in np.arange(3, 6):
        ax1.set_xlabel('Forecast probability')
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
        if i == 2:
            ax2.set_xticks(xticks)
            ax2.set_xticklabels(xticks)
            ax2.set_xlabel('Forecast bin')
        else:
            ax2.set_xticks(xticks)
            ax2.set_xticklabels([])

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
    fig.ax.add_feature(cartopy.feature.OCEAN, zorder=100, edgecolor='k', facecolor='gray')
    
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
    fig.ax.add_feature(cartopy.feature.OCEAN, zorder=100, edgecolor='k', facecolor='gray')

    cmap = clr.ListedColormap(colors)
    norm = BoundaryNorm(lvls, ncolors=cmap.N, extend=extend)

    c = fig.ax.pcolormesh(x, y, z, cmap=cmap, norm=norm)
    if cbar:
        cbar = fig.add_colorbar(c, lvls, extend, 'horizontal', cax=fig.ax)
    fig.ax.set_title(title)
    
    return c, cbar

def plot_obs_ypred_maps(x, y, time, Yobs, Ypred, Yobs_anom, Ypred_anom, Yobs_categs, Ypred_probs, varsel_weights, var_names, title, filename, savedir, figsize=(12, 8),
                        levels_totals=[0, 50, 100, 150, 200, 300, 400, 500, 600, 700, 800]):
    
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
        lvl = levels_totals
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
            x0 = fig.ax.get_position().x0
            x1 = fig.ax.get_position().x1
            cax_length = ((x1 - x0)/3)
            y0 = fig.ax.get_position().y0
            cax1 = fig.fig.add_axes([x0, y0-0.05, cax_length, 0.015])  # Add a new axes for the colorbar
            cax2 = fig.fig.add_axes([x0+cax_length, y0-0.05, cax_length, 0.015])
            cax3 = fig.fig.add_axes([x0+cax_length*2, y0-0.05, cax_length, 0.015])
            cbar1 = fig.fig.colorbar(c1, cax=cax1, orientation='horizontal', ticks=ticks, extend='max', extendfrac=0.2)
            cbar2 = fig.fig.colorbar(c2, cax=cax2, orientation='horizontal', ticks=ticks, extend='max', extendfrac=0.2)
            cbar3 = fig.fig.colorbar(c3, cax=cax3, orientation='horizontal', ticks=ticks, extend='max', extendfrac=0.2)
            for cbar in [cbar1, cbar2, cbar3]:
                cbar.ax.set_xticklabels(['40', '55', '70', '85'])
                cbar.ax.tick_params(labelsize=5, direction='in')
        k+=1
    plt.suptitle(title)
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    plt.savefig(f'{savedir}/{filename}', dpi=300, bbox_inches='tight')
    plt.close()

def plot_rps_global_avg(mean_RPS_avg, models_titles, conf_interval, result_dir, filename):

    plt.figure(figsize=(15, 5))
    x = np.arange(len(models_titles))
    plt.errorbar(x, mean_RPS_avg, yerr=conf_interval, capsize=5, color='k', ecolor='black', elinewidth=1)
    plt.xticks(x, models_titles)
    plt.ylabel('RPS avg')
    # plt.title('RPS_avg with 95% Confidence Intervals')
    plt.tight_layout()
    plt.savefig(f'{result_dir}/{filename}.png', dpi=300)
    plt.close()

def scorebars(data, xlabels, models_name, init_months, metric_name, result_dir, ylim=(-50, 50), figsize=(12, 10)):
    fig = MakePlot(figsize)
    fig.create_fig_instance(2, 2, wspace=0.1, hspace=0.15)
    x = np.arange(data.shape[3])  # nleads
    width = 0.8 / data.shape[1]
    offset_xticks = width * (data.shape[1]//2) if data.shape[1] % 2 == 0 else width * (data.shape[1]//2) + width/2
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
        fig.ax.set_xticks(x+offset_xticks, xlabels[init_months[i]])
        fig.ax.set_ylim(ylim[0], ylim[1])
        if i == 1:
            fig.ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1), bbox_transform=fig.ax.transAxes, alignment='right')

    plt.savefig(f'{result_dir}/{metric_name}_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_varsel_wgts(plt_ax, wgts, var_names):

    plt_ax.bar(var_names, wgts*100)
    plt_ax.set_title('Variable selection weights')
    plt_ax.set_xticks(range(len(var_names)))
    plt_ax.set_xticklabels(var_names, fontdict={'fontsize': 8})

def plot_varsel_maps(fig, varsel, x, y, index_name):

    cmap = plt.get_cmap('viridis')
    lvls = np.arange(0., 1.1, 0.1)
    norm = BoundaryNorm(lvls, ncolors=cmap.N, clip=False, extend='neither')

    fig.ax.add_feature(cartopy.feature.OCEAN, zorder=100, edgecolor='k', facecolor='gray')
    fig.ax.set_extent([np.min(x[0]), np.max(x[0]), np.min(y[:, 0]), np.max(y[:, 0])])
    fig.add_gridlines()
    c = fig.ax.pcolormesh(x, y, varsel, cmap=cmap, norm=norm)
    fig.ax.set_title(index_name.upper())
    
    return c

def plot_boxplot(avg_rps, result_dir, var):
    fig = plt.figure(figsize=(10, 6))
    fig.ax = fig.add_subplot(111)
    plt.boxplot(avg_rps, positions=range(0, len(avg_rps)), showmeans=True)
    plt.xlabel('Config Number')
    plt.ylabel(f'Average {var}')
    plt.title(f'Distribution of validation {var} per config')
    fig.ax.xaxis.set_major_locator(plt.MultipleLocator(5))
    fig.ax.xaxis.set_major_formatter('{x:.0f}')
    fig.ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, f'avg_{var}_boxplot.png'), dpi=500)
    plt.close()

def plot_mean_varsel_maps(varsel, lat, lon, leads, indices_name, savedir, figname, gridspecs, plot_cbar=True, cbar_orientation='horizontal', savefig=True):

    fsize = gridspecs['fig_size']
    nrows_plot, ncols_plot, wspace_plot, hspace_plot = gridspecs['plot_specs']
    nrows_subplot, ncols_subplot, wspace_subplot, hspace_subplot = gridspecs['subplot_specs']
   
    for m, lead in enumerate(leads):
        fig = MakePlot(fsize)
        fig.create_fig_instance(nrows_plot, ncols_plot, wspace_plot, hspace_plot)
        fig.create_subplot_instance(nrows_subplot, ncols_subplot, 0, wspace_subplot, hspace_subplot)
        cmap = plt.get_cmap('viridis')
        lvls = np.arange(0., 1.1, 0.1)
        norm = BoundaryNorm(lvls, ncolors=cmap.N, clip=False)
        extend = 'neither'
        for mm, index in enumerate(indices_name):
            irow = mm // ncols_subplot
            icol = mm % ncols_subplot
            fig.create_axis_instance(irow, irow+1, icol, icol+1, ccrs.PlateCarree())
            fig.ax.add_feature(cartopy.feature.OCEAN, zorder=100, edgecolor='k', facecolor='gray')
            fig.ax.set_extent([np.min(lon), np.max(lon), np.min(lat), np.max(lat)], crs=ccrs.PlateCarree())
            fig.add_gridlines()
            c = fig.ax.pcolormesh(lon, lat, np.median(varsel, 0)[m, mm], cmap=cmap, norm=norm)
            fig.ax.set_title(index.upper())
        if plot_cbar:
            fig.add_colorbar(c, lvls, extend, cbar_orientation)

        if savefig:
            plt.savefig(f'{savedir}/{figname}_{lead}.png', dpi=300, bbox_inches='tight')
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

def plot_error_maps(coords, errors, titles_errors, figname, suptitle, gridspecs, metric_name, anom=True, cbar_orientation='horizontal', plot_cbar=True, savefig=True, draw_rects=False):

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
            lvls = np.arange(0.4, 2.4, 0.2)
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
        if draw_rects:
            # rect_SE = patches.Rectangle((-60, -35), 10, 10, linewidth=1, edgecolor='k', facecolor='none', transform=ccrs.PlateCarree(), zorder=2)
            rect_N = patches.Rectangle((-70, -5), 20, 15, linewidth=1, edgecolor='k', facecolor='none', transform=ccrs.PlateCarree(), zorder=2)
            # plt_obj.ax.add_patch(rect_SE)
            plt_obj.ax.add_patch(rect_N)
        plt_obj.ax.add_feature(cartopy.feature.OCEAN, zorder=100, edgecolor='k', facecolor='gray')
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

def plot_pca_maps(loadings, coefs, coords, models_name, savedir, figname, gridspecs, plot_cbar=True, cbar_orientation='horizontal', savefig=True):

    """
    x: meshed lat lon 2d array
    y: meshed lat lon 2d array
    """

    fsize = gridspecs['fig_size']
    nrows_plot, ncols_plot, wspace_plot, hspace_plot = gridspecs['plot_specs']
    n_plots = len(models_name)
    n_components = loadings.shape[1]
    plots_dict = {}
    next_plot = True
    n = 0
    for i in range(n_plots):
        if next_plot:
            irow = 0
            icol = 0
            nrows_subplot, ncols_subplot, wspace_subplot, hspace_subplot = gridspecs['subplot_specs'][n]
            next_plot = False
        plots_dict[i] = [n, irow, irow+1, icol, icol+1, nrows_subplot, ncols_subplot, wspace_subplot, hspace_subplot]
        if irow+1 == nrows_subplot and icol+1 != ncols_subplot:
            irow = 0
            icol+=1
        elif irow+1 != nrows_subplot and icol+1 == ncols_subplot:
            irow+=1
            icol = 0
        elif irow+1 != nrows_subplot and icol+1 != ncols_subplot:
            icol+=1 
        elif irow+1 == nrows_subplot and icol+1 == ncols_subplot:
            n+=1
            next_plot = True
    
    for m in range(n_components):
        fig = MakePlot(fsize)
        fig.create_fig_instance(nrows_plot, ncols_plot, wspace_plot, hspace_plot)
        k = 0
        cmap = plt.get_cmap('RdBu')
        init_lvl = (np.max(np.abs(loadings[:, m]))).round(2)
        lvls = np.linspace(-init_lvl, init_lvl, 11)
        norm = BoundaryNorm(lvls, ncolors=cmap.N, clip=False)
        extend = 'both'
        mm = 0
        for name, coord in zip(models_name, coords):
            x, y = coord[0], coord[1]
            n, irow, frow, icol, fcol, nrows, ncols, wspace, hspace = plots_dict[k]
            fig.create_subplot_instance(nrows, ncols, n, wspace, hspace)
            fig.create_axis_instance(irow, frow, icol, fcol, ccrs.PlateCarree())
            fig.ax.set_extent([np.min(x[0]), np.max(x[0]), np.min(y[:, 0]), np.max(y[:, 0])], crs=ccrs.PlateCarree())
            fig.ax.add_feature(cartopy.feature.OCEAN, zorder=100, edgecolor='k', facecolor='gray')
            fig.add_gridlines()
            c = fig.ax.pcolormesh(x, y, loadings[mm, m], cmap=cmap, norm=norm)
            fig.ax.set_title(f'{name}\nExplained Variance: {coefs[mm, m]:.2f}', fontdict={'fontsize': 8})
            k+=1
            mm+=1
        
        if plot_cbar:
            fig.add_colorbar(c, lvls, extend, cbar_orientation)

        if savefig:
            plt.savefig(f'{savedir}/{figname}_{m+1}.png', dpi=300, bbox_inches='tight')
            plt.close()