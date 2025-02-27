import matplotlib
matplotlib.rcParams['backend'] = 'Agg'
import numpy as np
# from cartopy import feature
from cartopy.feature import ShapelyFeature
from shapely.geometry import Polygon
from matplotlib import pyplot as plt
from matplotlib import colors as clr
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec


## Colorbars
correl_colors = ('#00FFFF', '#0DCAFF', '#1A94FF', '#285EFF', '#4141FF', '#8989FF', '#D1D1FF', '#FFFFFF', '#FFFFFF',
                '#FFFFFF', '#FFFFFF', '#FFD1D1', '#FF8989', '#FF4141', '#FF5E28', '#FF9417',  '#FFFF00')

BlRd_pallete = ('#0033FF', '#4141FF', '#285EFF', '#007FFF', '#00CCFF', '#FFFFFF',
                '#FFFFFF', '#FFCA0D', '#FF9900', '#FF5E28', '#FF3300', '#A50000')
BlRd_camp_neither = clr.ListedColormap(BlRd_pallete)
BlRd_camp_both = clr.ListedColormap(BlRd_pallete[1:-1])
BlRd_camp_both.set_under(BlRd_pallete[0])
BlRd_camp_both.set_over(BlRd_pallete[-1])

class MakePlot:

    def __init__(self, figsize):

        self.fs = figsize
        self.fig = plt.figure(figsize=self.fs)

    def create_fig_instance(self, nrows, ncols, wspace=None, hspace=None, wratios=None, hratios=None):

        self.gs0 = GridSpec(nrows, ncols, wspace=wspace, hspace=hspace, width_ratios=wratios, height_ratios=hratios)    

    def create_subplot_instance(self, nrows, ncols, gs_pos, wspace=None, hspace=None, hratios=None, wratios=None):

        self.gs1 = GridSpecFromSubplotSpec(nrows, ncols, subplot_spec=self.gs0[gs_pos], wspace=wspace, hspace=hspace, width_ratios=wratios, height_ratios=hratios)

    def create_axis_instance(self, irow, frow, icol, fcol, proj=None):

        self.ax = self.fig.add_subplot(self.gs1[irow:frow, icol:fcol], projection=proj)

    def add_gridlines(self, pos=[0, 1, 0, 1]):
        gl = self.ax.gridlines(draw_labels=False, linestyle='--', color='black', linewidth=0.3, zorder=4)
        gl.xlabel_style = {'size': 8}
        gl.ylabel_style = {'size': 8}
        gl.top_labels, gl.bottom_labels, gl.right_labels, gl.left_labels = pos
    
    # def fill_continent(self, color):

    #     self.ax.add_feature(feature.LAND, edgecolor='k', facecolor=color)
    
    # def coastlines(self):

    #     self.ax.add_feature(feature.LAND, edgecolor='k', facecolor='none')
    
    def mask_shape_outside(self, shp_poly, color, proj=None):
        x0,x1 = self.ax.get_xlim()
        y0,y1 = self.ax.get_ylim()
        xs = [x1, x0, x0, x1, x1]
        ys = [y1, y1, y0, y0, y0]
        rect = Polygon([(x, y) for x, y in zip(xs, ys)])
        mask = rect.difference(shp_poly)
        mask_feature = ShapelyFeature([mask], proj)
        self.ax.add_feature(mask_feature, facecolor=color, zorder=3, edgecolor='none', alpha=1)
    
    def add_colorbar(self, mappable, ticks, extend, orientation, cax=None, x_offset=0.015, y_offset=0.05, **kwargs):

        if cax == None:
            if orientation == 'vertical':
                y0 = min([i.get_position().y0 for i in self.fig.axes[:]])
                y1 = max([i.get_position().y1 for i in self.fig.axes[:]])
                x1 = max([i.get_position().x1 for i in self.fig.axes[:]])
                cax = self.fig.add_axes([x1+x_offset, y0, 0.015, y1-y0])
            
            elif orientation == 'horizontal':
                x0 = min([i.get_position().x0 for i in self.fig.axes[:]])
                x1 = max([i.get_position().x1 for i in self.fig.axes[:]])
                y0 = min([i.get_position().y0 for i in self.fig.axes[:]])
                cax = self.fig.add_axes([x0, y0-y_offset, x1-x0, 0.02])
        
        else:
            if orientation == 'vertical':
                y0 = cax.get_position().y0
                y1 = cax.get_position().y1
                x1 = cax.get_position().x1
                cax = self.fig.add_axes([x1+x_offset, y0, 0.015, y1-y0])
            elif orientation == 'horizontal':
                x0 = cax.get_position().x0
                x1 = cax.get_position().x1
                y0 = cax.get_position().y0
                cax = self.fig.add_axes([x0, y0-y_offset, x1-x0, 0.02])

        cbar = self.fig.colorbar(mappable, cax=cax, orientation=orientation, ticks=ticks, extend=extend, **kwargs)
        return cbar
