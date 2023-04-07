from __future__ import print_function
import os
import sys
import json
from pickle import NONE
from re import T, U, X
from tkinter import W
import cmaps
import datetime
import linecache
from glob       import glob
from pathlib    import Path 
from time       import time


import cmaps
import numpy  as np
import pandas as pd
import xarray as xr

import cartopy.crs       as ccrs   
import cartopy.feature   as cfeat 
import matplotlib.pyplot as plt
import matplotlib.colors as col
import matplotlib.ticker as ticker
import matplotlib.dates  as mdates
import scipy.ndimage     as ndimage

from xgrads     import CtlDescriptor
from xgrads     import open_CtlDataset
from netCDF4    import Dataset

import wrf
from   wrf                   import getvar, ALL_TIMES, ll_to_xy, CoordPair, pw, vertcross, latlon_coords, interplevel
from   wrf.extension         import (_tv,_pw)
from   cartopy.mpl.ticker    import LongitudeFormatter,LatitudeFormatter 
from   cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
import cartopy.io.shapereader as shpreader

import metpy.calc as mpcalc
from metpy.plots import SkewT
from metpy.units import units

def save_pic(fig=None, savepath=None, savename=None,if_resave=True):
    savefile=os.path.join(savepath,savename)
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    if not os.path.exists(savefile):
        print('saving pic: '+savefile)
        try:
            fig.savefig(savefile, bbox_inches='tight')
        except:
            fig.savefig(savefile)
        plt.close()
    else:
        if if_resave:
            print('re-saving pic: '+savefile)
            os.remove(savefile)
            try:
                fig.savefig(savefile, bbox_inches='tight')
            except:
                fig.savefig(savefile)
            plt.close()
        else:
            print(savefile,' exist')

def add_scatter_map(fig, ax, plon, plat, pval,levels,cmap,norm,zorder=1):
    cs   = ax.scatter(plon, plat, c=pval, s=18, cmap=cmap,norm=norm, edgecolors='black')
    # cbar = plt.colorbar(cs, shrink=0.6, pad=0.02, aspect=50,fraction=0.2,orientation='horizontal',extend='both')
    return fig

def draw_contour_map(fig,ax,lats,lons,var,data_proj,plot_proj,levels,cmap,norm,lat_s=26,lat_e=34.5,lon_s=97.2,lon_e=108.7,tick_inv=2):

    # process lat lon
    if lons.ndim == 2 and lats.ndim == 2:
        lon2d, lat2d = lons, lats
    else:
        lon2d, lat2d = np.meshgrid(lons, lats)

    # process time
    try:
        time     = pd.to_datetime(var.time.values).strftime("%Y-%m-%d %H:%M:%S")
    except:
        time     = pd.to_datetime(var.Time.values).strftime("%Y-%m-%d %H:%M:%S")

    # add shp
    PATH_SRC = r'/public/home/ipm_zhengq/local/SHPFILE'
    shpfile  = str(Path(PATH_SRC) / r'Province.shp')
    # shpfile  = str(Path(PATH_SRC) / r'SiChuan_Province.shp')
    sc_shp   = list(shpreader.Reader(shpfile).geometries())
    ax.add_geometries(sc_shp, ccrs.PlateCarree(), edgecolor='k', linewidth=1.2, linestyle='-', facecolor='None')

    shpfile  = str(Path(PATH_SRC) / r'Sichuan.shp')
    sw_shp   = list(shpreader.Reader(shpfile).geometries())
    # ax.add_geometries(sw_shp, ccrs.PlateCarree(), edgecolor='b', linewidth=1.2, facecolor='None')

    # ticks
    ax.set_xticks(np.arange(int(lon_s),lon_e,tick_inv),  crs=plot_proj)
    ax.set_yticks(np.arange(int(lat_s),lat_e,tick_inv),  crs=plot_proj)
    ax.set_extent([lon_s, lon_e, lat_s, lat_e],  crs=plot_proj)
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.tick_params(labelsize=16)

    # title
    try:
        ax.set_title(var.long_name, loc='left', fontsize='large')
    except:
        try:
            ax.set_title(var.description,loc='left', fontsize='large')
        except:
            print('no description')
    ax.set_title(time,              loc='right',fontsize='large')

    # plot
    ac   = ax.contourf(lon2d,lat2d,var,levels=levels,norm=norm,cmap=cmap,extend='both',transform=data_proj)

    # add colorbar
    l,b,w,h = 0.25, 0.01, 0.5, 0.03
    rect = [l,b,w,h]
    cbar_ax = fig.add_axes(rect)
    cb = plt.colorbar(ac, cax = cbar_ax,orientation='horizontal',spacing='proportional')
    try:
        cb.set_label(var.units,loc='center')
    except:
        print('no units')
    cb.ax.tick_params(labelsize=16)
    # cb.formatter.set_powerlimits((0, 0))
    return ax

def add_windbar(ax,lats,lons,u,v,data_proj):

    # process lat lon
    if lons.ndim == 2 and lats.ndim == 2:
        lon2d, lat2d = lons, lats
    else:
        lon2d, lat2d = np.meshgrid(lons, lats)
    ac   = ax.barbs(lon2d,lat2d,u,v,pivot='tip',color='purple',barb_increments=dict(half=2,full=4,flag=20), transform=data_proj,length=5,sizes=dict(emptybarb=0, spacing=0.3, height=0.5))

def add_quiver(ax,lats,lons,u,v,data_proj,scale=250):

    # process lat lon
    if lons.ndim == 2 and lats.ndim == 2:
        lon2d, lat2d = lons, lats
    else:
        lon2d, lat2d = np.meshgrid(lons, lats)
    quiver = ax.quiver(lon2d,lat2d,u,v,color='k',pivot='mid', transform=data_proj,width=0.004, scale=scale,headwidth=3,headlength=3)
    return quiver

def plot_post_T():

    plot_levels     = [700.0]
    lat_s           = 20.
    lat_e           = 40.
    lon_s           = 90.
    lon_e           = 115.
    savepath        = r'/public/home/ipm_zhengq/local/CMA_MESO/GRAPES_MESO5.1_op/fcst/grapes_model/run'

    # READ MULTI WRFOUT
    wrfout_filepath = r'/public/home/ipm_zhengq/local/CMA_MESO/GRAPES_MESO5.1_op/fcst/grapes_model/run'
    wrfout_filename = r'output.nc'
    wrfout_file_str = os.path.join(wrfout_filepath,wrfout_filename)
    ds              = xr.open_dataset(wrfout_file_str)
    ds              = ds.rename({'time':'Time','lev':'level'})
    var             = ds['t']
    lats            = var.coords['lat']
    lons            = var.coords['lon']

    # Intersection of plot and data levels
    plot_levels     = list(map(int, plot_levels))   # convert to int
    plot_levels     = list(set(plot_levels) & set(var.coords['level'].values))

    # PLOT
    # set levels and colormap
    levels          = np.arange(1,15,1)*0.5 +8 + 273
    cmap            = cmaps.temp_19lev
    idx             = np.round(np.linspace(0, cmap.N - 1, len(levels) + 1)).astype(int)    # extend=both
    colors          = cmap(idx)
    colormap, norm  = col.from_levels_and_colors(levels, colors, extend='both')

    for plot_time in var.coords['Time']:
        for plot_level in plot_levels:
            print('-------------plot:  ',plot_level)
            # creat canvas
            fig         = plt.figure(figsize=(12,6),dpi=150)
            ax          = fig.add_subplot(111,projection = ccrs.PlateCarree())

            # get var
            var_sel     = var.sel(Time=plot_time,level=plot_level).squeeze()


            # drw
            ax          = draw_contour_map(fig,ax,lats,lons,var_sel,ccrs.PlateCarree(),ccrs.PlateCarree(),levels,colormap,norm,
                                                lat_s=lat_s,lat_e=lat_e,lon_s=lon_s,lon_e=lon_e,tick_inv=4)

            # save
            date_str        = pd.to_datetime(var_sel.coords['Time'].values).strftime("%Y%m%d%H")
            savename        = 'Tc_'+str(plot_level)+'_'+date_str+'.png'
            save_pic(fig,savepath,savename)

def plot_post_QV():

    plot_levels     = [850.0]
    lat_s           = 20.
    lat_e           = 40.
    lon_s           = 90.
    lon_e           = 115.
    savepath        = r'/public/home/ipm_zhengq/local/CMA_MESO/GRAPES_MESO5.1_op/fcst/grapes_model/run'

    # READ MULTI WRFOUT
    wrfout_filepath = r'/public/home/ipm_zhengq/local/CMA_MESO/GRAPES_MESO5.1_op/fcst/grapes_model/run'
    wrfout_filename = r'output.nc'
    wrfout_file_str = os.path.join(wrfout_filepath,wrfout_filename)
    ds              = xr.open_dataset(wrfout_file_str)
    ds              = ds.rename({'time':'Time','lev':'level'})
    var             = ds['qv']
    var.values      = var.values*1000
    lats            = var.coords['lat']
    lons            = var.coords['lon']

    # Intersection of plot and data levels
    plot_levels     = list(map(int, plot_levels))   # convert to int
    plot_levels     = list(set(plot_levels) & set(var.coords['level'].values))

    # PLOT
    # set levels and colormap
    levels          = np.arange(1,10,1)*1.0 +10   # 850
    cmap            = cmaps.CBR_wet
    idx             = np.round(np.linspace(0, cmap.N - 1, len(levels) + 1)).astype(int)    # extend=both
    colors          = cmap(idx)
    colormap, norm  = col.from_levels_and_colors(levels, colors, extend='both')

    for plot_time in var.coords['Time']:
        for plot_level in plot_levels:
            print('-------------plot:  ',plot_level)
            # creat canvas
            fig         = plt.figure(figsize=(12,6),dpi=150)
            ax          = fig.add_subplot(111,projection = ccrs.PlateCarree())

            # get var
            var_sel     = var.sel(Time=plot_time,level=plot_level).squeeze()


            # drw
            ax          = draw_contour_map(fig,ax,lats,lons,var_sel,ccrs.PlateCarree(),ccrs.PlateCarree(),levels,colormap,norm,
                                                lat_s=lat_s,lat_e=lat_e,lon_s=lon_s,lon_e=lon_e,tick_inv=4)

            # save
            date_str        = pd.to_datetime(var_sel.coords['Time'].values).strftime("%Y%m%d%H")
            savename        = 'Qv_'+str(plot_level)+'_'+date_str+'.png'
            save_pic(fig,savepath,savename)

def plot_post_static():
    plt.rc('font',family='Arial')
    lat_s           = 17.
    lat_e           = 41.
    lon_s           = 78.
    lon_e           = 116.7
    if_simulation_domain = True
    savepath        = r'/public/home/ipm_zhengq/local/TEST/CMA_MESO/PLOT/PIC/Terrain'

    # READ MULTI WRFOUT
    filepath        = r'/public/home/ipm_zhengq/local/TEST/CMA_MESO/GRAPES_MESO5.1_op/DATABAK/cold/2022071700'
    filename_ctl    = r'postvar.ctl_202207170000000'
    file_ctl        = os.path.join(filepath,filename_ctl)
    ctl             = CtlDescriptor(file=file_ctl)
    dset            = open_CtlDataset(CtlDescriptor(file=file_ctl))
    var             = dset['zs'].load()
    lats            = var.coords['lat']
    lons            = var.coords['lon']

    # PLOT
    # set levels and colormap
    levels          = np.arange(1,25,1)*200.0 
    cmap            = cmaps.topo_15lev
    cmap            = plt.cm.get_cmap('terrain')
    cmap_idx_s      = 80
    idx             = np.round(np.linspace(cmap_idx_s, cmap.N - 1, len(levels) + 1)).astype(int)    # extend=both

    # for specifical level
    cmap_idx_ocean  = 30
    level_ocean     = 0
    idx             = np.insert(idx,0,cmap_idx_ocean)
    levels          = np.insert(levels,0,level_ocean)
    colors          = cmap(idx)
    colormap, norm  = col.from_levels_and_colors(levels, colors, extend='both')

    for plot_time in var.coords['time']:
        # creat canvas
        fig         = plt.figure(figsize=(12,6),dpi=150)
        ax          = fig.add_subplot(111,projection = ccrs.PlateCarree())

        # get var
        var_sel     = var.sel(time=plot_time).squeeze()

        # drw
        ax          = draw_contour_map(fig,ax,lats,lons,var_sel,ccrs.PlateCarree(),ccrs.PlateCarree(),levels,colormap,norm,
                                            lat_s=lat_s,lat_e=lat_e,lon_s=lon_s,lon_e=lon_e,tick_inv=5)
        ax.set_title('', loc='right', fontsize='large')
        ax.set_title('GRAPES Domain ', loc='center', fontsize=18)
        # save
        date_str        = pd.to_datetime(var_sel.coords['time'].values).strftime("%Y%m%d%H")
        savename        = 'terrain_'+date_str+'.png'
        save_pic(fig,savepath,savename)

def main():

    # plot_post_T()    
    # plot_post_QV()
    plot_post_static() 

def test():
    print(np.trapz([1,2,3]))
    print(np.trapz([1,0,3]))
if __name__ == '__main__':
    main()



