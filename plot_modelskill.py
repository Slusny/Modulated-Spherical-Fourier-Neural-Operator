#
import xarray as xr
import os
import numpy as np
import sys
import json
from datetime import datetime
import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
from time import sleep, time

from multiprocessing import Pool, active_children
from S2S_on_SFNO.Models.provenance import system_monitor

variable = "2m_temperature"
timestp = "20240201-1555"
path   = "/mnt/V/Master/climate/skillscores/"+variable+"/"+timestp+"/"
save_path = "/mnt/V/Master/climate/skillscores/"+variable+"/"+timestp+"/"
monitor_savepath = os.path.join(save_path,"monitor_parllel_"+datetime.now().strftime("%Y%m%d-%H%M")+".json")

variable_index = 2

variables = [
    ('10m_u_component_of_wind', '10u'),
    ('10m_v_component_of_wind','10v'),
    ('2m_temperature','2t'),
    ('total_column_water_vapour','tcwv')
]
variable = variables[variable_index][0]
dataset_var = variables[variable_index][1]

save_interval = 100
end = 200

savepath_sfno = os.path.join(save_path,"plot_sfno")
savepath_fcn  = os.path.join(save_path,"plot_fcn")
if not os.path.exists(savepath_sfno): os.makedirs(savepath_sfno)
if not os.path.exists(savepath_fcn): os.makedirs(savepath_fcn)

savepath_sfno_log = os.path.join(save_path,"plot_sfno_log")
savepath_fcn_log  = os.path.join(save_path,"plot_fcn_log")
if not os.path.exists(savepath_sfno_log): os.makedirs(savepath_sfno_log)
if not os.path.exists(savepath_fcn_log): os.makedirs(savepath_fcn_log)

savepath_sfno_globe = os.path.join(save_path,"plot_sfno_globe")
savepath_fcn_globe  = os.path.join(save_path,"plot_fcn_globe")
if not os.path.exists(savepath_sfno_globe): os.makedirs(savepath_sfno_globe)
if not os.path.exists(savepath_fcn_globe): os.makedirs(savepath_fcn_globe)

def plot(idx):
    print(idx)
    s = (idx+1)*6
    file_sfno = os.path.join(path,'sfno','rmse_global_sfno_10m_u_component_of_wind_step_{}.nc').format(s)
    file_fcn = os.path.join(path,'fourcastnet','rmse_global_fcn_10m_u_component_of_wind_step_{}.nc').format(s)

    ds_sfno = xr.open_dataset(file_sfno)['rmse'].squeeze()
    ds_fcn = xr.open_dataset(file_fcn)['rmse'].squeeze()

    # sfno
    fig = plt.figure(figsize=(10, 5))
    ax = plt.axes(projection=ccrs.Robinson())
    ax.set_global()
    ds_sfno.plot.contourf(ax=ax, transform=ccrs.PlateCarree(),levels=list(range(0,60,6)))
    ax.coastlines()
    ax.set_title(np.datetime_as_string(ds_sfno.time.values,unit="h"))
    fig.savefig(os.path.join(savepath_sfno,str(idx)+".png"),dpi=300)

    # fcn
    fig = plt.figure(figsize=(10, 5))
    ax = plt.axes(projection=ccrs.Robinson())
    ax.set_global()
    ds_fcn.plot.contourf(ax=ax, transform=ccrs.PlateCarree(),levels=list(range(0,60,6)))
    ax.coastlines()
    ax.set_title(np.datetime_as_string(ds_fcn.time.values,unit="h"))
    fig.savefig(os.path.join(savepath_fcn,str(idx)+".png"),dpi=300)

    # sfno - globe
    fig = plt.figure(figsize=(10, 5))
    ax = plt.axes(projection=ccrs.Orthographic(20, 45))
    ax.set_global()
    ds_sfno.plot.contourf(ax=ax, transform=ccrs.PlateCarree(),levels=list(range(0,60,6)))
    ax.coastlines()
    ax.set_title(np.datetime_as_string(ds_sfno.time.values,unit="h"))
    fig.savefig(os.path.join(savepath_sfno_globe,str(idx)+".png"),dpi=300)

    # fcn - globe
    fig = plt.figure(figsize=(10, 5))
    ax = plt.axes(projection=ccrs.Orthographic(20, 45))
    ax.set_global()
    ds_fcn.plot.contourf(ax=ax, transform=ccrs.PlateCarree(),levels=list(range(0,60,6)))
    ax.coastlines()
    ax.set_title(np.datetime_as_string(ds_fcn.time.values,unit="h"))
    fig.savefig(os.path.join(savepath_fcn_globe,str(idx)+".png"),dpi=300)


def print_monitor():
    sys_dict = system_monitor(False,pids,names)
    sys_dict["time"] = str(datetime.now() - start_time)
    with open(monitor_savepath, 'a+') as fp: 
        json.dump(sys_dict, fp, indent=4, separators=(',',': '))

# for idx in range(end - 1):
#     plot(idx)

work = list(range(end - 1))
len_work = len(work)
results = []


start_time = datetime.now()
print('starting ' + str(sys.argv[1]) + ' processes')
print("main pid ",os.getpid())

with Pool(int(sys.argv[1])) as p:
    results.append(p.map_async(plot, work))
    print('Pool started : ', flush = True)
    a_childs = active_children()
    print(a_childs, flush = True)

    pids = [ child.pid for child in a_childs]
    pids.append(os.getpid())
    names = [ child.name for child in a_childs]
    names.append("main")

    print("looping monitor until completion", flush = True)
    while True:
        print_monitor()
        print("......................................................")
        print("len results ",len(results), flush = True)
        print([ar.ready() for ar in results])
        print(results)
        print("......................................................")
        if len(results) == len_work and all([ar.ready() for ar in results]):
            print('Pool done', flush = True)
            break
        sleep(60)