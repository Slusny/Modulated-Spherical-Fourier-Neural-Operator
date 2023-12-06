import xarray as xr
import os
import numpy as np
import sys
import json
from datetime import datetime
from S2S_on_SFNO.Models.provenance import system_monitor
from multiprocessing import Pool, active_children
from time import sleep, time
from test_numpy import IterMean
from multiprocessing import Process, Manager
from multiprocessing.managers import BaseManager

years = list(range(1990,1993))
variable = '10m_v_component_of_wind'
basePath = "/mnt/qb/goswami/data/era5"
saveBasePath = "/mnt/qb/work2/goswami0/gkd965/climate"
saveFileName = "mean_parallel_for_"+variable+"_from_"+str(years[0])+"_to_"+str(years[-1])+"created_"+datetime.now().strftime("%Y%m%d-%H%M")+".nc"
savepath = os.path.join(saveBasePath,saveFileName)
file_paths = os.path.join(basePath, 'single_pressure_level', variable, "10m_v_component_of_wind_{}.nc")
monitor_savepath = os.path.join(savepath,"monitor_parllel_"+datetime.now().strftime("%Y%m%d-%H%M")+".json")


# def calc_mean(coords):
#     lat, long = coords
#     process = os.getpid()
#     print(f"Process {process} works on {lat}-{long}", flush = True)
#     ds_lat_long = ds.isel(latitude=lat,longitude=long)
#     ds_hourofyear = ds_lat_long.groupby("hourofyear").mean()
#     savefile = os.path.join(savepath,f"testslice_10m_v_1959-2021_hourofyear_mean_{lat}-lat_{long}-long.nc")
#     ds_hourofyear.to_netcdf(savefile)
#     return 1

def calc_mean(year):
    print("--------------------------",flush=True)
    print(year,flush=True)
    data = xr.open_dataset(file_paths.format(year)) 
    if year in range(1948,2025,4):
        print("leap year",flush=True)
        print("timesteps: ",data.dims["time"]  ,flush=True)
        if (data.dims["time"]   != 8784): 
            print("ERROR: v10_1.dims.time != 8784",flush=True)
            return
        data  = data.drop_isel(time=list(range((31+28)*24,(31+29)*24)))
    if (data.dims["time"]   != 8760): 
        print("ERROR: v10_1.dims.time != 8760",flush=True)
        return

    # calculate mean
    mean + data.to_array().squeeze().assign_coords(time=list(range(0,8760))) # numpy / xarray

def print_monitor():
    sys_dict = system_monitor(False,pids,names)
    sys_dict["time"] = str(datetime.now() - start_time)
    with open(monitor_savepath, 'a+') as fp: 
        json.dump(sys_dict, fp, indent=4, separators=(',',': '))
	
if __name__ == '__main__':
    results = []
    start_time = datetime.now()
    print('starting ' + str(sys.argv[1]) + ' processes')
    print("main pid ",os.getpid())

    # Multiprocessing Manager
    BaseManager.register('IterMean', IterMean)
    manager = BaseManager()
    manager.start()
    mean = manager.IterMean(xr.open_dataset(file_paths.format(years[0])).to_array().squeeze().assign_coords(time=list(range(0,8760))))

    work = years[1:]
    len_work = len(work)

    print("len work: ",len_work, flush = True)
    with Pool(int(sys.argv[1])) as p:
        results.append(p.map_async(calc_mean, work))
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
            if len(results) == len_work and all([ar.ready() for ar in results]):
                print('Pool done', flush = True)
                break
            sleep(60)

    # with Pool(int(sys.argv[1])) as p:
    #     results.append(p.map(calc_mean, work))
    mean.save(savepath)
    end_time = time()
    print("time calc mean: " ,end_time - start_time)
