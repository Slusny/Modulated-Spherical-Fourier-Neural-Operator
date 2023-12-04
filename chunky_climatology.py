import xarray as xr
import os
import numpy as np
import sys
import json
from datetime import datetime
from S2S_on_SFNO.Models.provenance import system_monitor
from multiprocessing import Pool, active_children
from time import sleep

basePath = "/mnt/qb/goswami/data/era5"
savepath = "/mnt/qb/work2/goswami0/gkd965/climate/chunky"
monitor_savepath = os.path.join(savepath,"monitor"+datetime.now().strftime("%Y%m%d-%H%M")+".json")
ds = xr.open_mfdataset(os.path.join(basePath, 'single_pressure_level', '10m_v_component_of_wind', "10m_v_component_of_wind_????.nc"),parallel=True)

ds['hourofyear'] = xr.DataArray(ds.time.dt.strftime('%m-%d %H'), coords=ds.time.coords)

# for lat in range(ds.dimensions.latitude):
#     for long in range(ds.dimensions.longitude):
#         ds_lat_long = ds.isel(latitude=lat,longitude=long)
#         ds_hourofyear = ds_lat_long.groupby("hourofyear").mean()
#         savefile = os.path.join(savepath,f"10m_v_1959-2021_hourofyear_mean_{lat}-lat_{long}-long.nc")
#         ds_hourofyear.to_netcdf(savefile)

work = [(lat,long) for lat in range(ds.sizes["latitude"]) for long in range(ds.sizes["longitude"])]
len_work = len(work)

def calc_mean(lat,long):
    process = os.getpid()
    print(f"Process {process} works on {lat}-{long}", flush = True)
    ds_lat_long = ds.isel(latitude=lat,longitude=long)
    ds_hourofyear = ds_lat_long.groupby("hourofyear").mean()
    savefile = os.path.join(savepath,f"10m_v_1959-2021_hourofyear_mean_{lat}-lat_{long}-long.nc")
    ds_hourofyear.to_netcdf(savefile)
    return 1

def test_worker(lat, long):
    the_time = 3
    print("in active monitor", flush = True)
    print(active_children(), flush = True)
    pid = os.getpid()
    print(f"Processs {pid}\tWaiting {the_time} seconds")
    sleep(the_time)
    print(f"Process {pid}\tDONE")

def print_monitor():
    print("in active monitor", flush = True)
    print(active_children(), flush = True)
    pids = [ child.pid for child in active_children()]
    print(pids, flush = True)
    names = [ child.name() for child in active_children()]
    names.append("main")
    pids.append(os.getpid())
    sys_dict = system_monitor(False,pids,names)
    sys_dict["time"] = str(datetime.now() - start_time)
    if not os.path.isfile(monitor_savepath):
        with open(monitor_savepath, 'w+') as f: 
            json.dump([sys_dict], f, indent=4, separators=(',',': '))
    else:
        with open(monitor_savepath, 'w') as f: 
            listObj = json.load(f)
            listObj.append(sys_dict)
            json.dump(listObj, f, indent=4, separators=(',',': '))
	
if __name__ == '__main__':
    results = []
    start_time = datetime.now()
    print('starting ' + str(sys.argv[1]) + ' processes')
    print("main pid ",os.getpid())
    print("len work: ",len(work), flush = True)
    with Pool(int(sys.argv[1])) as p:
        results.append(p.map_async(test_worker, work))
        # p.close()
        # p.join()
        print('Pool started : ', flush = True)
        print("second pid ",os.getpid())
        print(active_children(), flush = True)
        print("looping monitor until compleation", flush = True)
        while True:
            print_monitor()
            print(results)
            if len(results) == len_work and all([ar.ready() for ar in results]):
                print('Pool done', flush = True)
                break
            sleep(60)
