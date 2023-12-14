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

work = [(lat,long) for lat in range(ds.sizes["latitude"]) for long in range(ds.sizes["longitude"])]
step = 5
work = [(slice(lat,lat+step),slice(long,long+step)) for lat in range(0,ds.sizes["latitude"],step) for long in range(0,ds.sizes["longitude"],step)]
len_work = len(work)

def calc_mean(coords):
    lat, long = coords
    process = os.getpid()
    print(f"Process {process} works on {lat}-{long}", flush = True)
    ds_lat_long = ds.isel(latitude=lat,longitude=long)
    ds_hourofyear = ds_lat_long.groupby("hourofyear").mean()
    savefile = os.path.join(savepath,f"testslice_10m_v_1959-2021_hourofyear_mean_{lat}-lat_{long}-long.nc")
    ds_hourofyear.to_netcdf(savefile)
    return 1

def test_worker(lat):
    print(lat)
    the_time = 3
    print("in active monitor", flush = True)
    pid = os.getpid()
    print(f"Processs {pid}\tWaiting {the_time} seconds")
    sleep(the_time)
    print(f"Process {pid}\tDONE")

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
    print("len work: ",len(work), flush = True)
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
            print(results)
            if len(results) == len_work and all([ar.ready() for ar in results]):
                print('Pool done', flush = True)
                break
            sleep(60)
