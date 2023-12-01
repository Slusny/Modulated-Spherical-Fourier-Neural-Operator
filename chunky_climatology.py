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

ds = xr.open_mfdataset(os.path.join(basePath, 'single_pressure_level', '10m_v_component_of_wind', "10m_v_component_of_wind_????.nc"),parallel=True)

ds['hourofyear'] = xr.DataArray(ds.time.dt.strftime('%m-%d %H'), coords=ds.time.coords)

# for lat in range(ds.dimensions.latitude):
#     for long in range(ds.dimensions.longitude):
#         ds_lat_long = ds.isel(latitude=lat,longitude=long)
#         ds_hourofyear = ds_lat_long.groupby("hourofyear").mean()
#         savefile = os.path.join(savepath,f"10m_v_1959-2021_hourofyear_mean_{lat}-lat_{long}-long.nc")
#         ds_hourofyear.to_netcdf(savefile)

work = [(lat,long) for lat in range(ds.sizes["latitude"]) for long in range(ds.sizes["longitude"])]

def calc_mean(lat,long):
    process = os.getpid()
    print(f"Process {process} works on {lat}-{long}")
    ds_lat_long = ds.isel(latitude=lat,longitude=long)
    ds_hourofyear = ds_lat_long.groupby("hourofyear").mean()
    savefile = os.path.join(savepath,f"10m_v_1959-2021_hourofyear_mean_{lat}-lat_{long}-long.nc")
    ds_hourofyear.to_netcdf(savefile)
    return 1

def print_monitor():
    pids = [ child.pid for child in active_children()]
    names = [ child.name() for child in active_children()]
    names.append("main")
    pids.append(os.getpid())
    sys_dict = system_monitor(False,pids,names)
    sys_dict["time"] = str(datetime.now() - start_time)
    with open(os.path.join(savepath,"monitor"+datetime.now().strftime("%Y%m%d-%H%M")+".json"), 'a+') as f: 
        json.dump(sys_dict, f)
	
if __name__ == '__main__':
    results = []
    start_time = datetime.now()
    with Pool(int(sys.argv[1])) as p:
        results.append(p.map_async(calc_mean, work))
    
    while True:
        print_monitor()
        if all([ar.ready() for ar in results]):
            print('Pool done')
            break
        sleep(60)
