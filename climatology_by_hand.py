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

work = [(lat,long) for lat in range(ds.sizes["latitude"]) for long in range(ds.sizes["longitude"])]

for lat, long in work:
    for year in range(1959,2022):
        for month in range(1,13):
            for day in range(1,32):
                if day == 31 and month in [2,4,6,9,11]:
                    continue
                if day == 30 and month == 2:
                    continue
                if day == 29 and month == 2 and year not in range(1948,2025,4):
                    continue
                for hour in range(0,24,6):
                    ds_lat_long = ds.sel(latitude=lat,longitude=long,time=datetime(year,month,day,hour,0))
                    ds_lat_long.info()
                    break
                break
            break
        break
    break

