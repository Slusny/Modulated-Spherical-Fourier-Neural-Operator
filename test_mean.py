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
saveBasePath = "/mnt/qb/work2/goswami0/gkd965/climate"
saveFileName = "mean_for_loop_xarray_4years"
savepath = os.path.join(saveBasePath,saveFileName+datetime.now().strftime("%Y%m%d-%H%M")+".nc")
file_paths = os.path.join(basePath, 'single_pressure_level', '10m_v_component_of_wind', "10m_v_component_of_wind_{}.nc")
year_range = [1990,1994]

coords = [
    {"latitude":10,"longitude":10,"time":10},
    {"latitude":100,"longitude":100,"time":100},
    {"latitude":500,"longitude":500,"time":500},
]

mean_list = [[]]*len(coords)
for year in range(year_range[0],year_range[1]):
    data = xr.open_dataset(file_paths.format(year))
    for i,coord in enumerate(coords):
        mean_list[i].append(data.isel(coord))
        

for i,coord in enumerate(coords):
    mean_list[i] = np.mean(mean_list[i],axis=0)

data = xr.open_dataset(file_paths.format(year))
data = xr.open_dataset(file_paths.format(year))