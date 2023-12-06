import xarray as xr
import os
import numpy as np
import sys
import json
from datetime import datetime
from S2S_on_SFNO.Models.provenance import system_monitor
from multiprocessing import Pool, active_children
from time import sleep


years = list(range(1990,1993))
variable = '10m_v_component_of_wind'
basePath = "/mnt/qb/goswami/data/era5"
saveBasePath = "/mnt/qb/work2/goswami0/gkd965/climate"
saveFileName = "mean_for_loop_xarray_4years20231206-1615.nc"
savepath = os.path.join(saveBasePath,saveFileName)
file_paths = os.path.join(basePath, 'single_pressure_level', variable, "10m_v_component_of_wind_{}.nc")


coords = [
    {"latitude":10,"longitude":10,"time":10},
    {"latitude":100,"longitude":100,"time":100},
    {"latitude":500,"longitude":500,"time":500},
]

mean_list = np.zeros((len(years),len(coords)))
for i,year in enumerate(years):
    data = xr.open_dataset(file_paths.format(year))
    for j,coord in enumerate(coords):
        mean_list[i,j] = data[coord].to_array().to_numpy()[0]
        
print(mean_list)
means = mean_list.mean(axis=0)

print("means")
for i,coord in enumerate(coords):
    print("--------")
    print("coords",coord)
    print("real: ",means[i])
    print("calc: ",xr.open_dataset(savepath)[coord].to_array().to_numpy()[0])