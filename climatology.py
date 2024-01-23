import xarray as xr
import os
import numpy as np
import sys
import json
from datetime import datetime
from S2S_on_SFNO.Models.provenance import system_monitor
from multiprocessing import Pool, active_children
from time import sleep
import psutil
import gc
from time import time



# Data available in cluster:
# 10m Wind                      : 1879 - 2021
# 2mt                           : 1959 - 2021
# sp (surface pressure)         : 1979 - 2019
# msl (mean sea level pressure) : 1959 - 2022     
# tcwv                          : 1959 - 2022  
# t                             : 1979 - 2020 (for 150, else 1959 - 2021)
# u/v                           : 1979 - 2019 (for 150)                                 
# z                             : 1979 - 2019 (for 150, else 1959 - 2021) 
#    -50                        : 1959 - 2022            
# r                             : 1979 - 2020 (for 150)                          


# missng: 
#   - 100u 100v, 
#   - t/u/v/z/r in 925
#   - t,r, in 50
#
#
#

# The combined year data set (e.g. 2m_temperature_sfc_1959_2021.nc) are daily means)
# The yearly data is hourly data (e.g. 2m_temperature_2021.nc)

years = list(range(1979,2019)) # 
variable_index = 0
multi_pressure_level = None # the pressure level in hPa (e.g. 850)
variables = [
    # single_pressure_level
    'total_column_water_vapour',
    'u_component_of_wind',
    'v_component_of_wind',
    # multi_pressure_level
    'relative_humidity',
    'geopotential',
    'temperature',

]
variable = variables[variable_index]
# variable = '10m_v_component_of_wind'
basePath = "/mnt/qb/goswami/data/era5"
saveBasePath = "/mnt/qb/work2/goswami0/gkd965/climate"
if multi_pressure_level:
    ERA5_subdir = "multi_pressure_level"
    file_paths = os.path.join(basePath, ERA5_subdir, variable, str(multi_pressure_level), variable+"_{}_"+str(multi_pressure_level)+".nc")
else:
    ERA5_subdir = 'single_pressure_level'
    file_paths = os.path.join(basePath, ERA5_subdir, variable, variable+"_{}.nc")

saveFileName = "hourofyear_mean_for_"+variable+"_from_"+str(years[0])+"_to_"+str(years[-1])+"created_"+datetime.now().strftime("%Y%m%d-%H%M")+".nc"
savepath = os.path.join(saveBasePath,saveFileName)
save_interval = 4


class IterMean():
    def __init__(self, ds):
        self.iter = 1
        self.mean = ds
    def __add__(self,ds2):
        self.iter += 1
        self.mean = self.mean + (1/self.iter)*(ds2 - self.mean)
        # del ds2
    def get(self):
        return self.mean
    def save(self,savepath):
        print("saving to "+savepath,flush=True)
        if type(self.mean) == np.ndarray:
            xr.DataArray(self.mean,dims=["longitude","latitude","time"],name="v10").to_netcdf(savepath) 
        else:
            self.mean.to_netcdf(savepath)


def calc_mean(variable_path,years,savepath):
    if years[0] in range(1948,2025,4):
        print("please don't start with a leap year")
        exit(0)
    mean = IterMean(xr.open_dataset(variable_path.format(years[0])).to_array().squeeze().assign_coords(time=list(range(0,8760))))#.to_numpy()) # numpy / xarray
    idx = 0
    for year in years[1:]:
        print("--------------------------")
        print(year)
        data = xr.open_dataset(variable_path.format(year))
        if year in range(1948,2025,4):
            print("leap year")
            print("timesteps: ",data.dims["time"])
            if (data.dims["time"] != 8784): 
                print("ERROR: leapyear timesteps != 8784")
                continue
            data  = data.drop_isel(time=list(range((31+28)*24,(31+29)*24)))
        if (data.dims["time"] != 8760): 
            print("ERROR: timesteps per year != 8760")
            continue

        # calculate mean
        mean + data.to_array().squeeze().assign_coords(time=list(range(0,8760)))#.to_numpy() # numpy / xarray
        stats = system_monitor(True,[os.getpid()],["main"])
        idx += 1
        if idx % save_interval == 0:
            mean.save(savepath)
    mean.save(savepath)

print("using xarray")
start_time = time()
calc_mean(file_paths, years, savepath)
end_time = time()
print("time calc mean: " ,end_time - start_time)