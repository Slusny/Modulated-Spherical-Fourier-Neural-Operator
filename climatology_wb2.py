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

# Edit this
years = list(range(1979,2019)) # 
variable_index = 6
multi_pressure_level = 1000 #1000 # the pressure level in hPa (e.g. 850)





variables = [
    # single_pressure_level
    'total_column_water_vapour',    #0
    'u_component_of_wind',          #1
    'v_component_of_wind',          #2
    '2m_temperature',               #3 
    'mean_sea_level_pressure',      #4
    "surface_pressure",             #5
    # multi_pressure_level
    'relative_humidity',            #6
    'geopotential',                 #7
    'temperature',                  #8

]
variable = variables[variable_index]
# variable = '10m_v_component_of_wind'

basePath = "/mnt/qb/goswami/data/era5/weatherbench2"
wb_file = "1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"
wb = xr.open_zarr(os.path.join(basePath,wb_file))

# # get indices for the same hour of year across dataset
# days = wb.groupby("time.dayofyear").groups
# for k,v in days.items():
#     hr_sorted = np.array(v).reshape(-1,4)
#     for hr in range(4):
#         time_idx = hr_sorted[:,hr]

saveBasePath = "/mnt/qb/work2/goswami0/gkd965/climate"
if multi_pressure_level:
    ERA5_subdir = "multi_pressure_level"
    # file_paths = os.path.join(basePath, ERA5_subdir, variable, str(multi_pressure_level), variable+"_{}_"+str(multi_pressure_level)+".nc")
    saveFileName = "hourofyear_mean_for_"+variable+str(multi_pressure_level)+"_from_"+str(years[0])+"_to_"+str(years[-1])+"created_"+datetime.now().strftime("%Y%m%d-%H%M")+".nc"
else:
    ERA5_subdir = 'single_pressure_level'
    # file_paths = os.path.join(basePath, ERA5_subdir, variable, variable+"_{}.nc")
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


def calc_mean(variable,level,years,savepath):
    if years[0] in range(1948,2025,4):
        print("please don't start with a leap year")
        exit(0)
    mean = IterMean(wb.sel(time=slice(str(years[0])+'-01-01', str(years[0])+'-12-31'),level=level)[variable].load())
    idx = 0
    for year in years[1:]:
        print("--------------------------")
        print(year)
        data = wb.sel(time=slice(str(year)+'-01-01', str(year)+'-12-31'),level=level)[variable]
        if year in range(1948,2025,4):
            print("leap year")
            if (data.time.size != 1464): 
                print("ERROR: leapyear timesteps != 8784")
                sys.exit(0)
            data = data.drop_isel(time=[236,237,238,239])
        if (data.time.size != 1460): 
            print("ERROR: timesteps per year != 8760")
            sys.exit(0)

        # calculate mean
        mean + data.load()
        stats = system_monitor(True,[os.getpid()],["main"])
        idx += 1
        if idx % save_interval == 0:
            mean.save(savepath)
    # mean.get().compute()
    mean.save(savepath)

print("using xarray")
print("variable: ",variable)
start_time = time()
calc_mean(variable, multi_pressure_level, years, savepath)
end_time = time()
print("time calc mean: " ,end_time - start_time)