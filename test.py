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


basePath = "/mnt/qb/goswami/data/era5"

def system_monitor(printout=False,pids=[],names=[]):
    system = {}
    mem_percent = psutil.virtual_memory()[2]
    mem_used = psutil.virtual_memory()[3]/1000000000
    mem_total = psutil.virtual_memory()[1]/1000000000
    processes = []
    for i,pid in enumerate(pids):
        python_process = psutil.Process(pid)
        mem_proc = python_process.memory_percent()
        mem_proc_percent = python_process.memory_info()[0]/2.**30  # memory use in GB...I think
        processes.append({"pid":pid,"process":names[i],"memory percent":mem_proc,"memory used(GB)":mem_proc_percent})
    cores = os.cpu_count()
    cpu_percent = psutil.cpu_percent()
    loads = psutil.getloadavg()
    cpu_usage = [load/cores* 100 for load in loads]
    if printout:
        print("RAM:")
        print('    memory % used:', mem_percent)
        print('    Used (GB):', mem_used)
        print('    Total available (GB):', mem_total)
        # use memory_profiler  for line by line memory analysis, add @profile above function
        # Ram for certain process:
        print('    Process memory used (GB):', mem_proc)
        print("    Process memory used : ", mem_proc_percent)

        print("CPU:")
        print("    available cores: ",cores)
        print("    util: ",cpu_percent)
        print("    averge load over 1, 5 ,15 min: ",cpu_usage)
    
    system["memory percent"] =  mem_percent
    system["memory used(GB)"] =  mem_used
    system["memory total available (GB)"] =  mem_total
    system["processes"] =  processes
    system["cpu percent"] =  cpu_percent
    system["cpu usage"] =  cpu_usage
    system["cores"] =  cores

    return system


savepath = "/mnt/qb/work2/goswami0/gkd965/climate/testmean.nc"
stats = system_monitor(True,[os.getpid()],["main"])

# load
start_load = time()
v10_1 = xr.open_dataset(os.path.join(basePath, 'single_pressure_level', '10m_v_component_of_wind', "10m_v_component_of_wind_1990.nc")).to_array().squeeze()#.to_numpy()
if (v10_1.dims.time != 8760): print("ERROR: v10_1.dims.time != 8760")
print("v10 timesteps: ",v10_1.dims.time)
v10_1.assign_coords(time=list(range(0,8760)))
end_load = time()
print("time loading one: " ,end_load - start_load)
print("----------------")
print("v10 1 shape: ",v10_1.shape)

v10_2 = xr.open_dataset(os.path.join(basePath, 'single_pressure_level', '10m_v_component_of_wind', "10m_v_component_of_wind_1991.nc")).to_array().squeeze()#.to_numpy()
print("v10 2 shape: ",v10_2.shape)
if (v10_2.dims.time != 8760): print("ERROR: v10_2.dims.time != 8760")
print("v10 timesteps: ",v10_2.dims.time)
v10_2.assign_coords(time=list(range(0,8760)))

# print(xr.align(v10_1,v10_2, join='exact'))

print("stats after two years in RAM")
stats = system_monitor(True,[os.getpid()],["main"])

class IterMean():
    def __init__(self, ds):
        self.iter = 1
        self.mean = ds
    def __add__(self,ds2):
        self.iter += 1
        self.mean = self.mean + (1/self.iter+1)*(ds2 - self.mean)
        print("mean shape: ",self.mean.shape)
        del ds2
    def get(self):
        return self.mean
    def save(self,savepath):
        # xr.DataArray(self.mean,dims=["longitude","latitude","time"],name="v10").to_netcdf(savepath)
        self.mean.to_netcdf(savepath)

# mean
mean = IterMean(v10_1)
start_mean = time()
mean + v10_2
del v10_2
end_mean = time()
print("time calc mean: " ,end_mean - start_mean)
stats = system_monitor(True,[os.getpid()],["main"])

# save
start_save = time()
mean.save(savepath)
end_save = time()
print("time saving: " ,end_save - start_save)
stats = system_monitor(True,[os.getpid()],["main"])

print("number of references: ",len(gc.get_referrers(v10_2)))

def calc_mean(variable_path,year_range,savepath):
    mean = IterMean(variable_path.format(year_range[0]))
    for year in range(year_range[0]+1,year_range[1]):
        print("--------------------------")
        print(year)
        mean + xr.open_dataset(variable_path.format(year)).to_array()
        stats = system_monitor(True,[os.getpid()],["main"])
    mean.save(savepath)

# calc_mean(os.path.join(basePath, 'single_pressure_level', '10m_v_component_of_wind', "10m_v_component_of_wind_{}.nc"),
#           [1990,1993],
            # "/mnt/qb/work2/goswami0/gkd965/climate/mean_for_loop.nc"
#           )