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


class IterMean():
    def __init__(self, ds):
        self.iter = 1
        self.mean = ds
    def __add__(self,ds2):
        self.iter += 1
        self.mean = self.mean + (1/self.iter+1)*(ds2 - self.mean)
        print("mean shape: ",self.mean.shape)
        # del ds2
    def get(self):
        return self.mean
    def save(self,savepath):
        if type(self.mean) == np.ndarray:
            xr.DataArray(self.mean,dims=["longitude","latitude","time"],name="v10").to_netcdf(savepath) 
        else:
            self.mean.to_netcdf(savepath)


def calc_mean(variable_path,year_range,savepath):
    if year in range(1948,2025,4):
        print("please don't start with a leap year")
        exit(0)
    mean = IterMean(xr.open_dataset(variable_path.format(year_range[0])).assign_coords(time=list(range(0,8760))).to_array().squeeze()) # numpy / xarray
    for year in range(year_range[0]+1,year_range[1]):
        print("--------------------------")
        print(year)
        data = xr.open_dataset(variable_path.format(year))
        timesteps = data.dims["time"]   
        if year in range(1948,2025,4):
            print("leap year")
            if (timesteps != 8784): 
                print("ERROR: v10_1.dims.time != 8760")
                continue
            data  = data.drop_isel(time=list(range((31+28)*24,(31+29)*24)))
        if (timesteps != 8760): 
            print("ERROR: v10_1.dims.time != 8760")
            continue

        # only needed if one wants to work with xarrays not numpy arrays
        data = data.assign_coords(time=list(range(0,8760)))

        # calculate mean
        mean + data.to_array().squeeze() # numpy / xarray
        stats = system_monitor(True,[os.getpid()],["main"])
    mean.save(savepath)


start_time = time()
calc_mean(os.path.join(basePath, 'single_pressure_level', '10m_v_component_of_wind', "10m_v_component_of_wind_{}.nc"),
          [1990,1994],
            "/mnt/qb/work2/goswami0/gkd965/climate/mean_for_loop_xarray_4years.nc"
          )

end_time = time()
print("time calc mean: " ,end_time - start_time)