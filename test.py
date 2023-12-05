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


basePath = "/mnt/qb/goswami/data/era5"
savepath = "/mnt/qb/work2/goswami0/gkd965/climate/chunky"
monitor_savepath = os.path.join(savepath,"monitor"+datetime.now().strftime("%Y%m%d-%H%M")+".json")

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

stats = system_monitor(True,[os.getpid()],["main"])

ds = xr.open_dataset(os.path.join(basePath, 'single_pressure_level', '10m_v_component_of_wind', "10m_v_component_of_wind_1990.nc"))

print(ds.info())

stats = system_monitor(True,[os.getpid()],["main"])