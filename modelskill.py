import xarray as xr
import xskillscore as xs
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

variable = '2m_temperature'
basePath = "/mnt/qb/work2/goswami0/gkd965/"
mean_file = os.path.join(basePath,"climate","hourofyear_mean_for_"+variable+"_from_1979_to_2019created_20231206-1713.nc")
model_file_sfno = os.path.join(basePath,'outputs','v10_SFNO_HalfYearForecast_startfrom_2021-01-01.nc')
model_file_fcn = os.path.join(basePath,'outputs','v10_FCN_HalfYearForecast_startfrom_2021-01-01.nc')
save_file = os.path.join(basePath,'')
dataPath = os.path.join("/mnt/qb/goswami/data/era5","single_pressure_level",variable,variable+"_{}.nc")

ds_sfno = xr.open_dataset(model_file_sfno).to_array().squeeze()
ds_fcn = xr.open_dataset(model_file_sfno).to_array().squeeze()
ds_ref  = xr.open_dataset(mean_file).to_array().squeeze()[:ds_sfno.dims["step"]*6:6]
g_truth = xr.open_dataset(dataPath.format(2021)).to_array().squeeze()[:ds_sfno.dims["step"]*6:6]


skill_score_sfno = 1 - xs.rmse(ds_model,ds_ref,dim="time")/xs.rmse(g_truth,ds_ref,dim="time")

skill_score_mean = skill_score.mean(dim=["latitude","longitude"]).to_pandas().to_numpy()
