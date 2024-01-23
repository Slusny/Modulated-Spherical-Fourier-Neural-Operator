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

variable_index = 0
variables = [
    '10m_u_component_of_wind',
    '10m_v_component_of_wind',
    '2m_temperature',
]
variable = variables[variable_index]

basePath = "/mnt/qb/work2/goswami0/gkd965/"
mean_files = {
    '10m_u_component_of_wind':'hourofyear_mean_for_10m_u_component_of_wind_from_1979_to_2017created_20240123-0404.nc',
    '10m_v_component_of_wind':'hourofyear_mean_for_10m_v_component_of_wind_from_1979_to_2019created_20231211-1339.nc',
    '2m_temperature':'hourofyear_mean_for_2m_temperature_from_1979_to_2017created_20240123-0343.nc',
    'total_column_water_vapour':'hourofyear_mean_for_total_column_water_vapour_from_1979_to_2017created_20240123-0415.nc'

}
mean_file = os.path.join(basePath,"climate",mean_files[variable])
model_file_sfno = os.path.join(basePath,'outputs/sfno','leadtime_8760_20231213-1809.grib')
model_file_fcn = os.path.join(basePath,'outputs/fourcastnet','leadtime_8760_startDate_202101010_createdOn20231214-1204.grib')
save_file = os.path.join(basePath,'')
dataPath = os.path.join("/mnt/qb/goswami/data/era5","single_pressure_level",variable,variable+"_{}.nc")

ds_sfno = xr.open_dataset(model_file_sfno)[variable].to_array().squeeze()
ds_fcn = xr.open_dataset(model_file_sfno)[variable].to_array().squeeze()
ds_ref  = xr.open_dataset(mean_file).to_array().squeeze()[:ds_sfno.dims["step"]*6:6]
g_truth = xr.open_dataset(dataPath.format(2021)).to_array().squeeze()[:ds_sfno.dims["step"]*6:6]


skill_score_sfno = 1 - xs.rmse(ds_model,ds_ref,dim="time")/xs.rmse(g_truth,ds_ref,dim="time")

skill_score_mean = skill_score.mean(dim=["latitude","longitude"]).to_pandas().to_numpy()
