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
import matplotlib.pyplot as plt

year = 2019
variable_index = 0
variables = [
    ('10m_u_component_of_wind', '10u'),
    ('10m_v_component_of_wind','10v'),
    ('2m_temperature','2t'),
    ('total_column_water_vapour','tcwv')
]
variable = variables[variable_index][0]
dataset_var = variables[variable_index][1]

basePath = "/mnt/qb/work2/goswami0/gkd965/"
mean_files = {
    '10m_u_component_of_wind':'hourofyear_mean_for_10m_u_component_of_wind_from_1979_to_2017created_20240123-0404.nc',
    '10m_v_component_of_wind':'hourofyear_mean_for_10m_v_component_of_wind_from_1979_to_2019created_20231211-1339.nc',
    '2m_temperature':'hourofyear_mean_for_2m_temperature_from_1979_to_2017created_20240123-0343.nc',
    'total_column_water_vapour':'hourofyear_mean_for_total_column_water_vapour_from_1979_to_2017created_20240123-0415.nc'

}
# only single pressure level
mean_file = os.path.join(basePath,"climate",mean_files[variable])
model_file_sfno = os.path.join(basePath,'outputs/sfno','leadtime_8760_startDate_201901010_createdOn_20240123T0337/*')
model_file_fcn = os.path.join(basePath,'outputs/fourcastnet','leadtime_8760_startDate_201901010_createdOn_20240123T0408/*')
save_file = os.path.join(basePath,'')
dataPath = os.path.join("/mnt/qb/goswami/data/era5","single_pressure_level",variable,variable+"_{}.nc")

ds_sfno = xr.open_mfdataset(model_file_sfno)[dataset_var] # needs 4 min
ds_fcn = xr.open_mfdataset(model_file_fcn)[dataset_var]    # needs 22 mins ,chunks={'step':1}

min_step = np.min([ds_sfno.shape[ds_sfno.dims.index('step')],ds_fcn.shape[ds_fcn.dims.index('step')]])

ds_ref  = xr.open_dataset(mean_file,chunks={'time':1}).to_array().squeeze()[:min_step*6:6]
g_truth = xr.open_dataset(dataPath.format(year)).to_array().squeeze()[:min_step*6:6]

rmse_sfno = xs.rmse(ds_sfno,g_truth,dim=["latitude","longitude"])
rmse_fcn  = xs.rmse(ds_fcn ,g_truth,dim=["latitude","longitude"])
rmse_ref  = xs.rmse(ds_ref,g_truth,dim=["latitude","longitude"])

skill_score_sfno = 1 - rmse_sfno/rmse_ref
skill_score_fcn  = 1 - rmse_fcn/rmse_ref

plt.figure()
plt.plot(skill_score_sfno)
plt.show()

# skill_score_mean = xs.mean(dim=["latitude","longitude"]).to_pandas().to_numpy()
