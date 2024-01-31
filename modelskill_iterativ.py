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
cluster = True
variables = [
    ('10m_u_component_of_wind', '10u'),
    ('10m_v_component_of_wind','10v'),
    ('2m_temperature','2t'),
    ('total_column_water_vapour','tcwv')
]
variable = variables[variable_index][0]
dataset_var = variables[variable_index][1]

save_interval = 6
if cluster:
    save_path = "/mnt/qb/work2/goswami0/gkd965/climate/skillscores"
    basePath = "/mnt/qb/work2/goswami0/gkd965/"
    dataPath = os.path.join("/mnt/qb/goswami/data/era5","single_pressure_level",variable,variable+"_{}.nc")
else:
    save_path = "/mnt/V/Master/climate/skillscores"
    basePath = "/mnt/V/Master"
    dataPath = os.path.join("/mnt/V/Master/data",variable,variable+"_{}.nc") # no single_pressure_level dir on local machine

mean_files = {
    '10m_u_component_of_wind':'hourofyear_mean_for_10m_u_component_of_wind_from_1979_to_2017created_20240123-0404.nc',
    '10m_v_component_of_wind':'hourofyear_mean_for_10m_v_component_of_wind_from_1979_to_2019created_20231211-1339.nc',
    '2':'hourofyear_mean_for_2m_temperature_from_1979_to_2017created_20240123-0343.nc',
    'total_column_water_vapour':'hourofyear_mean_for_total_column_water_vapour_from_1979_to_2017created_20240123-0415.nc'

}

date_string = datetime.now().strftime("%Y%m%d-%H%M")
mean_file = os.path.join(basePath,"climate",mean_files[variable])
model_file_sfno = os.path.join(basePath,'outputs/sfno','leadtime_8760_startDate_201901010_createdOn_20240123T0337/leadtime_8760_startDate_201901010_createdOn_20240123T0337_step_{}.nc')
model_file_fcn = os.path.join(basePath,'outputs/fourcastnet','leadtime_8760_startDate_201901010_createdOn_20240123T0408/leadtime_8760_startDate_201901010_createdOn_20240123T0408_step_{}.nc')
save_file = os.path.join(basePath,'')


ds_ref_alltimes  = xr.open_dataset(mean_file,chunks={'time':1})#.to_array().squeeze()[:min_step*6:6]
g_truth = xr.open_dataset(dataPath.format(year))#.to_array().squeeze()[:min_step*6:6]

end = 128

num_nans = {"sfno":[],"fcn":[]}
skill_scores = {"sfno":[],"fcn":[]}

for idx in range(end):
    s = (idx+1)*6
    ds_sfno = xr.open_dataset(model_file_sfno.format(s))[dataset_var] # needs 4 min
    ds_fcn = xr.open_mfdataset(model_file_fcn.format(s))[dataset_var] 
    ds_ref = ds_ref_alltimes.sel(time=idx).to_array().squeeze()
    truth = g_truth.sel(time=np.datetime64(str(year)+'-01-01T00:00:00.000000000') + np.timedelta64(s, 'h')).to_array().squeeze()


    nans_sfno = np.isnan(ds_sfno)
    nans_fcn  = np.isnan(ds_fcn)
    num_nans["sfno"].append(nans_sfno.sum())
    num_nans["fcn"].append(nans_fcn.sum())

    rmse_sfno = xs.rmse(ds_sfno,truth,dim=["latitude","longitude"],skipna=True)
    rmse_fcn  = xs.rmse(ds_fcn ,truth,dim=["latitude","longitude"],skipna=True)
    rmse_ref  = xs.rmse(ds_ref,truth,dim=["latitude","longitude"])

    skill_score_sfno = 1 - rmse_sfno/rmse_ref
    skill_score_fcn  = 1 - rmse_fcn/rmse_ref
    skill_scores["sfno"].append(skill_score_sfno)
    skill_scores["fcn"].append(skill_score_fcn)

    rmse_sfno_globe = xs.rmse(ds_sfno,truth,dim=[],skipna=True)
    rmse_fcn_globe  = xs.rmse(ds_fcn ,truth,dim=[],skipna=True)

    rmse_sfno_globe.save(os.path.join(save_path,"rmse_global_sfno_"+variable+"_hr_"+str(s)+"_"+date_string+".nc"))
    rmse_fcn_globe.save(os.path.join(save_path,"rmse_global_fcn_"+variable+"_hr_"+str(s)+"_"+date_string+".nc"))

    if idx%save_interval == 0:
        print("saving skill scores to "+save_path,flush=True)
        np.save(os.path.join(save_path,"rmse_sfno_"+variable+"_"+date_string+".npz"),skill_scores['sfno'])
        np.save(os.path.join(save_path,"rmse_fcn_"+variable+"_"+date_string+".npz"),skill_scores['fcn'])
        np.save(os.path.join(save_path,"nans_sfno_"+variable+"_"+date_string+".npz"),num_nans['sfno'])
        np.save(os.path.join(save_path,"nans_fcn_"+variable+"_"+date_string+".npz"),num_nans['fcn'])

np.save(os.path.join(save_path,"rmse_sfno_"+variable+"_"+date_string+"_fin.npz"),skill_scores['sfno'])
np.save(os.path.join(save_path,"rmse_fcn_"+variable+"_"+date_string+"_fin.npz"),skill_scores['fcn'])
np.save(os.path.join(save_path,"nans_sfno_"+variable+"_"+date_string+"_fin.npz"),num_nans['sfno'])
np.save(os.path.join(save_path,"nans_fcn_"+variable+"_"+date_string+"_fin.npz"),num_nans['fcn'])

