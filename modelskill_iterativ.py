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
# import matplotlib.pyplot as plt

##


year = 2019
variable_index = 0
cluster = False
variables = [
    ('10m_u_component_of_wind', '10u'),
    ('10m_v_component_of_wind','10v'),
    ('2m_temperature','2t'),
    ('total_column_water_vapour','tcwv'),
    ('z1000','z1000')
]
variable = variables[variable_index][0]
dataset_var = variables[variable_index][1]

save_interval = 100
end = 440
start = 0

print("-------------------------")
print("Variable: ",variable)
print("-------------------------")

if cluster:
    save_path = os.path.join("/mnt/qb/work2/goswami0/gkd965/climate/skillscores",variable)
    basePath = "/mnt/qb/work2/goswami0/gkd965/"
    dataPath = os.path.join("/mnt/qb/goswami/data/era5","single_pressure_level",variable,variable+"_{}.nc")
    model_file_sfno = os.path.join(basePath,'outputs/sfno','leadtime_8760_startDate_201901010_createdOn_20240123T0337/leadtime_8760_startDate_201901010_createdOn_20240123T0337_step_{}.nc')
    model_file_fcn = os.path.join(basePath,'outputs/fourcastnet','leadtime_8760_startDate_201901010_createdOn_20240123T0408/leadtime_8760_startDate_201901010_createdOn_20240123T0408_step_{}.nc')
else:
    save_path = os.path.join("/mnt/V/Master/climate/skillscores",variable)
    basePath = "/mnt/V/Master"
    basePath2 = "/mnt/ssd2/Master/S2S_on_SFNO"
    dataPath = os.path.join("/mnt/V/Master/data",variable,variable+"_{}.nc") # no single_pressure_level dir on local machine
    model_file_sfno = os.path.join(basePath2,'outputs/sfno','leadtime_8760_startDate_201901010_createdOn_20240129T2243/leadtime_8760_startDate_201901010_createdOn_20240129T2243_step_{}.nc')
    model_file_fcn = os.path.join(basePath2,'outputs/fourcastnet','leadtime_8760_startDate_201901010_createdOn_20240129T2244/leadtime_8760_startDate_201901010_createdOn_20240129T2244_step_{}.nc')

mean_files = {
    '10m_u_component_of_wind':'hourofyear_mean_for_10m_u_component_of_wind_from_1979_to_2017created_20240123-0404.nc',
    '10m_v_component_of_wind':'hourofyear_mean_for_10m_v_component_of_wind_from_1979_to_2019created_20231211-1339.nc',
    '2m_temperature':'hourofyear_mean_for_2m_temperature_from_1979_to_2017created_20240123-0343.nc',
    'total_column_water_vapour':'hourofyear_mean_for_total_column_water_vapour_from_1979_to_2017created_20240123-0415.nc',
    'z1000':'hourofyear_mean_for_geopotential1000_from_1979_to_2018created_20240131-0011.nc'

}

date_string = datetime.now().strftime("%Y%m%d-%H%M")
mean_file = os.path.join(basePath,"climate",mean_files[variable])
save_file = os.path.join(basePath,'')


ds_ref_alltimes  = xr.open_dataset(mean_file,chunks={'time':1})#.to_array().squeeze()[:min_step*6:6]
g_truth = xr.open_dataset(dataPath.format(year))#.to_array().squeeze()[:min_step*6:6]


num_nans = {"sfno":[],"fcn":[]}
rmse_ = {"sfno":[],"fcn":[],"ref":[]}


# savepath_sfno  = os.path.join(save_path,date_string,"sfno")
# savepath_fcn  = os.path.join(save_path,date_string,"fourcastnet")
savepath_sfno  = os.path.join("/mnt/V/Master/ai-models")
savepath_fcn  = os.path.join("/mnt/V/Master/ai-models")
savepath_numpy  = os.path.join(save_path,date_string)
if not os.path.exists(savepath_sfno): os.makedirs(savepath_sfno)
if not os.path.exists(savepath_fcn): os.makedirs(savepath_fcn)

# for ref pic
savepath_ref  = os.path.join(save_path,"ref",)

for idx in range(start,end - 1):
    s = (idx+1)*6
    path_sfno = model_file_sfno.format(s)
    print('loading '+path_sfno,flush=True)
    path_fcn = model_file_fcn.format(s)
    print('loading '+model_file_fcn,flush=True)
    ds_sfno = xr.open_dataset(path_sfno)[dataset_var] # needs 4 min
    ds_fcn = xr.open_mfdataset(path_fcn)[dataset_var] 
    ds_ref = ds_ref_alltimes.sel(time=idx).to_array().squeeze()
    truth = g_truth.sel(time=np.datetime64(str(year)+'-01-01T00:00:00.000000000') + np.timedelta64(s, 'h')).to_array().squeeze()


    nans_sfno = np.isnan(ds_sfno)
    nans_fcn  = np.isnan(ds_fcn)
    num_nans["sfno"].append(nans_sfno.sum())
    num_nans["fcn"].append(nans_fcn.sum())

    rmse_sfno = xs.rmse(ds_sfno,truth,dim=["latitude","longitude"],skipna=True)
    rmse_fcn  = xs.rmse(ds_fcn ,truth,dim=["latitude","longitude"],skipna=True)
    rmse_ref  = xs.rmse(ds_ref,truth,dim=["latitude","longitude"])

    # skill_score_sfno = 1 - rmse_sfno/rmse_ref
    # skill_score_fcn  = 1 - rmse_fcn/rmse_ref
    rmse_["sfno"].append(rmse_sfno)
    rmse_["fcn"].append(rmse_fcn)
    rmse_["ref"].append(rmse_ref)

    rmse_sfno_globe = xs.rmse(ds_sfno,truth,dim=[],skipna=True)
    rmse_fcn_globe  = xs.rmse(ds_fcn ,truth,dim=[],skipna=True)
    rmse_ref_globe  = xs.rmse(ds_ref ,truth,dim=[],skipna=True) # ref pic

    # rmse_sfno_globe.to_dataset(name="rmse").assign_coords(step=[s]).to_netcdf(os.path.join(savepath_sfno,"rmse_global_sfno_"+variable+"_step_"+str(s)+".nc"))
    # rmse_fcn_globe.to_dataset(name="rmse").assign_coords(step=[s]).to_netcdf(os.path.join(savepath_fcn,"rmse_global_fcn_"+variable+"_step_"+str(s)+".nc"))

    #
    # rmse_ref_globe.to_dataset(name="rmse").assign_coords(step=[s]).to_netcdf(os.path.join(savepath_ref,"rmse_global_ref_"+variable+"_step_"+str(s)+".nc")) # ref pic

    # if idx%save_interval == 0:
    #     print("saving skill scores to "+save_path,flush=True)
    #     np.save(os.path.join(savepath_sfno,"rmse_sfno_"+variable+"_"+date_string),rmse_['sfno'])
    #     np.save(os.path.join(savepath_fcn,"rmse_fcn_"+variable+"_"+date_string),rmse_['fcn'])
    #     np.save(os.path.join(savepath_sfno,"nans_sfno_"+variable+"_"+date_string),num_nans['sfno'])
    #     np.save(os.path.join(savepath_fcn,"nans_fcn_"+variable+"_"+date_string),num_nans['fcn'])


np.save(os.path.join(savepath_numpy,"rmse_sfno_"+variable+"_"+date_string+"_fin"),rmse_['sfno'])
np.save(os.path.join(savepath_numpy,"rmse_fcn_"+variable+"_"+date_string+"_fin"),rmse_['fcn'])
np.save(os.path.join(savepath_numpy,"rmse_ref_"+variable+"_"+date_string+"_fin"),rmse_['ref'])
np.save(os.path.join(savepath_numpy,"nans_sfno_"+variable+"_"+date_string+"_fin"),num_nans['sfno'])
np.save(os.path.join(savepath_numpy,"nans_fcn_"+variable+"_"+date_string+"_fin"),num_nans['fcn'])

