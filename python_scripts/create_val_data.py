import xarray as xr
import numpy as np

zarr_save_path = ''

obs = xr.open_dataset("/mnt/qb/goswami/data/era5/weatherbench2/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr")
param_sfc_ERA5 = ["10m_u_component_of_wind", "10m_v_component_of_wind", "2m_temperature", "surface_pressure", "mean_sea_level_pressure", "total_column_water_vapour","u_component_of_wind","v_component_of_wind","geopotential","temperature","relative_humidity"]
sfc = ["10m_u_component_of_wind", "10m_v_component_of_wind", "2m_temperature", "surface_pressure", "mean_sea_level_pressure", "total_column_water_vapour"]
sfc_idx = [0,1,4,5,6,7]
pl  = ["u_component_of_wind","v_component_of_wind","geopotential","temperature","relative_humidity"]
sfno_data = obs.sel(time=slice("01-01-2016","31-12-2017"))[param_sfc_ERA5]

means = np.load("/mnt/qb/work2/goswami0/gkd965/Assets/sfno/global_means.npy")
stds = np.load("/mnt/qb/work2/goswami0/gkd965/Assets/sfno/global_stds.npy")




wb_ordering_scf = {
    "10m_u_component_of_wind":0,
    "10m_v_component_of_wind":1,
    "2m_temperature":4,
    "surface_pressure":5,
    "mean_sea_level_pressure":6,
    "total_column_water_vapour":7
}
wb_ordering_pl = {
    "u_component_of_wind": np.arange(8,21),
    "v_component_of_wind":np.arange(21,34), 
    "geopotential":np.arange(34,47), 
    "temperature":np.arange(47,60), 
    "relative_humidity":np.arange(60,73),  
}
level = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
level.reverse()

data_dict={}
lat_coords = np.arange(-90,90.25,0.25)
lat_coords = lat_coords[::-1]
self.output_data = np.array(self.output_data)

# order dates if shuffled
if not self.cfg.no_shuffle:
    idx = sorted(range(len(self.time_dim)), key=lambda k: self.time_dim[k])
    self.time_dim = np.array(self.time_dim)[idx]
    self.output_data = self.output_data[:,idx]

for out_var,idx in wb_ordering_scf.items():
    # if out_var in self.output_variables:
    data_dict[out_var] = xr.DataArray(self.output_data[:,:,idx],
        dims=["prediction_timedelta","time","latitude","longitude"],
        coords=dict(
            prediction_timedelta=("prediction_timedelta",self.time_delta),
            time=("time",self.time_dim),
            latitude=("latitude", lat_coords),
            longitude=("longitude",np.arange(0,360,0.25)),
        )
    )

for out_var,idx in wb_ordering_pl.items():
    # if out_var in self.output_variables:
    data_dict[out_var] = xr.DataArray(self.output_data[:,:,idx],
        dims=["prediction_timedelta","time","level","latitude","longitude"],
        coords=dict(
            prediction_timedelta=("prediction_timedelta",self.time_delta),
            time=("time",self.time_dim),
            level=("level",level),
            latitude=("latitude", lat_coords),
            longitude=("longitude",np.arange(0,360,0.25)),
        )
    )

if self.cfg.no_shuffle:
    start_time = datetime.strptime(self.time_dim[0].astype(str)[:-3], '%Y-%m-%dT%H:%M:%S.%f').strftime("%d.%m.%Y")
    end_time = datetime.strptime(self.time_dim[-1].astype(str)[:-3], '%Y-%m-%dT%H:%M:%S.%f').strftime("%d.%m.%Y")
    time_str = 'time='+start_time+'-'+end_time
else:
    time_str = 'time='+str(self.cfg.validationset_start_year)+'-'+str(self.cfg.validationset_end_year)+'-shuffled'
zarr_save_path = os.path.join(self.cfg.path,'forecast_lead_time='+str(self.cfg.multi_step_validation)+"_"+time_str + '.zarr')
print("saving zarr to ",zarr_save_path,flush=True)
if iter ==0 :
    xr.Dataset(data_vars=data_dict).chunk({'time': 1, 'prediction_timedelta':1, 'latitude':721,'longitude':1440}).to_zarr(zarr_save_path)
else:
    xr.Dataset(data_vars=data_dict).chunk({'time': 1, 'prediction_timedelta':1, 'latitude':721,'longitude':1440}).to_zarr(zarr_save_path,mode="a",append_dim="time")

# clean up
self.output_data = [[]for _ in range(0,self.cfg.multi_step_validation+1,1+self.cfg.validation_step_skip)]
self.time_dim = []


























time = sfno_data.time.to_numpy()
level = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
level.reverse()
lat_coords = np.arange(-90,90.25,0.25)
lat_coords = lat_coords[::-1]

for idx,var in enumerate(sfc):
    data = sfno_data[var].to_array().to_numpy()
    data = (data - means[sfc_idx[idx]])/stds[sfc_idx[idx]]
    # save data as an xarray
    data_array = xr.DataArray(data,
            dims=["time","latitude","longitude"],
            coords=dict(
                time=("time",time),
                latitude=("latitude", lat_coords),
                longitude=("longitude",np.arange(0,360,0.25)),
            )
        )
    xr.Dataset(data_vars={var:data_array}).chunk({'time': 1, 'latitude':721,'longitude':1440}).to_zarr(zarr_save_path,mode="a")
    else:

        for out_var,idx in wb_ordering_pl.items():
            # if out_var in self.output_variables:
            data_dict[out_var] = xr.DataArray(self.output_data[:,:,idx],
                dims=["prediction_timedelta","time","level","latitude","longitude"],
                coords=dict(
                    prediction_timedelta=("prediction_timedelta",self.time_delta),
                    time=("time",self.time_dim),
                    level=("level",level),
                    latitude=("latitude", lat_coords),
                    longitude=("longitude",np.arange(0,360,0.25)),
                )
            )
        sfno = xr.DataArray(data, dims=["time","lat","lon"], coords={"time":sfno_data.time,"lat":sfno_data.lat,"lon":sfno_data.lon})