import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset#, IterableDataset
import torch.cuda.amp as amp
from calendar import isleap
import xarray as xr
import numpy as np
import os
import sys

class ERA5_galvani(Dataset):
    """
        Dataset for the ERA5 data on Galvani cluster at university of tuebingen.
        A time point in the dataset can be selected by a numerical index.
        The index is counted from the first year in param:total_dataset_year_range in 6hr increments (default steps_per_day=4),
        e.g. if the first year in the range is 1959, the 1.1.1959 0:00 is index 0, 1.1.1959 6:00 is index 1, etc.

        :param class    model       : reference to the model class the dataset is instanciated for (e.g. FourCastNetv2)
        :param str      path        : Path to the zarr weatherbench dataset on the Galvani cluster
        :param str      path_era5   : Path to the era5 dataset on the /mnt/qb/... , this is nessessary since u100/v100 are not available in the weatherbench dataset 
        :param int      start_year  : The year from which the training data should start
        :param int      end_year    : Years later than end_year won't be included in the training data
        :param int      steps_per_day: How many datapoints per day in the dataset. For a 6hr increment we have 4 steps per day.
        :param bool     sst         : wether to include sea surface temperature in the data as seperate tuple element (used for filmed models)
        :param bool     uv100       : wether to include u100 and v100 in the data 
        :param int      auto_regressive_steps : how many consecutive datapoints should be loaded to used to calculate an autoregressive loss 
        
        :return: weather data as torch.tensor or tuple (weather data, sea surface temperature)   
    """
    def __init__(
            self, 
            model,
            # path="/mnt/ceph/goswamicd/datasets/weatherbench2/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr",#weatherbench2/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr", #1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr", 
            #path="/mnt/ceph/goswamicd/datasets/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr",#weatherbench2/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr", #1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr", 
            path = "/mnt/qb/goswami/data/era5/weatherbench2/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr", # start date: 1959-01-01 end date : 2023-01-10T18:00
            # chunked
            # path = "/mnt/qb/goswami/data/era5/weatherbench2/1959-2022-wb13-6h-0p25deg-chunk-1.zarr-v2", # start date: 1959-01-01 end date : 2023-01-10T18:00 # not default, needs derived variables
            
            u100_path = "/mnt/qb/goswami/data/era5/u100m_v100m_721x1440/u100m_1959-2022_721x1440_correct_chunk_new_mean_INTERPOLATE.zarr",
            v100_path = "/mnt/qb/goswami/data/era5/u100m_v100m_721x1440/v100m_1959-2023-10_721x1440_correct_chunk_new_mean_INTERPOLATE.zarr",

            start_year=2000,
            end_year=2022,
            steps_per_day=4,
            sst=True,
            coarse_level=4,
            uv100=True,
            temporal_step=28,
            cls=None,
            auto_regressive_steps=0
        ):
        self.model = model
        self.sst = sst
        self.auto_regressive_steps = auto_regressive_steps
        self.coarse_level = coarse_level
        self.uv100 = uv100
        self.temporal_step = temporal_step
        if path.endswith(".zarr"):  self.dataset = xr.open_zarr(path,chunks=None)
        else:                       self.dataset = xr.open_dataset(path,chunks=None)
        if self.uv100:
            #qb
            # self.dataset_u100 = xr.open_mfdataset(os.path.join(path_era5+"100m_u_component_of_wind/100m_u_component_of_wind_????.nc"))
            # self.dataset_v100 = xr.open_mfdataset(os.path.join(path_era5+"100m_v_component_of_wind/100m_v_component_of_wind_????.nc"))
            #ceph
            # self.dataset_uv100 = xr.open_mfdataset("/mnt/ceph/goswamicd/datasets/weatherbench2/era5/1959-2023_01_10-u100mv100m-6h-1440x721"))
            # qb zarr
            
            if u100_path.endswith(".zarr"): self.dataset_u100 = xr.open_zarr(u100_path,chunks=None)
            else:                           self.dataset_u100 = xr.open_mfdataset(u100_path,chunks=None) # sd: 1959-01-01, end date : 2022-12-30T18
            if v100_path.endswith(".zarr"): self.dataset_v100 = xr.open_zarr(v100_path,chunks=None)
            else:                           self.dataset_v100 = xr.open_mfdataset(v100_path,chunks=None) # sd: 1959-01-01 end date: 2023-10-31

        if cls:
            self.cls = torch.from_numpy(np.load(cls))
        else:
            self.cls = None
        # check if the 100uv-datasets and era5 have same start and end date
        # Check if set Start date to be viable
        startdate = np.array([self.dataset.time[0].to_numpy() ,self.dataset_v100.time[0].to_numpy() ,self.dataset_u100.time[0].to_numpy()])
        possible_startdate = startdate.max()
        if not (startdate == startdate[0]).all(): 
            print("Start dates of all arrays need to be the same! Otherwise changes to the Dataset class are needed!")
            print("For ERA5, 100v, 100u the end dates are",startdate)
            sys.exit(0)
        # if int(np.datetime_as_string(possible_startdate,"M")) != 1 and int(np.datetime_as_string(possible_startdate,"D")) != 1 :
        #     print("Start dates need to be the 1/1 of a year! Otherwise changes to the Dataset class are needed!")
        #     print("For ERA5, 100v, 100u the end dates are",startdate)
        #     sys.exit(0)
        dataset_start = int(np.datetime_as_string(possible_startdate,"Y"))
        if start_year < int(np.datetime_as_string(possible_startdate,"Y")):
            print("chosen start year is earlier than the earliest common start date to all of the datasets")
            print("Start year is set to ",int(np.datetime_as_string(possible_startdate,"Y")))
            print("For ERA5, 100v, 100u the end dates are",startdate)
            start_year = dataset_start
        
        # Check if set Start date to be viable
        enddate = np.array([self.dataset.time[-1].to_numpy() ,self.dataset_v100.time[-1].to_numpy() ,self.dataset_u100.time[-1].to_numpy()])
        possible_enddate = enddate.min()
        if end_year > int(np.datetime_as_string(possible_enddate,"Y")):
            print("chosen end year is later than the latest common end date to all of the datasets")
            print("End year is set to ",int(np.datetime_as_string(possible_enddate,"Y")))
            print("For ERA5, 100v, 100u the end dates are",enddate)
            end_year = int(np.datetime_as_string(possible_enddate,"Y"))

        self.start_idx = steps_per_day * sum([366 if isleap(year) else 365 for year in list(range(dataset_start, start_year))])
        self.end_idx = steps_per_day * sum([366 if isleap(year) else 365 for year in list(range(dataset_start, end_year))]) -1

        print("Using years: ",start_year," - ", end_year," (total length: ",self.end_idx - self.start_idx,") (availabe date range: ",np.datetime_as_string(possible_startdate,"Y"),"-",np.datetime_as_string(possible_enddate,"Y"),")")
        print("")

    def __len__(self):
        return self.end_idx - self.start_idx
    
    def __getitem__(self, idx):
        level_list = self.model.param_level_pl[1].copy()
        level_list.reverse()

        def format(sample,sst=None):
            scf = sample[self.model.param_sfc_ERA5].to_array().to_numpy()
            pl = sample[list(self.model.levels_per_pl.keys())].sel(level=level_list).to_array().to_numpy()
            pl = pl.reshape((pl.shape[0]*pl.shape[1], pl.shape[2], pl.shape[3]))
            time = str(sample.time.to_numpy())
            time = torch.tensor(int(time[0:4]+time[5:7]+time[8:10]+time[11:13])) # time in format YYYYMMDDHH
            if self.uv100:
                # ERA5 data on QB
                u100_t = self.dataset_u100.isel(time=self.start_idx + idx)
                v100_t = self.dataset_v100.isel(time=self.start_idx + idx)
                if "expver" in set(u100_t.coords.dims): u100_t = u100_t.sel(expver=1)
                if "expver" in set(v100_t.coords.dims): v100_t = v100_t.sel(expver=1)
                u100 = u100_t["u100"].to_numpy()[None]
                v100 = v100_t["v100"].to_numpy()[None]
                data =  torch.from_numpy(np.vstack((
                    scf[:2],
                    u100,
                    v100,
                    scf[2:],
                    pl)))
            else: 
                data = torch.from_numpy(np.vstack((scf,pl))) # transpose to have the same shape as expected by SFNO (lat,long)
            
            return [data.float(),time]
        
        def get_sst(idx):
            # temporal step 0 for normal sst output
            slice_idx = self.temporal_step -1
            sst = self.dataset.isel(time=slice(self.start_idx + idx -slice_idx ,self.start_idx + idx +self.auto_regressive_steps+2))["sea_surface_temperature"]
            sst = sst.coarsen(latitude=self.coarse_level,longitude=self.coarse_level,boundary='trim').mean().to_numpy()
            return torch.from_numpy(sst)

        if self.sst:
            sst = get_sst(idx)

        data = []
        for i in range(0, self.auto_regressive_steps+2):
            era5 = format(self.dataset.isel(time=self.start_idx + idx + i))
            if self.sst:
                era5.insert(1,(sst[i:i+self.temporal_step]).float())
            elif self.cls is not None:
                era5.insert(1,(self.cls[idx+i]).float())
            data.append(era5)

        return data

class SST_galvani(Dataset):
    def __init__(
            self, 
            path = "/mnt/qb/goswami/data/era5/weatherbench2/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr", # start date: 1959-01-01 end date : 2023-01-10T18:00
            start_year=2000,
            end_year=2022,
            steps_per_day=4,
            coarse_level=4,
            temporal_step=6,
            ground_truth=False,
            precompute_temporal_average=False
        ):
        self.temporal_step = temporal_step
        self.gt = ground_truth
        self.coarse_level = coarse_level
        if path.endswith(".zarr"):  self.dataset = xr.open_zarr(path,chunks=None)
        else:                       self.dataset = xr.open_dataset(path,chunks=None)
        
        startdate = np.array([self.dataset.time[0].to_numpy()])
        possible_startdate = startdate.max()
        if not (startdate == startdate[0]).all(): 
            print("Start dates of all arrays need to be the same! Otherwise changes to the Dataset class are needed!")
            sys.exit(0)
        # if int(np.datetime_as_string(possible_startdate,"M")) != 1 and int(np.datetime_as_string(possible_startdate,"D")) != 1 :
        #     print("Start dates need to be the 1/1 of a year! Otherwise changes to the Dataset class are needed!")
        #     print("For ERA5, 100v, 100u the end dates are",startdate)
        #     sys.exit(0)
        dataset_start = int(np.datetime_as_string(possible_startdate,"Y"))
        if start_year < int(np.datetime_as_string(possible_startdate,"Y")):
            print("chosen start year is earlier than the earliest common start date to all of the datasets")
            print("Start year is set to ",int(np.datetime_as_string(possible_startdate,"Y")))
            start_year = dataset_start
        
        # Check if set Start date to be viable
        enddate = np.array([self.dataset.time[-1].to_numpy() ])
        possible_enddate = enddate.min()
        if end_year > int(np.datetime_as_string(possible_enddate,"Y")):
            print("chosen end year is later than the latest common end date to all of the datasets")
            print("End year is set to ",int(np.datetime_as_string(possible_enddate,"Y")))
            end_year = int(np.datetime_as_string(possible_enddate,"Y"))

        self.start_idx = steps_per_day * sum([366 if isleap(year) else 365 for year in list(range(dataset_start, start_year))])
        self.end_idx = steps_per_day * sum([366 if isleap(year) else 365 for year in list(range(dataset_start, end_year))]) -1

        print("Using years: ",start_year," - ", end_year," (total length: ",self.end_idx - self.start_idx,") (availabe date range: ",np.datetime_as_string(possible_startdate,"Y"),"-",np.datetime_as_string(possible_enddate,"Y"),")")
        print("")
        
    def __len__(self):
        return self.end_idx - self.start_idx

    def __getitem__(self,idx):
        input = self.dataset.isel(time=slice(self.start_idx+idx, self.start_idx+idx + self.temporal_step))[["sea_surface_temperature"]].to_array()
        def format(input):
            time = str(input.time.to_numpy()[0])
            time = torch.tensor(int(time[0:4]+time[5:7]+time[8:10]+time[11:13])) # time in format YYYYMMDDHH  
            if self.coarse_level > 1:
                sst = input.coarsen(latitude=self.coarse_level,longitude=self.coarse_level,boundary='trim').mean()
            return torch.from_numpy(sst.to_numpy()).float(), time
        if self.gt:
            g_truth = self.dataset.isel(time=slice(self.start_idx+idx+1, self.start_idx+idx+1 + self.temporal_step))[["sea_surface_temperature"]].to_array()
            return format(input), format(g_truth)
        else:
            return [format(input)]
        # precompute_temporal_average not implemented
