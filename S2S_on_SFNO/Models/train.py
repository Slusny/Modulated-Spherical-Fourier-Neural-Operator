import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset#, IterableDataset
from calendar import isleap
import xarray as xr
import numpy as np
import os

# from .sfno.model import get_model
from .sfno.sfnonet import GCN
import wandb
from time import time

# BatchSampler(drop_last=True)

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
        :param list     total_dataset_year_range  : the range of years the overall dataset at path.
        :param int      steps_per_day: How many datapoints per day in the dataset. For a 6hr increment we have 4 steps per day.
        :param bool     sst         : wether to include sea surface temperature in the data as seperate tuple element (used for filmed models)
        :param bool     uv100       : wether to include u100 and v100 in the data 
        
        :return: weather data as torch.tensor or tuple (weather data, sea surface temperature)   
    """
    def __init__(
            self, 
            model,
            # path="/mnt/ceph/goswamicd/datasets/weatherbench2/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr",#weatherbench2/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr", #1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr", 
            #path="/mnt/ceph/goswamicd/datasets/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr",#weatherbench2/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr", #1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr", 
            path = "/mnt/qb/goswami/data/era5/weatherbench2/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr", # start date: 1959-01-01 end date : 2023-01-10T18:00
            path_era5="/mnt/qb/goswami/data/era5/single_pressure_level/",
            start_year=2000,
            end_year=2010,
            total_dataset_year_range=[1959, 2023], # first date is 1/1/1959 last is 12/31/2022
            steps_per_day=4,
            sst=True,
            coarse_level=4,
            uv100=True,
        ):
        self.model = model
        self.sst = sst
        self.coarse_level = coarse_level
        self.uv100 = uv100
        if path.endswith(".zarr"):  self.dataset = xr.open_zarr(path)
        else:                       self.dataset = xr.open_dataset(path)
        if self.uv100:
            #qb
            # self.dataset_u100 = xr.open_mfdataset(os.path.join(path_era5+"100m_u_component_of_wind/100m_u_component_of_wind_????.nc"))
            # self.dataset_v100 = xr.open_mfdataset(os.path.join(path_era5+"100m_v_component_of_wind/100m_v_component_of_wind_????.nc"))
            #ceph
            # self.dataset_uv100 = xr.open_mfdataset("/mnt/ceph/goswamicd/datasets/weatherbench2/era5/1959-2023_01_10-u100mv100m-6h-1440x721"))
            # qb zarr
            file_u100 = "/mnt/qb/goswami/data/era5/u100m_v100m_721x1440/u100m_1959-2022_721x1440_correct_chunk_new_mean_INTERPOLATE.zarr"
            if file_u100.endswith(".zarr"): self.dataset_u100 = xr.open_zarr(file_u100)
            else:                           self.dataset_u100 = xr.open_mfdataset(file_u100) # sd: 1959-01-01, end date : 2022-12-30T18
            file_v100 = "/mnt/qb/goswami/data/era5/u100m_v100m_721x1440/v100m_1959-2023-10_721x1440_correct_chunk_new_mean_INTERPOLATE.zarr"
            if file_u100.endswith(".zarr"): self.dataset_v100 = xr.open_zarr(file_v100)
            else:                           self.dataset_v100 = xr.open_mfdataset(file_v100) # sd: 1959-01-01 end date: 2023-10-31

        print("Using years: ",start_year," - ", end_year)
        print("")

        self.start_idx = steps_per_day * sum([366 if isleap(year) else 365 for year in list(range(total_dataset_year_range[0], start_year))])
        self.end_idx = steps_per_day * sum([366 if isleap(year) else 365 for year in list(range(total_dataset_year_range[0], end_year))])

    def __len__(self):
        return self.end_idx - self.start_idx
    
    def __getitem__(self, idx):
        level_list = self.model.param_level_pl[1].copy()
        level_list.reverse()
        input = self.dataset.isel(time=self.start_idx + idx)
        g_truth = self.dataset.isel(time=self.start_idx + idx+1)

        def format(sample):
            scf = sample[self.model.param_sfc_ERA5].to_array().to_numpy()
            pl = sample[list(self.model.levels_per_pl.keys())].sel(level=level_list).to_array().to_numpy()
            pl = pl.reshape((pl.shape[0]*pl.shape[1], pl.shape[2], pl.shape[3]))
            
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
                data = torch.from_numpy(np.vstack((scf,pl)))
            if self.sst:
                sst = sample["sea_surface_temperature"]
                if self.coarse_level > 1:
                    sst = sst.coarsen(latitude=self.coarse_level,longitude=self.coarse_level,boundary='trim').mean().to_numpy()
                # if self.coarse_level > 1:
                #     sst = sst.to_numpy()[:-1:self.coarse_level,::self.coarse_level] # or numpy at the end
                return (data,torch.from_numpy(sst))
            else:
                return data
        
        return format(input), format(g_truth)

class ERA5_galvani_coarsen(Dataset):
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
        :param list     total_dataset_year_range  : the range of years the overall dataset at path.
        :param int      steps_per_day: How many datapoints per day in the dataset. For a 6hr increment we have 4 steps per day.
        :param bool     sst         : wether to include sea surface temperature in the data as seperate tuple element (used for filmed models)
        :param bool     uv100       : wether to include u100 and v100 in the data 
        
        :return: weather data as torch.tensor or tuple (weather data, sea surface temperature)   
    """
    def __init__(
            self, 
            model,
            # path="/mnt/ceph/goswamicd/datasets/weatherbench2/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr",#weatherbench2/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr", #1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr", 
            #path="/mnt/ceph/goswamicd/datasets/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr",#weatherbench2/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr", #1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr", 
            path = "/mnt/qb/goswami/data/era5/weatherbench2/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr", # start date: 1959-01-01 end date : 2023-01-10T18:00
            path_era5="/mnt/qb/goswami/data/era5/single_pressure_level/",
            start_year=2000,
            end_year=2010,
            total_dataset_year_range=[1959, 2023], # first date is 1/1/1959 last is 12/31/2022
            steps_per_day=4,
            sst=True,
            coarse_level=4,
            uv100=True,
        ):
        self.model = model
        self.sst = sst
        self.coarse_level = coarse_level
        self.uv100 = uv100
        if path.endswith(".zarr"):  self.dataset = xr.open_zarr(path)
        else:                       self.dataset = xr.open_dataset(path)
        if self.uv100:
            #qb
            # self.dataset_u100 = xr.open_mfdataset(os.path.join(path_era5+"100m_u_component_of_wind/100m_u_component_of_wind_????.nc"))
            # self.dataset_v100 = xr.open_mfdataset(os.path.join(path_era5+"100m_v_component_of_wind/100m_v_component_of_wind_????.nc"))
            #ceph
            # self.dataset_uv100 = xr.open_mfdataset("/mnt/ceph/goswamicd/datasets/weatherbench2/era5/1959-2023_01_10-u100mv100m-6h-1440x721"))
            # qb zarr
            file_u100 = "/mnt/qb/goswami/data/era5/u100m_v100m_721x1440/u100m_1959-2022_721x1440_correct_chunk_new_mean_INTERPOLATE.zarr"
            if file_u100.endswith(".zarr"): self.dataset_u100 = xr.open_zarr(file_u100)
            else:                           self.dataset_u100 = xr.open_mfdataset(file_u100) # sd: 1959-01-01, end date : 2022-12-30T18
            file_v100 = "/mnt/qb/goswami/data/era5/u100m_v100m_721x1440/v100m_1959-2023-10_721x1440_correct_chunk_new_mean_INTERPOLATE.zarr"
            if file_u100.endswith(".zarr"): self.dataset_v100 = xr.open_zarr(file_v100)
            else:                           self.dataset_v100 = xr.open_mfdataset(file_v100) # sd: 1959-01-01 end date: 2023-10-31

        print("Using years: ",start_year," - ", end_year)
        print("")

        self.start_idx = steps_per_day * sum([366 if isleap(year) else 365 for year in list(range(total_dataset_year_range[0], start_year))])
        self.end_idx = steps_per_day * sum([366 if isleap(year) else 365 for year in list(range(total_dataset_year_range[0], end_year))])

    def __len__(self):
        return self.end_idx - self.start_idx
    
    def __getitem__(self, idx):
        level_list = self.model.param_level_pl[1].copy()
        level_list.reverse()
        input = self.dataset.isel(time=self.start_idx + idx)
        g_truth = self.dataset.isel(time=self.start_idx + idx+1)

        def format(sample):
            scf = sample[self.model.param_sfc_ERA5].to_array().to_numpy()
            pl = sample[list(self.model.levels_per_pl.keys())].sel(level=level_list).to_array().to_numpy()
            pl = pl.reshape((pl.shape[0]*pl.shape[1], pl.shape[2], pl.shape[3]))
            
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
                data = torch.from_numpy(np.vstack((scf,pl)))
            if self.sst:
                sst = sample["sea_surface_temperature"]
                if self.coarse_level > 1:
                    s = time()
                    for i in range(500):
                        tst = sst.coarsen(latitude=self.coarse_level,longitude=self.coarse_level,boundary='trim').mean()
                    e = time()
                    print("Time to coarsen: ", e-s)
                    s = time()
                    for i in range(500):
                        tst = sst[:-1:self.coarse_level,::self.coarse_level]
                    e = time()
                    print("Time to mask: ", e-s)
                # if self.coarse_level > 1:
                    # sst = sst[:-1:self.coarse_level,::self.coarse_level]
                return (data,torch.from_numpy(sst.to_numpy()))
            else:
                return data
        
        return format(input), format(g_truth)

param_level_pl = (
        [ "u", "v", "z", "t", "r"],
        [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50],
    )
param_sfc_ERA5 = ["10m_u_component_of_wind", "10m_v_component_of_wind", "2m_temperature", "surface_pressure", "mean_sea_level_pressure", "total_column_water_vapour"]
levels_per_pl = {"u_component_of_wind":[1000,925,850,700,600,500,400,300,250,200,150,100,50],
                     "v_component_of_wind":[1000,925,850,700,600,500,400,300,250,200,150,100,50],
                     "geopotential":[1000,925,850,700,600,500,400,300,250,200,150,100,50],
                     "temperature":[1000,925,850,700,600,500,400,300,250,200,150,100,50],
                     "relative_humidity":[1000,925,850,700,600,500,400,300,250,200,150,100,50]}
# params = {"param_level_pl":param_level_pl, "param_sfc_ERA5":param_sfc_ERA5, "levels_per_pl":levels_per_pl}
class param:
    param_level_pl = param_level_pl 
    param_sfc_ERA5 = param_sfc_ERA5
    levels_per_pl = levels_per_pl
params = param()

def train(kwargs):
    # model = get_model(kwargs)

    # dataset = ERA5_galvani(
    #     params,
    #     path=kwargs["trainingdata_path"], 
    #     start_year=kwargs["trainingset_start_year"],
    #     end_year=kwargs["trainingset_end_year"])

    dataset = ERA5_galvani(
        params
    )

    #

    training_loader = DataLoader(dataset,shuffle=True,num_workers=kwargs["training_workers"], batch_size=kwargs["batch_size"])

    # w_run = wandb.init(project="GCN to One 2",config=kwargs)

    l1 = time()
    test=False
    if test:
        for i in range(0,100,10):

            model1 = GCN(kwargs["batch_size"])
            model1.eval()
            model1.load_state_dict(torch.load("/mnt/qb/work2/goswami0/gkd965/GCN/model_2_10.pth"))

        # model2 = GCN(kwargs["batch_size"])
        # model2.eval()
        # model2.load_state_dict(torch.load("/mnt/qb/work2/goswami0/gkd965/GCN/model_2_20.pth"))
        # model3 = GCN(kwargs["batch_size"])
        # model3.eval()
        # model3.load_state_dict(torch.load("/mnt/qb/work2/goswami0/gkd965/GCN/model_2_30.pth"))
        # while True:
        #     for i, data in enumerate(training_loader):
        #         print("Batch: ", i+1, "/", len(training_loader))

        #         input, truth = data
        #         sst = input[1] 
        #         outputs1 = model1(sst)
        #         outputs2 = model2(sst)
        #         outputs3 = model3(sst)
        #         print(outputs1)
        #         print(outputs2)
        #         print(outputs3)
        data = next(iter(training_loader))
        input, truth = data
        sst = input[1] 
        for i in range(0,110,10):
            model1 = GCN(kwargs["batch_size"])
            model1.eval()
            model1.load_state_dict(torch.load("/mnt/qb/work2/goswami0/gkd965/GCN/model_2_{}.pth".format(i)))
            outputs1 = model1(sst)
            print("mean 0",outputs1[0].mean())
            print("std 0",outputs1[0].std())
            print("mean 1",outputs1[1].mean())
            print("std 1",outputs1[1].std())
            print("---------------------")

        sys.exit(0)
# 0 ---------------------
# mean 0 tensor(0.2065, grad_fn=<MeanBackward0>)
# std 0 tensor(3.0127, grad_fn=<StdBackward0>)
# mean 1 tensor(0.0764, grad_fn=<MeanBackward0>)
# std 1 tensor(2.9966, grad_fn=<StdBackward0>)
# 10 ---------------------
# mean 0 tensor(0.1641, grad_fn=<MeanBackward0>)
# std 0 tensor(0.3530, grad_fn=<StdBackward0>)
# mean 1 tensor(0.0021, grad_fn=<MeanBackward0>)
# std 1 tensor(0.3527, grad_fn=<StdBackward0>)
# 20 ---------------------
# mean 0 tensor(0.5030, grad_fn=<MeanBackward0>)
# std 0 tensor(0.3747, grad_fn=<StdBackward0>)
# mean 1 tensor(-0.0113, grad_fn=<MeanBackward0>)
# std 1 tensor(0.3448, grad_fn=<StdBackward0>)
# 30 ---------------------
# mean 0 tensor(0.8485, grad_fn=<MeanBackward0>)
# std 0 tensor(0.3082, grad_fn=<StdBackward0>)
# mean 1 tensor(-0.0072, grad_fn=<MeanBackward0>)
# std 1 tensor(0.2392, grad_fn=<StdBackward0>)
# 40 ---------------------
# mean 0 tensor(0.9506, grad_fn=<MeanBackward0>)
# std 0 tensor(0.1761, grad_fn=<StdBackward0>)
# mean 1 tensor(-0.0022, grad_fn=<MeanBackward0>)
# std 1 tensor(0.1178, grad_fn=<StdBackward0>)
# 50 ---------------------
# mean 0 tensor(0.9833, grad_fn=<MeanBackward0>)
# std 0 tensor(0.1050, grad_fn=<StdBackward0>)
# mean 1 tensor(0.0017, grad_fn=<MeanBackward0>)
# std 1 tensor(0.0729, grad_fn=<StdBackward0>)
# 60 ---------------------/////////////////////////
# mean 0 tensor(0.9937, grad_fn=<MeanBackward0>)
# std 0 tensor(0.0602, grad_fn=<StdBackward0>)
# mean 1 tensor(0.0008, grad_fn=<MeanBackward0>)
# std 1 tensor(0.0434, grad_fn=<StdBackward0>)
# 70 ---------------------
# mean 0 tensor(1.0010, grad_fn=<MeanBackward0>)
# std 0 tensor(0.0346, grad_fn=<StdBackward0>)
# mean 1 tensor(7.5858e-05, grad_fn=<MeanBackward0>)
# std 1 tensor(0.0267, grad_fn=<StdBackward0>)
# 80 ---------------------
# mean 0 tensor(0.9971, grad_fn=<MeanBackward0>)
# std 0 tensor(0.0201, grad_fn=<StdBackward0>)
# mean 1 tensor(0.0002, grad_fn=<MeanBackward0>)
# std 1 tensor(0.0160, grad_fn=<StdBackward0>)
# 90 ---------------------
# mean 0 tensor(1.0015, grad_fn=<MeanBackward0>)
# std 0 tensor(0.0121, grad_fn=<StdBackward0>)
# mean 1 tensor(0.0001, grad_fn=<MeanBackward0>)
# std 1 tensor(0.0099, grad_fn=<StdBackward0>)
# 100 ---------------------
# mean 0 tensor(0.9977, grad_fn=<MeanBackward0>)
# std 0 tensor(0.0069, grad_fn=<StdBackward0>)
# mean 1 tensor(5.8214e-05, grad_fn=<MeanBackward0>)
# std 1 tensor(0.0058, grad_fn=<StdBackward0>)
# ---------------------

    model = GCN(kwargs["batch_size"])
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()
    mean_batch_time = 0
    mean_model_time = 0
    for i, data in enumerate(training_loader):
        print("Batch: ", i+1, "/", len(training_loader))
        # time
        l2 = time()
        tb = l2-l1
        mean_batch_time = mean_batch_time+(tb - mean_batch_time)/(i+1)
        print("Time to load batch: ", tb , " mean : ", mean_batch_time) 
        # needs 40s for 1 worker with 4 batch size
        # needs 10s for 3 workers with 4 batch size
        # needs 4.6-1.2=3.4s for 3 workers with 1 batch size
        l1 = l2

        input, truth = data
        sst = input[1] 
        # # if coarsen isn't already done on disk
        # corse_deg = 4
        # sst = sst.coarse_level,longitude=self.coarse_level,boundary='trim').mean().to_array()[0]
        optimizer.zero_grad()

        # Make predictions for this batch
        s = time()
        outputs = model(sst) 
        # runs 3.3s, more workers 4.5s
        # one batch runs 1.2s
        e = time()
        tm = e-s
        mean_model_time = mean_model_time+(tm - mean_model_time)/(i+1)
        print("Time to run model: ", tm , " mean : ", mean_model_time)
        truth = torch.stack([torch.ones_like(outputs[0]),torch.zeros_like(outputs[1])])

        # Compute the loss and its gradients
        loss = loss_fn(outputs, truth)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Log the loss
        # wandb.log({"loss": loss.item()})

        # save the model
        # if i % 10 == 0:
        #     print("saving model")
        #     torch.save(model.state_dict(), "/mnt/qb/work2/goswami0/gkd965/GCN/model_2_{}.pth".format(i))

def test(kwargs):

    dataset_masked = ERA5_galvani(
        params,
        start_year=2000,
        end_year=2010,
    )
    dataset_coarsen = ERA5_galvani_coarsen(
        params,
        start_year=1990,
        end_year=2000,
    )
    for i in range(5,8):
        print("--- Workers: ", i, " ---")
        coarsen_loader = DataLoader(dataset_coarsen,shuffle=True,num_workers=i, batch_size=kwargs["batch_size"])
        masked_loader = DataLoader(dataset_masked,shuffle=True,num_workers=i, batch_size=kwargs["batch_size"])

        # s_coarsen = time()
        c_times = []
        count = 1
        end_count = 10
        s_coarsen = time()
        for i, data in enumerate(coarsen_loader):
            e_coarsen = time()
            count += 1
            if count == end_count: break
            # print(len(data))
            c_times.append(e_coarsen-s_coarsen)
            s_coarsen = time()
        # e_coarsen = time()
        # print("Time to load coarsen: ", (e_coarsen-s_coarsen)/count)
        print("Time to load coarsen: ", np.array(c_times).mean())  
        print(c_times)  

        # s_masked = time()
        m_times = []
        count = 0
        s_masked = time()
        for i, data in enumerate(masked_loader):
            e_masked = time()
            count += 1
            if count == end_count: break
            # print(len(data))
            m_times.append(e_masked-s_masked)
            s_masked = time()
        # e_masked = time()
        # print("Time to load masked: ", (e_masked-s_masked)/count)
        print("Time to load masked: ", np.array(m_times).mean())    
        print(m_times)