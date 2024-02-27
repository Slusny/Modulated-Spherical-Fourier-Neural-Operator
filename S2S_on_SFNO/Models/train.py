import torch
from torch.utils.data import BatchSampler, IterableDataset
from calendar import isleap
import xarray as xr
import numpy as np
import os

from S2S_on_SFNO.Models.sfno.model import get_model
from S2S_on_SFNO.Models.sfno.sfnonet import GCN
from torch.utils.data import DataLoader

# BatchSampler(drop_last=True)

class ERA5_galvani(IterableDataset):
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
            path="/mnt/ceph/goswamicd/datasets/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr", 
            path_era5="/mnt/qb/goswami/data/era5/single_pressure_level/",
            start_year=2000,
            end_year=2010,
            total_dataset_year_range=[1959, 2023],
            steps_per_day=4,
            sst=True,
            uv100=True,
        ):
        self.model = model
        self.sst = sst
        self.uv100 = uv100
        self.dataset = xr.open_dataset(path)
        if self.uv100:
            self.dataset_u100 = xr.open_mfdataset(os.path.join(path_era5+"100m_u_component_of_wind/100m_u_component_of_wind_????.nc"))
            self.dataset_v100 = xr.open_mfdataset(os.path.join(path_era5+"100m_v_component_of_wind/100m_v_component_of_wind_????.nc"))

        print("Training on years:")
        print("    ", start_year," - ", end_year)

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
                u100 = self.dataset_u100.isel(time=self.start_idx + idx)["u100"].to_numpy()[None]
                v100 = self.dataset_v100.isel(time=self.start_idx + idx)["v100"].to_numpy()[None]
                data =  torch.tensor(np.vstack((
                    scf[:2],
                    u100,
                    v100,
                    scf[2:],
                    pl)))
            else: 
                data = torch.tensor(np.vstack((scf,pl)))
            if self.sst:
                sst = sample["sea_surface_temperature"].to_numpy()
                (data,torch.tensor(sst))
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
params = {param_level_pl:param_level_pl, param_sfc_ERA5:param_sfc_ERA5, levels_per_pl:levels_per_pl}

def train(kwargs):
    # model = get_model(kwargs)
    model = GCN()

    # dataset = ERA5_galvani(
    #     params,
    #     path=kwargs["trainingdata_path"], 
    #     start_year=kwargs["trainingset_start_year"],
    #     end_year=kwargs["trainingset_end_year"])

    dataset = ERA5_galvani(
        params
    )

    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    loss_fn = torch.nn.MSELoss()

    training_loader = DataLoader(dataset,shuffle=True,num_workers=kwargs["training_workers"], batch_size=kwargs["batch_size"])

    for i, data in enumerate(training_loader):
        input, _ = data
        sst = input[1]
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(sst)
        labels = torch.stack([torch.zeros_like(outputs[0]),torch.zeros_like(outputs[1])])

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

train({"training_workers":1,"batch_size":3})