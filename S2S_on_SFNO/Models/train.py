import torch
from torch.utils.data import BatchSampler, IterableDataset
from calendar import isleap
import xarray as xr
import numpy as np
import os

# BatchSampler(drop_last=True)

class ERA5_galvani(IterableDataset):
    def __init__(
            self, 
            model,
            path="/mnt/ceph/goswamicd/datasets/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr", 
            path_era5="/mnt/qb/goswami/data/era5/single_pressure_level/",
            # model="sfno", 
            start_year=2000,
            end_year=2010,
            total_dataset_year_range=[1959, 2023],
            steps_per_day=4
        ):
        # self.path = path
        self.model = model
        # self.years = years
        # self.steps_per_day = steps_per_day
        # self.total_dataset_year_range = total_dataset_year_range
        self.dataset = xr.open_dataset(path)
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
        sample = self.dataset.isel(time=self.start_idx + idx)
        g_truth = self.dataset.isel(time=self.start_idx + idx+1)

        def format(sample):
            scf = sample[self.model.param_sfc_ERA5].to_array().to_numpy()
            pl = sample[list(self.model.levels_per_pl.keys())].sel(level=level_list).to_array().to_numpy()
            pl = pl.reshape((pl.shape[0]*pl.shape[1], pl.shape[2], pl.shape[3]))
            u100 = self.dataset_u100.isel(time=self.start_idx + idx)["u100"].to_numpy()[None]
            v100 = self.dataset_v100.isel(time=self.start_idx + idx)["v100"].to_numpy()[None]
            return torch.tensor(np.vstack((
                scf[:2],
                u100,
                v100,
                scf[2:],
                pl)))
        
        return format(sample), format(g_truth)