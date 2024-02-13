import torch
from torch.utils.data import BatchSampler, IterableDataset
from calendar import isleap
import xarray as xr

BatchSampler(drop_last=True)

class ERA5_galvani(IterableDataset):
    def __init__(
            self, 
            path="/mnt/ceph/goswamicd/datasets/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr", 
            model="sfno", 
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
        self.dataset = xr.open_dataset(self.path)

        self.start_idx = steps_per_day * sum([366 if isleap(year) else 365 for year in list(range(total_dataset_year_range[0], start_year))])
        self.end_idx = steps_per_day * sum([366 if isleap(year) else 365 for year in list(range(total_dataset_year_range[0], end_year))])

    def __len__(self):
        return self.end_idx - self.start_idx
    
    def __getitem__(self, idx):

        sample = xr.open_zarr(self.path).isel(time=self.start_idx + idx)

        return sample
    