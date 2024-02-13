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
            years=[2017, 2018, 2019]
        ):
        self.path = path
        self.model = model
        self.years = years
        self.dataset = xr.open_dataset(self.path)

    def __len__(self):
        # steps_per_day = 4
        # return steps_per_day * sum([366 if isleap(year) else 365 for year in self.years])
        # time idx in zarr file: 93544, type np.datetime64[ns]
    
    def __getitem__(self, idx):

        sample = xr.open_zarr(self.path).sel

        return sample
    