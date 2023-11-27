import xarray as xr
from atmodata.utils import SequentialTransform
from atmodata.datasets import ERA5
from atmodata.builder import AtmodataPipeBuilder
from atmodata.tasks import ForecastingTask
from torch.utils.data import DataLoader
# import metview as mv

# ds = (mv.read("ERA5_levels.grib")).to_dataset()

# ds = xr.load_dataset("/mnt/ssd2/Master/S2S_on_SFNO/outputs/sfno/SFNO_HalfYearForecast_20231116-1441.grib", engine="cfgrib")
# print(ds.info())
print("start")
path = "/mnt/qb/goswami/data/era5"
variables = ['z500', 't500']
years = [2021]
era5 = ERA5(path,variables,years)

task = SequentialTransform(
    ForecastingTask(
        1,
        1
    )
)


from torch.utils.data.datapipes.iter.sharding import SHARDING_PRIORITIES
from torchdata.datapipes.iter import IterDataPipe


class ForecastingIterDataPipe(IterDataPipe):
    def __init__(self, dp, steps, rate, dim='time', crop_size=None, crops_per_sample=1, shuffle=False):
        assert crops_per_sample >= 1
        assert rate >= 1
        assert steps >= 1

        pipe = dp.xr_unroll_indices(dim=dim, shuffle=shuffle)
        pipe = pipe.sharding_filter(SHARDING_PRIORITIES.MULTIPROCESSING)
        pipe = pipe.xr_extract_timeseries(steps, rate, dim=dim)
        if crop_size:
            if crops_per_sample > 1:
                pipe = pipe.repeat(crops_per_sample)
            pipe = pipe.xr_random_crop(crop_size, wraps={'lon': True})
            if crops_per_sample > 1:
                pipe = pipe.shuffle(buffer_size=crops_per_sample * 4)
        self.dp = pipe

    def __iter__(self):
        return iter(self.dp)


ForecastingTask = ForecastingIterDataPipe.as_transform

builder = AtmodataPipeBuilder(
        era5,
        task,
        batch_size=1,
        num_parallel_shards=3,
        dataloading_prefetch_cnt=3,
        device_prefetch_cnt=2,
    )
# if args.cuda:
#     builder.transfer_to_device('cuda')
workers=3
dataloader = builder.multiprocess(workers).build_dataloader()

first = next(iter(dataloader))
print(first)
print(first.shape)
print("end")