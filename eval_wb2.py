import apache_beam   # Needs to be imported separately to avoid TypingError
import weatherbench2
import xarray as xr
from weatherbench2 import config
from weatherbench2.metrics import MSE, ACC, RMSE, Bias
from weatherbench2.evaluation import evaluate_in_memory, evaluate_with_beam

forecast_path = 'gs://weatherbench2/datasets/hres/2016-2022-0012-64x32_equiangular_conservative.zarr'
obs_path = '/mnt/qb/goswami/data/era5/weatherbench2/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr'
climatology_path = 'gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_64x32_equiangular_conservative.zarr'

climatology = xr.open_zarr(climatology_path)

paths = config.Paths(
    forecast=forecast_path,
    obs=obs_path,
    output_dir='./',   # Directory to save evaluation results
)

selection = config.Selection(
    variables=[
        'geopotential',
        '2m_temperature'
    ],
    levels=[500, 700, 850],
    time_slice=slice('2016-01-01', '2016-12-31'),
)

data_config = config.Data(selection=selection, paths=paths)


eval_configs = {
  'deterministic': config.Eval(
      metrics={
          'mse': MSE(), 
          'acc': ACC(climatology=climatology),
          'bias': Bias(),
        'rmse': RMSE(),
      },
  )
}

evaluate_with_beam(
    data_config,
    eval_configs,
    runner='DirectRunner',
    input_chunks={'time': 20},
)