import xarray as xr
import matplotlib.pyplot as plt
# import metview as mv

# ds = (mv.read("ERA5_levels.grib")).to_dataset()

ds = xr.load_dataset("/mnt/ssd2/Master/S2S_on_SFNO/outputs/sfno/SFNO_HalfYearForecast_20231116-1441.grib", engine="cfgrib")
print(ds.info())
