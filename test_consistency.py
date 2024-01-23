import numpy as np
import xarray as xr
# import matplotlib.pyplot as plt
# import matplotlib

dir   = "/mnt/ssd2/Master/S2S_on_SFNO/outputs/sfno/"
sfno_files = [
     "leadtime_18_startDate_201901010_createdOn_20240123T0009",
     "leadtime_18_startDate_201901010_createdOn_20240123T1219",
     "leadtime_18_startDate_201901010_createdOn_20240123T1220",
     "leadtime_18_startDate_201901010_createdOn_20240123T1221",
     "leadtime_18_startDate_201901010_createdOn_20240123T1222",
     "leadtime_18_startDate_201901010_createdOn_20240123T1223",
]

l = []
g = []

for file in sfno_files:
    #sfno
    sfno_path = dir+file+"/*"
    sfno_data = xr.open_mfdataset(sfno_path)["10u"][0].to_numpy()
    l.append(sfno_data)
    # grib
    grib_path = dir+file+".grib"
    grib_data = xr.open_dataset(grib_path)["u10"][0].to_numpy()
    g.append(sfno_data)
    print("done with",file)

sfno_mean = np.mean(l,axis=0)
grib_mean = np.mean(g,axis=0)
# print(sfno_mean)

# max difference 1.9073486e-06 (exactly machine precision times 16) # machine precision 1.1920929e-07 , 6 accurate decimals
MSE_sfno = np.mean(np.array(l) - sfno_mean,axis=0)
MSE_grib = np.mean(np.array(g) - grib_mean,axis=0)

for i in range(1,len(l)):
    print(((np.array(l[i]) - np.array(l[0])) != 0.).sum())


for i in range(1,len(g)):
    print(((np.array(g[i]) - np.array(g[0])) != 0.).sum())


print(((np.array(l[0]) - np.array(g[0])) != 0.).sum())

# total_mean = np.mean(sfno_mean)
# print(MSE_grib)
# print(MSE_sfno)

# print(total_mean)