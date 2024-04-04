# import xarray as xr
# import os
# import numpy as np

# ds = xr.open_dataset("/mnt/qb/goswami/data/era5/weatherbench2/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr")
# sst = ds["sea_surface_temperature"]

# save_path = "/mnt/qb/work2/goswami0/gkd965/Assets/sfno"

# mean = sst.mean()
# print("mean sst: ", mean)
# np.save(os.path.join(save_path,"global_means_sst8.npy"), mean)
# std = sst.std()
# print("std sst: ", std)
# np.save(os.path.join(save_path,"global_std_sst8.npy"), std)
# print("done")



## 0.25 Degree


# import xarray as xr
# import numpy as np
# import os


# ds = xr.open_dataset("/mnt/qb/goswami/data/era5/weatherbench2/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr")
# sst = ds["sea_surface_temperature"]
# save_path = "/mnt/qb/work2/goswami0/gkd965/Assets/sfno"

# idx = np.arange(93544)
# np.random.shuffle(idx)
# np.save(os.path.join(save_path,"sst_gen_idx.npy"), idx)

# count = 0
# total_count = 0
# first_moment = 0
# second_moment = 0
# for i in idx:
#     sst_slice = sst.isel(time=i)
#     first_moment += sst_slice.sum().to_numpy()
#     second_moment += (sst_slice**2).sum().to_numpy()
#     count += 1
#     total_count += 686364
#     if count % 100 == 0:
#         print('----')
#         print(count)
#         mean = first_moment/total_count
#         std = np.sqrt(second_moment/total_count-mean**2)
#         print(mean)
#         print(std)
#         if count % 1000 == 0:
#             np.save(os.path.join(save_path,"global_means_sst_{}.npy".format(count)), mean)
#             np.save(os.path.join(save_path,"global_firstM_sst_{}.npy".format(count)), first_moment)
#             np.save(os.path.join(save_path,"global_std_sst_{}.npy".format(count)), std)
#             np.save(os.path.join(save_path,"global_secondM_sst_{}.npy".format(count)), second_moment)
            
# mean = first_moment/total_count
# std = np.sqrt(second_moment/total_count-mean**2)
# print(mean)
# print(std)
# np.save(os.path.join(save_path,"global_means_sst_fin.npy"), mean)
# np.save(os.path.join(save_path,"global_std_sst_fin.npy"), std)


## 1 Degree


import xarray as xr
import numpy as np
import os


ds = xr.open_dataset("/mnt/qb/goswami/data/era5/weatherbench2/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr")
sst = ds["sea_surface_temperature"]
save_path = "/mnt/qb/work2/goswami0/gkd965/Assets/sfno/1deg"
save_path_sst_4 = "/mnt/qb/goswami/data/era5/sst_1_deg"
coarse_level = 4

idx = np.arange(93544)
# np.random.shuffle(idx)
np.save(os.path.join(save_path,"sst_gen_1_deg_idx.npy"), idx)

count = 0
total_count = 0
first_moment = 0
second_moment = 0
for i in idx:
    sst_slice = sst.isel(time=i).coarsen(latitude=coarse_level,longitude=coarse_level,boundary='trim').mean()
    sst_slice.to_netcdf(os.path.join(save_path_sst_4,str(count)+".nc"))
    non_nans = (~np.isnan(sst_slice)).sum().to_numpy()
    first_moment += sst_slice.sum().to_numpy()
    second_moment += (sst_slice**2).sum().to_numpy()
    count += 1
    total_count += non_nans
    if count % 100 == 0:
        print('----')
        print(count)
        mean = first_moment/total_count
        std = np.sqrt(second_moment/total_count-mean**2)
        print(mean)
        print(std)
        if count % 1000 == 0:
            np.save(os.path.join(save_path,"global_means_sst_1deg_{}.npy".format(count)), mean)
            np.save(os.path.join(save_path,"global_firstM_sst_1deg_{}.npy".format(count)), first_moment)
            np.save(os.path.join(save_path,"global_std_sst_1deg_{}.npy".format(count)), std)
            np.save(os.path.join(save_path,"global_secondM_sst_1deg_{}.npy".format(count)), second_moment)
    # if count == 3: break
        
mean = first_moment/total_count
std = np.sqrt(second_moment/total_count-mean**2)
print(mean)
print(std)
np.save(os.path.join(save_path,"global_means_sst_1deg_fin.npy"), mean)
np.save(os.path.join(save_path,"global_std_sst_1deg_fin.npy"), std)

