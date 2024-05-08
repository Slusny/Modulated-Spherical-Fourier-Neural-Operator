import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset#, IterableDataset
import torch.cuda.amp as amp
from calendar import isleap
import xarray as xr
import numpy as np
import os
import sys
# from .sfno.model import get_model
from .sfno.sfnonet import GCN
import wandb
from time import time

# BatchSampler(drop_last=True)

from S2S_on_SFNO.Models.provenance import system_monitor
from .losses import CosineMSELoss, L2Sphere, NormalCRPS
from S2S_on_SFNO.utils import Timer, Attributes

import logging
LOG = logging.getLogger('S2S_on_SFNO')

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
        :param int      steps_per_day: How many datapoints per day in the dataset. For a 6hr increment we have 4 steps per day.
        :param bool     sst         : wether to include sea surface temperature in the data as seperate tuple element (used for filmed models)
        :param bool     uv100       : wether to include u100 and v100 in the data 
        :param int      auto_regressive_steps : how many consecutive datapoints should be loaded to used to calculate an autoregressive loss 
        
        :return: weather data as torch.tensor or tuple (weather data, sea surface temperature)   
    """
    def __init__(
            self, 
            model,
            # path="/mnt/ceph/goswamicd/datasets/weatherbench2/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr",#weatherbench2/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr", #1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr", 
            #path="/mnt/ceph/goswamicd/datasets/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr",#weatherbench2/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr", #1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr", 
            path = "/mnt/qb/goswami/data/era5/weatherbench2/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr", # start date: 1959-01-01 end date : 2023-01-10T18:00
            # chunked
            # path = "/mnt/qb/goswami/data/era5/weatherbench2/1959-2022-wb13-6h-0p25deg-chunk-1.zarr-v2", # start date: 1959-01-01 end date : 2023-01-10T18:00 # not default, needs derived variables
            
            path_era5="/mnt/qb/goswami/data/era5/single_pressure_level/",
            start_year=2000,
            end_year=2022,
            steps_per_day=4,
            sst=True,
            coarse_level=4,
            uv100=True,
            auto_regressive_steps=0
        ):
        self.model = model
        self.sst = sst
        self.auto_regressive_steps = auto_regressive_steps
        self.coarse_level = coarse_level
        self.uv100 = uv100
        if path.endswith(".zarr"):  self.dataset = xr.open_zarr(path,chunks=None)
        else:                       self.dataset = xr.open_dataset(path,chunks=None)
        if self.uv100:
            #qb
            # self.dataset_u100 = xr.open_mfdataset(os.path.join(path_era5+"100m_u_component_of_wind/100m_u_component_of_wind_????.nc"))
            # self.dataset_v100 = xr.open_mfdataset(os.path.join(path_era5+"100m_v_component_of_wind/100m_v_component_of_wind_????.nc"))
            #ceph
            # self.dataset_uv100 = xr.open_mfdataset("/mnt/ceph/goswamicd/datasets/weatherbench2/era5/1959-2023_01_10-u100mv100m-6h-1440x721"))
            # qb zarr
            file_u100 = "/mnt/qb/goswami/data/era5/u100m_v100m_721x1440/u100m_1959-2022_721x1440_correct_chunk_new_mean_INTERPOLATE.zarr"
            if file_u100.endswith(".zarr"): self.dataset_u100 = xr.open_zarr(file_u100,chunks=None)
            else:                           self.dataset_u100 = xr.open_mfdataset(file_u100,chunks=None) # sd: 1959-01-01, end date : 2022-12-30T18
            file_v100 = "/mnt/qb/goswami/data/era5/u100m_v100m_721x1440/v100m_1959-2023-10_721x1440_correct_chunk_new_mean_INTERPOLATE.zarr"
            if file_u100.endswith(".zarr"): self.dataset_v100 = xr.open_zarr(file_v100,chunks=None)
            else:                           self.dataset_v100 = xr.open_mfdataset(file_v100,chunks=None) # sd: 1959-01-01 end date: 2023-10-31

        # check if the 100uv-datasets and era5 have same start and end date
        # Check if set Start date to be viable
        startdate = np.array([self.dataset.time[0].to_numpy() ,self.dataset_v100.time[0].to_numpy() ,self.dataset_u100.time[0].to_numpy()])
        possible_startdate = startdate.max()
        if not (startdate == startdate[0]).all(): 
            print("Start dates of all arrays need to be the same! Otherwise changes to the Dataset class are needed!")
            print("For ERA5, 100v, 100u the end dates are",startdate)
            sys.exit(0)
        # if int(np.datetime_as_string(possible_startdate,"M")) != 1 and int(np.datetime_as_string(possible_startdate,"D")) != 1 :
        #     print("Start dates need to be the 1/1 of a year! Otherwise changes to the Dataset class are needed!")
        #     print("For ERA5, 100v, 100u the end dates are",startdate)
        #     sys.exit(0)
        dataset_start = int(np.datetime_as_string(possible_startdate,"Y"))
        if start_year < int(np.datetime_as_string(possible_startdate,"Y")):
            print("chosen start year is earlier than the earliest common start date to all of the datasets")
            print("Start year is set to ",int(np.datetime_as_string(possible_startdate,"Y")))
            print("For ERA5, 100v, 100u the end dates are",startdate)
            start_year = dataset_start
        
        # Check if set Start date to be viable
        enddate = np.array([self.dataset.time[-1].to_numpy() ,self.dataset_v100.time[-1].to_numpy() ,self.dataset_u100.time[-1].to_numpy()])
        possible_enddate = enddate.min()
        if end_year > int(np.datetime_as_string(possible_enddate,"Y")):
            print("chosen end year is later than the latest common end date to all of the datasets")
            print("End year is set to ",int(np.datetime_as_string(possible_enddate,"Y")))
            print("For ERA5, 100v, 100u the end dates are",enddate)
            end_year = int(np.datetime_as_string(possible_enddate,"Y"))

        self.start_idx = steps_per_day * sum([366 if isleap(year) else 365 for year in list(range(dataset_start, start_year))])
        self.end_idx = steps_per_day * sum([366 if isleap(year) else 365 for year in list(range(dataset_start, end_year))]) -1

        print("Using years: ",start_year," - ", end_year," (total length: ",self.end_idx - self.start_idx,") (availabe date range: ",np.datetime_as_string(possible_startdate,"Y"),"-",np.datetime_as_string(possible_enddate,"Y"),")")
        print("")

    def __len__(self):
        return self.end_idx - self.start_idx
    
    def __getitem__(self, idx):
        level_list = self.model.param_level_pl[1].copy()
        level_list.reverse()

        def format(sample):
            scf = sample[self.model.param_sfc_ERA5].to_array().to_numpy()
            pl = sample[list(self.model.levels_per_pl.keys())].sel(level=level_list).to_array().to_numpy()
            pl = pl.reshape((pl.shape[0]*pl.shape[1], pl.shape[2], pl.shape[3]))
            time = str(sample.time.to_numpy())
            time = torch.tensor(int(time[0:4]+time[5:7]+time[8:10]+time[11:13])) # time in format YYYYMMDDHH
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
                data = torch.from_numpy(np.vstack((scf,pl))) # transpose to have the same shape as expected by SFNO (lat,long)
            if self.sst:
                sst = sample["sea_surface_temperature"]
                if self.coarse_level > 1:
                    sst = sst.coarsen(latitude=self.coarse_level,longitude=self.coarse_level,boundary='trim').mean().to_numpy()
                # if self.coarse_level > 1:
                #     sst = sst.to_numpy()[:-1:self.coarse_level,::self.coarse_level] # or numpy at the end
                return (data,torch.from_numpy(sst),time)
            else:
                return (data,time)
        
        if self.auto_regressive_steps > 0:
            data = []
            for i in range(self.auto_regressive_steps+2):
                data.append(format(self.dataset.isel(time=self.start_idx + idx + i)))
            return data
        else:
            input = self.dataset.isel(time=self.start_idx + idx)
            g_truth = self.dataset.isel(time=self.start_idx + idx+1)
            return format(input), format(g_truth)

    def autoregressive_sst(self,idx):
        ssts = []
        for step in range(1,self.auto_regressive_steps):
            sst = self.dataset.isel(time=self.start_idx + idx + step)["sea_surface_temperature"]
            if self.coarse_level > 1:
                sst = sst.coarsen(latitude=self.coarse_level,longitude=self.coarse_level,boundary='trim').mean()
                # if self.coarse_level > 1:
                #     sst = sst.to_numpy()[:-1:self.coarse_level,::self.coarse_level] # or numpy at the end
            ssts.append(torch.from_numpy(sst.to_numpy()))
        return ssts

class SST_galvani(Dataset):
    def __init__(
            self, 
            path = "/mnt/qb/goswami/data/era5/weatherbench2/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr", # start date: 1959-01-01 end date : 2023-01-10T18:00
            start_year=2000,
            end_year=2022,
            steps_per_day=4,
            coarse_level=4,
            temporal_step=6,
            ground_truth=False,
            precompute_temporal_average=False
        ):
        self.temporal_step = temporal_step
        self.gt = ground_truth
        self.coarse_level = coarse_level
        if path.endswith(".zarr"):  self.dataset = xr.open_zarr(path,chunks=None)
        else:                       self.dataset = xr.open_dataset(path,chunks=None)
        
        startdate = np.array([self.dataset.time[0].to_numpy()])
        possible_startdate = startdate.max()
        if not (startdate == startdate[0]).all(): 
            print("Start dates of all arrays need to be the same! Otherwise changes to the Dataset class are needed!")
            sys.exit(0)
        # if int(np.datetime_as_string(possible_startdate,"M")) != 1 and int(np.datetime_as_string(possible_startdate,"D")) != 1 :
        #     print("Start dates need to be the 1/1 of a year! Otherwise changes to the Dataset class are needed!")
        #     print("For ERA5, 100v, 100u the end dates are",startdate)
        #     sys.exit(0)
        dataset_start = int(np.datetime_as_string(possible_startdate,"Y"))
        if start_year < int(np.datetime_as_string(possible_startdate,"Y")):
            print("chosen start year is earlier than the earliest common start date to all of the datasets")
            print("Start year is set to ",int(np.datetime_as_string(possible_startdate,"Y")))
            start_year = dataset_start
        
        # Check if set Start date to be viable
        enddate = np.array([self.dataset.time[-1].to_numpy() ])
        possible_enddate = enddate.min()
        if end_year > int(np.datetime_as_string(possible_enddate,"Y")):
            print("chosen end year is later than the latest common end date to all of the datasets")
            print("End year is set to ",int(np.datetime_as_string(possible_enddate,"Y")))
            end_year = int(np.datetime_as_string(possible_enddate,"Y"))

        self.start_idx = steps_per_day * sum([366 if isleap(year) else 365 for year in list(range(dataset_start, start_year))])
        self.end_idx = steps_per_day * sum([366 if isleap(year) else 365 for year in list(range(dataset_start, end_year))]) -1

        print("Using years: ",start_year," - ", end_year," (total length: ",self.end_idx - self.start_idx,") (availabe date range: ",np.datetime_as_string(possible_startdate,"Y"),"-",np.datetime_as_string(possible_enddate,"Y"),")")
        print("")
        
    def __len__(self):
        return self.end_idx - self.start_idx

    def __getitem__(self,idx):
        input = self.dataset.isel(time=slice(self.start_idx+idx, self.start_idx+idx + self.temporal_step))[["sea_surface_temperature"]].to_array()
        def format(input):
            time = str(input.time.to_numpy()[0])
            time = torch.tensor(int(time[0:4]+time[5:7]+time[8:10]+time[11:13])) # time in format YYYYMMDDHH  
            if self.coarse_level > 1:
                sst = input.coarsen(latitude=self.coarse_level,longitude=self.coarse_level,boundary='trim').mean()
            return torch.from_numpy(sst.to_numpy()), time
        if self.gt:
            g_truth = self.dataset.isel(time=slice(self.start_idx+idx+1, self.start_idx+idx+1 + self.temporal_step))[["sea_surface_temperature"]].to_array()
            return format(input), format(g_truth)
        else:
            return [format(input)]
        # precompute_temporal_average not implemented

class Trainer():
    '''
    Trainer class for the models
    takes a initialized model class as first parameter and a dictionary of configuration
    '''
    def __init__(self, model, kwargs):
        self.cfg = Attributes(**kwargs)
        self.util = model
        self.model = model.model
        self.mem_log_not_done = True
        self.local_logging=False
        self.scale = 1.0
        self.local_log = {"loss":[],"valid_loss":[]}
        self.mse_all_vars = False
        self.epoch = 0

    def train(self):
        self.setup()
        while self.epoch <= self.cfg.training_epochs:
            # self.pre_epoch()
            self.train_epoch() 
            # self.evaluate_epoch() 
            # self.post_epoch() 
        self.post_training() 

    def set_wandb(self):
        if self.cfg.wandb   : 
            # config_wandb = vars(args).copy()
            # for key in ['notes','tags','wandb']:del config_wandb[key]
            # del config_wandb
            if self.cfg.wandb_resume is not None :
                wandb_run = wandb.init(project=self.cfg.model_type + " - " +self.cfg.model_version, 
                    config=self.cfg.__dict__,
                    notes=self.cfg.notes,
                    tags=self.cfg.tags,
                    resume="must",
                    id=self.cfg.wandb_resume)
            else:
                wandb_run = wandb.init(project=self.cfg.model_type + " - " +self.cfg.model_version, 
                    config=self.cfg.__dict__,
                    notes=self.cfg.notes,
                    tags=self.cfg.tags)
            # create checkpoint folder for run name
            new_save_path = os.path.join(self.cfg.save_path,wandb_run.name)
            os.mkdir(new_save_path)
            self.cfg.save_path = new_save_path
        else : 
            wandb_run = None
            if self.cfg.film_gen_type: film_gen_str = "_"+self.cfg.film_gen_type
            else:                  film_gen_str = ""
            new_save_path = os.path.join(self.cfg.save_path,self.cfg.model_type+"_"+self.cfg.model_version+film_gen_str+"_"+self.cfg.timestr)
            os.mkdir(new_save_path)
            self.cfg.save_path = new_save_path
            print("")
            print("no wandb")

    def train_epoch(self):
        self.iter = 0
        batch_loss = 0
        self.mem_log("loading data")
        for i, data in enumerate(self.training_loader):
            if (i+1) % (self.cfg.validation_interval*(self.cfg.accumulation_steps + 1)) == 0:
                self.validation()
            loss = 0
            discount_factor = 1
            with amp.autocast(self.cfg.enable_amp):
                for step in range(self.cfg.multi_step_training+1):
                    if step == 0 : input = self.util.normalise(data[step][0]).to(self.util.device)
                    else: input = output
                    output, gt = self.model_forward(input,data,step)
                    
                    if step % (self.cfg.training_step_skip+1) == 0:
                        loss = loss + self.get_loss(output, gt)/(self.cfg.multi_step_training+1)/self.cfg.batch_size #*discount_factor**step
                    
                loss = loss / (self.cfg.accumulation_steps+1)
                # only for logging the loss for the batch
                batch_loss += loss.item()
            
            #backward
            self.mem_log("backward pass")
            if self.cfg.enable_amp:
                self.gscaler.scale(loss).backward()
            else:
                loss.backward()

            # Adjust learning weights
            if ((i + 1) % (self.cfg.accumulation_steps + 1) == 0) or (i + 1 == len(self.training_loader)):
                # Update Optimizer
                self.mem_log("optimizer step",fin=True)
                if self.cfg.enable_amp:
                    self.gscaler.step(self.optimizer)
                    self.gscaler.update()
                else:
                    self.optimizer.step()
                self.model.zero_grad()

                # logging
                self.iter += 1
                self.iter_log(batch_loss,scale=None)
                batch_loss = 0
                
                
        # end of epoch
        self.epoch += 1
        print("End of epoch ",self.epoch)
        self.save_checkpoint()
        
    def model_forward(self,input,data,step):
        if self.cfg.model_version == "film" :
            input_sst  = self.util.normalise_film(data[step+1][1]).to(self.util.device)
            gt = self.util.normalise(data[step+1][0]).to(self.util.device)
            
            self.mem_log("forward pass")
            outputs = self.model(input,input_sst,self.scale)
        elif self.cfg.model_type == "mae":
            gt = input
            self.mem_log("forward pass")
            outputs = self.model(input,np.random.uniform(0.4,0.8))
        else:
            gt = self.util.normalise(data[step+1][0]).to(self.util.device)
            self.mem_log("forward pass")
            outputs = self.model(input)
        return outputs, gt

    def training_iteration(self):
        pass

    def setup(self):
        LOG.info("Save path: %s", self.cfg.save_path)
        self.set_wandb()
        if self.cfg.enable_amp == True:
            self.gscaler = amp.GradScaler()
        self.create_loss()
        self.create_optimizer()
        self.create_sheduler()
        self.ready_model()
        self.set_dataloader()

    def ready_model(self):
        self.util.load_model(self.util.checkpoint_path)
        self.model.train()
        self.util.load_statistics()
        self.util.set_seed(42)    

    def create_sheduler(self):
        # Scheduler
        if self.cfg.scheduler_type == 'ReduceLROnPlateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.2, patience=5, mode='min')
        elif self.cfg.scheduler_type == 'CosineAnnealingLR':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.cfg.scheduler_horizon)
        elif self.cfg.scheduler_type == 'CosineAnnealingWarmRestarts':
            self.scheduler =  torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer,T_0=self.cfg.scheduler_horizon)
        else:
            self.scheduler = None
    
    def step_scheduler(self,valid_mean):
        if self.cfg.scheduler_type == 'ReduceLROnPlateau':
            self.scheduler.step(valid_mean)
        elif self.cfg.scheduler_type == 'CosineAnnealingLR':
            self.scheduler.step()
            if (self.epoch*len(self.dataset)+self.iter*self.cfg.batch_size) >= self.cfg.scheduler_horizon:
                LOG.info("Terminating training after reaching params.max_epochs while LR scheduler is set to CosineAnnealingLR") 
        elif self.cfg.scheduler_type == 'CosineAnnealingWarmRestarts':
            self.scheduler.step(self.epoch*len(self.dataset)+self.iter*self.cfg.batch_size)
        
    def create_optimizer(self):
        self.optimizer = torch.optim.Adam(self.util.get_parameters(), lr=self.cfg.learning_rate)# store the optimizer and scheduler in the model class
        
    def create_loss(self):
        if self.cfg.loss_fn == "CosineMSE":
            self.loss_fn = CosineMSELoss(reduction='mean')
        elif self.cfg.loss_fn == "L2Sphere":
            self.loss_fn = L2Sphere(relative=True, squared=True)
        elif self.cfg.loss_fn == "NormalCRPS":
            self.loss_fn = NormalCRPS()
        else:
            self.loss_fn = torch.nn.MSELoss()

    def set_dataloader(self):
        if self.cfg.model_type == "mae":
            print("Trainig Data:")
            self.dataset = SST_galvani(
                path=self.cfg.trainingdata_path, 
                start_year=self.cfg.trainingset_start_year,
                end_year=self.cfg.trainingset_end_year,
                temporal_step=self.cfg.temporal_step
            )
            print("Validation Data:")
            self.dataset_validation = SST_galvani(
                path=self.cfg.trainingdata_path, 
                start_year=self.cfg.validationset_start_year,
                end_year=self.cfg.validationset_end_year,
                temporal_step=self.cfg.temporal_step)
        else:
            if self.cfg.model_version == 'film':
                sst = True
            else:
                sst = False

            print("Trainig Data:")
            self.dataset = ERA5_galvani(
                self.util,
                path=self.cfg.trainingdata_path, 
                start_year=self.cfg.trainingset_start_year,
                end_year=self.cfg.trainingset_end_year,
                auto_regressive_steps=self.cfg.multi_step_training,
                sst=sst
            )
            print("Validation Data:")
            self.dataset_validation = ERA5_galvani(
                self.util,
                path=self.cfg.trainingdata_path, 
                start_year=self.cfg.validationset_start_year,
                end_year=self.cfg.validationset_end_year,
                auto_regressive_steps=self.cfg.multi_step_validation,
                sst=sst
            )
        self.training_loader = DataLoader(self.dataset,shuffle=True,num_workers=self.cfg.training_workers, batch_size=self.cfg.batch_size)
        self.validation_loader = DataLoader(self.dataset_validation,shuffle=True,num_workers=self.cfg.training_workers, batch_size=self.cfg.batch_size)

        return #training_loader, validation_loader
    
    # train loop
    def get_loss(self,output,gt):
        if self.cfg.loss_fn == "NormalCRPS":
            mu = torch.nan_to_num(output[0][0],nan=0.0)
            gt = torch.nan_to_num(gt,nan=0.0)
            std = torch.nan_to_num(output[0][1],nan=1.0)
            return self.loss_fn(mu, std, gt) 
        else:
            return self.loss_fn(output,gt)
    
    def validation(self):
        val_loss = {}
        val_log  = {}
        loss_fn_pervar = torch.nn.MSELoss(reduction='none')
        self.model.eval()
        with torch.no_grad():
            # For loop over validation dataset, calculates the validation loss mean for number of kwargs["validation_epochs"]
            loss_pervar_list = []
            for val_idx, val_data in enumerate(self.validation_loader):
                # Calculates the validation loss for autoregressive model evaluation
                # if self.auto_regressive_steps = 0 the dataloader only outputs 2 datapoint 
                # and the for loop only runs once (calculating the ordinary validation loss with no auto regressive evaluation
                val_input_era5 = None
                for val_step in range(self.cfg.multi_step_validation+1):
                    
                    if val_step == 0 : input = self.util.normalise(val_data[val_step][0]).to(self.util.device)
                    else: input = output
                    output, gt = self.model_forward(input,val_data,val_step)
                    
                    val_loss_value = self.get_loss(output,gt)/ self.cfg.batch_size

                    # loss for each variable
                    if self.cfg.advanced_logging and self.mse_all_vars  and val_step == 0: # !! only for first multi validation step, could include next step with -> ... -> ... in print statement on same line
                        val_g_truth_era5 = self.normalise(val_data[val_step+1][0]).to(self.util.device)
                        loss_pervar_list.append(loss_fn_pervar(output, val_g_truth_era5).mean(dim=(0,2,3)) / self.cfg.batch_size)
                    
                    if val_idx == 0: 
                        val_loss["validation loss step={}".format(val_step)] = [val_loss_value.cpu()] #kwargs["validation_epochs"]
                    else:
                        val_loss["validation loss step={}".format(val_step)].append(val_loss_value.cpu())

                # end of validation 
                if val_idx > self.cfg.validation_epochs:
                    for k in val_loss.keys():
                        val_loss_array      = np.array(val_loss[k])
                        val_log[k]          = round(val_loss_array.mean(),5)
                        val_log["std " + k] = round(val_loss_array.std(),5)
                    break
            
            #scheduler
            valid_mean = list(val_log.values())[0]
            self.step_scheduler(valid_mean)

            # change scale value based on validation loss
            # if valid_mean < kwargs["val_loss_threshold"] and scale < 1.0:
            if self.scale < 1.0 and self.cfg.model_version == "film":
                val_log["scale"] = self.scale
                self.scale = self.scale + 0.002 # 5e-5 #

            self.valid_log(val_log,loss_pervar_list)

        # save model and training statistics for checkpointing
        if (self.iter+1) % (self.cfg.validation_interval*self.cfg.save_checkpoint_interval) == 0:
            self.save_checkpoint()
            if self.cfg.advanced_logging and self.cfg.model_version == "film":
                gamma_np = self.model.gamma.cpu().numpy()
                beta_np  = self.model.beta.cpu().numpy()
                np.save(os.path.join( self.save_path,"gamma_{}.npy".format(i)),gamma_np)
                np.save(os.path.join( self.save_path,"beta_{}.npy".format(i)),beta_np)
                print("gamma values mean : ",round(gamma_np.mean(),5),"+/-",round(gamma_np.std(),5))
                print("beta values mean  : ",round(beta_np.mean(),5),"+/-",round(beta_np.std(),5))
        if self.cfg.model_version == "film" :
            self.model.film_gen.train()
        else:
            self.model.train()
                
    def valid_log(self,val_log,loss_pervar_list):
        # little complicated console logging - looks nicer than LOG.info(str(val_log))
        print("-- validation after ",self.iter*self.cfg.batch_size, "training examples")
        val_log_keys = list(val_log.keys())
        for log_idx in range(0,self.cfg.multi_step_validation*2+1,2): 
            LOG.info(val_log_keys[log_idx] + " : " + str(val_log[val_log_keys[log_idx]]) 
                        + " +/- " + str(val_log[val_log_keys[log_idx+1]]))
        
        # log to local file
        # self.val_means[log_idx].append(val_log[val_log_keys[log_idx]]) ## error here
        # self.val_stds[log_idx].append(val_log[val_log_keys[log_idx+1]]) 
        
        # log scheduler
        if self.scheduler is not None and self.scheduler != "None": 
            lr = self.scheduler.get_last_lr()[0]
            val_log["learning rate"] = lr
        
        # MSE for all variables
        if self.cfg.advanced_logging and self.mse_all_vars:
            print("MSE for each variable:")
            val_loss_value_pervar = torch.stack(loss_pervar_list).mean(dim=0)
            for idx_var,var_name in enumerate(self.ordering):
                print("    ",var_name," = ",round(val_loss_value_pervar[idx_var].item(),5))
        
        # log film parameters gamma/beta
        if self.cfg.model_version == "film":
            gamma_np = self.model.gamma.cpu().numpy()
            beta_np  = self.model.beta.cpu().numpy()
            print("gamma values mean : ",round(gamma_np.mean(),5),"+/-",round(gamma_np.std(),5))
            print("beta values mean  : ",round(beta_np.mean(),5),"+/-",round(beta_np.std(),5))
        
        # wandb
        if self.cfg.wandb :
            wandb.log(val_log,commit=False)

    def mem_log(self,str,fin=False):
        if self.cfg.advanced_logging and self.mem_log_not_done:
            print("VRAM used before "+str+" : ",round(torch.cuda.memory_allocated(self.util.device)/10**9,2),
                  " GB, reserved: ",round(torch.cuda.memory_reserved(self.util.device)/10**9,2)," GB")
            if fin:
                self.mem_log_not_done = False 
            
    def iter_log(self,batch_loss,scale=None):
        if self.cfg.advanced_logging:
            if self.local_logging : self.local_log["losses"].append(round(batch_loss,5))
            if self.cfg.wandb:
                wandb.log({"loss": round(batch_loss,5) })
            if self.cfg.advanced_logging:
                if scale is not None:
                    print("Iteration: ", self.iter, " Loss: ", round(batch_loss,5)," - scale: ",round(scale,5))
                else :
                    print("Iteration: ", self.iter, " Loss: ", round(batch_loss,5))
    
    def save_checkpoint(self):
        save_file ="checkpoint_"+self.cfg.model_type+"_"+self.cfg.model_version+"_"+str(self.cfg.film_gen_type)+"_iter={}_epoch={}.pkl".format(self.iter,self.epoch)
        total_save_path = os.path.join( self.cfg.save_path,save_file)
        LOG.info("Saving checkpoint to %s",total_save_path)
        if self.local_logging : 
            print(" -> saving to : ",self.cfg.save_path)
            np.save(os.path.join( self.cfg.save_path,"val_means.npy"),self.val_means)
            np.save(os.path.join( self.cfg.save_path,"val_stds.npy"),self.val_stds)
            np.save(os.path.join( self.cfg.save_path,"losses.npy"),self.losses)

        if save_file is None: save_file ="checkpoint_"+self.cfg.timestr+"_final.pkl"
        save_dict = {
            "model_state":self.model.state_dict(),
            "epoch":self.epoch,
            "iter":self.iter,
            "optimizer_state_dict":self.optimizer.state_dict(),
            "hyperparameters": self.cfg.__dict__
            }
        if self.scheduler: save_dict["scheduler_state_dict"]= self.scheduler.state_dict()
        torch.save(save_dict,total_save_path)

                
    def test_model_speed(self):
        with Timer("Model speed test"):
            for i in range(100):
                # data_era5 = torch.randn(1,2,721,1440)
                data_sst = torch.randn(1,2,721,1440)
                self.model_forward(input,data_sst,0)
    
    def test_dataloader_speed(self):
        self.set_dataloader()
        with Timer("Dataloader speed test"):
            for i, data in enumerate(self.training_loader):
                pass

    def evaluate_model(self, checkpoint_list,save_path):
        """Evaluate model using checkpoint list"""
        with torch.no_grad():
            self.set_dataloader()
            self.util.load_statistics()
            self.util.set_seed(42)  
            for cp_idx, checkpoint in enumerate(checkpoint_list):
                cp = self.util.load_model(checkpoint)
                self.save_path = save_path
                for i, data in enumerate(self.validation_loader):
                    for step in range(self.cfg.multi_step_validation+1):
                        if step == 0 : input = self.util.normalise(data[step][0]).to(self.util.device)
                        else: input = output
                        output, gt = self.model_forward(input,data,step)
                        self.util.plot(output, gt, int(cp["iter"])*int(cp["epoch"]),checkpoint)
                        break
                
            print("done")
            
        


class MAETrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)









































def train_test(kwargs):
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
    # dataset_coarsen = ERA5_galvani_coarsen(
    #     params,
    #     start_year=1990,
    #     end_year=2000,
    # )
    for i in range(0,10):
        print("--- Workers: ", i, " ---")
        # coarsen_loader = DataLoader(dataset_coarsen,shuffle=True,num_workers=i, batch_size=kwargs["batch_size"])
        masked_loader = DataLoader(dataset_masked,shuffle=True,num_workers=i, batch_size=kwargs["batch_size"])
