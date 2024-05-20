import torch
from torch.utils.data import RandomSampler,BatchSampler, DataLoader, Dataset#, IterableDataset
import torch.cuda.amp as amp
from calendar import isleap
import xarray as xr
import numpy as np
import os
import sys
# from .sfno.model import get_model
from .sfno.sfnonet import GCN
from .data import SST_galvani, ERA5_galvani
import wandb
from time import time

# BatchSampler(drop_last=True)

from S2S_on_SFNO.Models.provenance import system_monitor
from .losses import CosineMSELoss, L2Sphere, NormalCRPS
from S2S_on_SFNO.utils import Timer, Attributes

from ..utils import LocalLog

import logging
LOG = logging.getLogger('S2S_on_SFNO')

# DataParallel

import torch.distributed as dist
# import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import  get_rank


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
        self.local_logging=True
        self.scale = 1.0
        self.mse_all_vars = self.cfg.model_type == "sfno"
        self.epoch = 0
        self.step = 0

    def train(self):
        self.setup()
        while self.epoch < self.cfg.training_epochs:
            self.pre_epoch()
            self.train_epoch() 
            self.post_epoch() 
        self.finalise()

    def set_logger(self):
        if self.cfg.rank == 0:
            print("")
            print("logger settings:")
            if self.cfg.wandb   : 
                # config_wandb = vars(args).copy()
                # for key in ['notes','tags','wandb']:del config_wandb[key]
                # del config_wandb
                if os.environ.get("SCRATCH") is not None:
                    wandb_dir = os.path.join(os.environ["SCRATCH"],"wandb")
                    if not os.path.exists(wandb_dir): os.mkdir(wandb_dir)
                elif os.path.exists("/mnt/qb/work2/goswami0/gkd965/wandb"):
                    wandb_dir = "/mnt/qb/work2/goswami0/gkd965/wandb"
                else:
                    wandb_dir = "./wandb"
                if self.cfg.wandb_resume is not None :
                    wandb_run = wandb.init(project=self.cfg.model_type + " - " +self.cfg.model_version, 
                        config=self.cfg.__dict__,
                        notes=self.cfg.notes,
                        tags=self.cfg.tags,
                        resume="must",
                        id=self.cfg.wandb_resume,
                        dir=wandb_dir,
                        )
                else:
                    wandb_run = wandb.init(project=self.cfg.model_type + " - " +self.cfg.model_version, 
                        config=self.cfg.__dict__,
                        notes=self.cfg.notes,
                        tags=self.cfg.tags,
                        dir=wandb_dir,
                        )
                # create checkpoint folder for run name
                if self.cfg.jobID is not None:  file_name = wandb_run.name+"-sID{"+self.cfg.jobID+"}"
                else:                           file_name = wandb_run.name
                new_save_path = os.path.join(self.cfg.save_path,file_name)
                os.mkdir(new_save_path)
                self.cfg.save_path = new_save_path
            else : 
                wandb_run = None
                if self.cfg.film_gen_type: film_gen_str = "_"+self.cfg.film_gen_type
                else:                  film_gen_str = ""
                file_name = self.cfg.model_type+"_"+self.cfg.model_version+film_gen_str+"_"+self.cfg.timestr
                if self.cfg.jobID is not None:  file_name = file_name+"-{sID"+self.cfg.jobID+"}"
                new_save_path = os.path.join(self.cfg.save_path,file_name)
                os.mkdir(new_save_path)
                self.cfg.save_path = new_save_path
                print("    no wandb")
        else:
            print("skip this process (",self.cfg.rank,") in logging")
        
        print("    Save path: %s", self.cfg.save_path)
        self.local_log = LocalLog(self.local_logging,self.cfg.save_path)

    def train_epoch(self):
        self.iter = 0
        batch_loss = 0

        self.mem_log("loading data")
        for i, data in enumerate(self.training_loader):
            if self.cfg.validation_interval > 0 and (self.iter+1) % (self.cfg.validation_interval)  == 0:
                self.validation()
            loss = 0
            discount_factor = 0.99
            with amp.autocast(self.cfg.enable_amp):
                for step in range(self.cfg.multi_step_training+1):
                    if step == 0 : input = self.util.normalise(data[step][0]).to(self.util.device)
                    else: input = output
                    output, gt = self.model_forward(input,data,step)
                    
                    if step % (self.cfg.training_step_skip+1) == 0:
                        loss = loss + self.get_loss(output, gt)/(self.cfg.multi_step_training+1)/self.cfg.batch_size *discount_factor**step
                    
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
                self.step = self.iter*self.cfg.batch_size+len(self.dataset)*self.epoch
                self.iter_log(batch_loss,scale=None)
                batch_loss = 0
  
        # end of epoch
    
    def pre_epoch(self):
        self.util.set_seed(self.epoch)
        if self.cfg.ddp:
            self.training_loader.sampler.set_epoch(self.epoch)
            self.validation_loader.sampler.set_epoch(self.epoch)   
    
    def post_epoch(self):
        self.epoch += 1
        self.iter = 0
        print("End of epoch ",self.epoch)
        self.validation()
        self.save_checkpoint()
        self.local_log.save("training_log_epoch{}.npy".format(self.epoch))

    def model_forward(self,input,data,step):
        self.mem_log("forward pass")
        if self.cfg.model_type == 'sfno' and self.cfg.model_version == "film" :
            if self.cfg.film_gen_type == "mae" and self.cfg.cls is not None:
                # class token doesn't need normalisation
                input_sst  = data[step][1].to(self.util.device)
            else:
                input_sst  = self.util.normalise_film(data[step+1][1]).to(self.util.device)
            gt = self.util.normalise(data[step+1][0]).to(self.util.device)
            outputs = self.model(input,input_sst,self.scale)
        elif self.cfg.model_type == "mae":
            if self.cfg.model_version == "lin-probe":
                gt = data[step+1][0].to(self.util.device)
                outputs = self.model(input)
            else:
                gt = input # input is already normalised
                outputs = self.model(input,np.random.uniform(0.4,0.8)) # outputs = (mean, std), mask, cls
        else:
            gt = self.util.normalise(data[step+1][0]).to(self.util.device)
            outputs = self.model(input)
        return outputs, gt

    def training_iteration(self):
        pass

    def setup(self):
        self.set_logger()
        if self.cfg.enable_amp == True:
            self.gscaler = amp.GradScaler()
        self.create_loss()
        self.create_optimizer()
        self.create_sheduler()
        self.ready_model()
        self.set_dataloader()

    def finalise(self):
        if not self.cfg.debug:
            self.local_log.save("training_log.npy")
            self.save_checkpoint()
            wandb.finish()
        sys.exit(0)

    def ready_model(self):
        if self.cfg.ddp: torch.cuda.set_device(self.cfg.rank)
        self.util.load_model(self.util.checkpoint_path)
        self.model.train()
        self.util.load_statistics() 
        if self.cfg.ddp:
            self.model = DDP(self.model,device_ids=[self.util.device])
            torch.cuda.empty_cache()

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
            if (self.iter/self.cfg.validation_interval) >= self.cfg.scheduler_horizon:
                LOG.info("Terminating training after reaching params.max_epochs while LR scheduler is set to CosineAnnealingLR") 
                self.finalise()
        elif self.cfg.scheduler_type == 'CosineAnnealingWarmRestarts':
            self.scheduler.step()
        
    def create_optimizer(self):
        if self.cfg.optimizer == "Adam":
            self.optimizer = torch.optim.Adam(self.util.get_parameters(), lr=self.cfg.learning_rate)# store the optimizer and scheduler in the model class
        elif self.cfg.optimizer == "SGD":
            self.optimizer = torch.optim.SGD(self.util.get_parameters(), lr=self.cfg.learning_rate, momentum=0.9)
        elif self.cfg.optimizer == "LBFGS":
            self.optimizer = torch.optim.LBFGS(self.util.get_parameters())# store the optimizer and scheduler in the model class

    def create_loss(self):
        if self.cfg.loss_fn == "CosineMSE":
            self.loss_fn = CosineMSELoss(reduction=self.cfg.loss_reduction)
        elif self.cfg.loss_fn == "L2Sphere":
            self.loss_fn = L2Sphere(relative=True, squared=True,reduction=self.cfg.loss_reduction)
        elif self.cfg.loss_fn == "NormalCRPS":
            self.loss_fn = NormalCRPS(reduction=self.cfg.loss_reduction)
        else:
            self.loss_fn = torch.nn.MSELoss()

    def set_dataloader(self):
        if self.cfg.model_type == "mae":
            if self.cfg.model_version =="lin-probe":
                oni = True
            else:
                oni = self.cfg.oni
            print("Trainig Data:")
            self.dataset = SST_galvani(
                path=self.cfg.trainingdata_path, 
                start_year=self.cfg.trainingset_start_year,
                end_year=self.cfg.trainingset_end_year,
                temporal_step=self.cfg.temporal_step,
                cls=self.cfg.cls,
                oni=oni,
            )
            print("Validation Data:")
            self.dataset_validation = SST_galvani(
                path=self.cfg.trainingdata_path, 
                start_year=self.cfg.validationset_start_year,
                end_year=self.cfg.validationset_end_year,
                temporal_step=self.cfg.temporal_step,
                cls=self.cfg.cls,
                oni=oni,
                )
        else:
            if self.cfg.model_version == 'film' and self.cfg.cls is None:
                sst = True
            else:
                sst = False

            print("Trainig Data:")
            self.dataset = ERA5_galvani(
                self.util,
                path=self.cfg.trainingdata_path, 
                u100_path=self.cfg.trainingdata_u100_path,
                v100_path=self.cfg.trainingdata_v100_path,
                start_year=self.cfg.trainingset_start_year,
                end_year=self.cfg.trainingset_end_year,
                auto_regressive_steps=self.cfg.multi_step_training,
                sst=sst,
                temporal_step=self.cfg.temporal_step,
                past_sst = self.cfg.past_sst,
                cls=self.cfg.cls,
            )
            print("Validation Data:")
            self.dataset_validation = ERA5_galvani(
                self.util,
                path=self.cfg.trainingdata_path, 
                u100_path=self.cfg.trainingdata_u100_path,
                v100_path=self.cfg.trainingdata_v100_path,
                start_year=self.cfg.validationset_start_year,
                end_year=self.cfg.validationset_end_year,
                auto_regressive_steps=self.cfg.multi_step_validation,
                sst=sst,
                temporal_step=self.cfg.temporal_step,
                past_sst = self.cfg.past_sst,
                cls=self.cfg.cls,
            )
        shuffle= not self.cfg.no_shuffle
        if self.cfg.ddp:
            self.training_loader = DataLoader(self.dataset,num_workers=self.cfg.training_workers, batch_size=self.cfg.batch_size,shuffle=False,pin_memory=True,sampler=DistributedSampler(self.dataset,shuffle=shuffle))
            self.validation_loader = DataLoader(self.dataset_validation,num_workers=self.cfg.training_workers, batch_size=self.cfg.batch_size,shuffle=False,pin_memory=True,sampler=DistributedSampler(self.dataset_validation,shuffle=shuffle))

        else:
            self.training_loader = DataLoader(self.dataset,shuffle=shuffle,num_workers=self.cfg.training_workers, batch_size=self.cfg.batch_size)
            self.validation_loader = DataLoader(self.dataset_validation,shuffle=shuffle,num_workers=self.cfg.training_workers, batch_size=self.cfg.batch_size)

        return #training_loader, validation_loader
    
    # train loop
    def get_loss(self,output,gt):
        if self.cfg.loss_fn == "NormalCRPS":
            mu = output[0][0]
            std =output[0][1]
            mask = output[1][1]
            return self.loss_fn(mu, std, gt,mask) 
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
                        val_g_truth_era5 = self.util.normalise(val_data[val_step+1][0]).to(self.util.device)
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
        
        # change scale value based on validation loss
        # if valid_mean < kwargs["val_loss_threshold"] and scale < 1.0:
        if self.scale < 1.0 and self.cfg.model_version == "film":
            val_log["scale"] = self.scale
            self.scale = self.scale + 0.002 # 5e-5 #

        self.valid_log(val_log,loss_pervar_list)

        #scheduler
        valid_mean = list(val_log.values())[0]
        self.step_scheduler(valid_mean)

        

        # save model and training statistics for checkpointing
        if (self.iter+1) % (self.cfg.validation_interval*self.cfg.save_checkpoint_interval) == 0 and self.cfg.save_checkpoint_interval > 0:
            self.save_checkpoint()
        
        # return to training
        if self.cfg.model_version == "film" :
            self.model.film_gen.train()
        else:
            self.model.train()
                
    def valid_log(self,val_log,loss_pervar_list):
        if self.cfg.rank != 0: return
        # little complicated console logging - looks nicer than LOG.info(str(val_log))
        if self.cfg.multi_step_validation > 0:multistep_notice = " (steps=... mutli-step-validation, each step an auto-regressive 6h-step)"
        else: multistep_notice = ""                                             
        print("-- validation after ",self.step, "training examples "+multistep_notice)
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
        if self.cfg.advanced_logging and self.mse_all_vars and self.cfg.model_type != "mae":
            print("MSE for each variable:")
            val_loss_value_pervar = torch.stack(loss_pervar_list).mean(dim=0)
            for idx_var,var_name in enumerate(self.util.ordering):
                print("    ",var_name," = ",round(val_loss_value_pervar[idx_var].item(),5))
                val_log["MSE "+var_name] = round(val_loss_value_pervar[idx_var].item(),5)
        
        # log film parameters gamma/beta
        if self.cfg.advanced_logging and self.cfg.model_version == "film":
            gamma_np = self.model.gamma.cpu().clone().detach().numpy()
            beta_np  = self.model.beta.cpu().clone().detach().numpy()
            print("gamma values mean : ",round(gamma_np.mean(),6),"+/-",round(gamma_np.std(),6))
            print("beta values mean  : ",round(beta_np.mean(),6),"+/-",round(beta_np.std(),6))
            val_log["gamma"] = round(gamma_np.mean(),6)
            val_log["beta"]  = round(beta_np.mean(),6)
            val_log["gamma_std"] = round(gamma_np.std(),6)
            val_log["beta_std"]  = round(beta_np.std(),6)
        
        # local logging
        self.local_log.log(val_log)
            
        # wandb
        if self.cfg.wandb :
            wandb.log(val_log,commit=False)

    def mem_log(self,str,fin=False):
        if self.cfg.rank != 0: return
        if self.cfg.advanced_logging and self.mem_log_not_done:
            print("VRAM used before "+str+" : ",round(torch.cuda.memory_allocated(self.util.device)/10**9,2),
                  " GB, reserved: ",round(torch.cuda.memory_reserved(self.util.device)/10**9,2)," GB")
            if fin:
                self.mem_log_not_done = False 
            
    def iter_log(self,batch_loss,scale=None):
        if self.cfg.rank != 0: return
        # local logging
        self.local_log.log({"loss": round(batch_loss,6),"step":self.step})

        if self.cfg.wandb:
            wandb.log({"loss": round(batch_loss,5),"step":self.step })
        if self.cfg.advanced_logging:
            if scale is not None:
                print("After ", self.step, " Samples - Loss: ", round(batch_loss,5)," - scale: ",round(scale,5))
            else :
                print("After ", self.step, " Samples - Loss: ", round(batch_loss,5))
    
    def save_checkpoint(self):
        if self.cfg.rank != 0: return
        save_file ="checkpoint_"+self.cfg.model_type+"_"+self.cfg.model_version+"_"+str(self.cfg.film_gen_type)+"_iter={}_epoch={}.pkl".format(self.iter,self.epoch)
        total_save_path = os.path.join( self.cfg.save_path,save_file)
        LOG.info("Saving checkpoint to %s",total_save_path)
       
        if save_file is None: save_file ="checkpoint_"+self.cfg.timestr+"_final.pkl"
        if self.cfg.ddp:
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
        save_dict = {
            "model_state":model_state,
            "epoch":self.epoch,
            "iter":self.iter,
            "optimizer_state_dict":self.optimizer.state_dict(),
            "hyperparameters": self.cfg.__dict__
            }
        if self.scheduler: save_dict["scheduler_state_dict"]= self.scheduler.state_dict()
        # wanted to see if ddp issue comes from here
        # for k,v in save_dict:
        #     print("")
        #     print("    ",k," : ",v)
        # torch.save(save_dict,total_save_path)

        # Gamma Beta
        if self.cfg.advanced_logging and self.cfg.model_version == "film":
            gamma_np = self.model.gamma.cpu().clone().detach().numpy()
            beta_np  = self.model.beta.cpu().clone().detach().numpy()
            np.save(os.path.join( self.cfg.save_path,"gamma_{}.npy".format(self.step)),gamma_np)
            np.save(os.path.join( self.cfg.save_path,"beta_{}.npy".format(self.step)),beta_np)
        
        print("save done")

    def save_forecast(self):
        self.ready_model()
        self.set_dataloader()
        self.model.eval()
        self.mem_log("loading data")
        self.output_data = []
        for i, data in enumerate(self.validation_loader):
            with amp.autocast(self.cfg.enable_amp):
                for step in range(self.cfg.multi_step_validation+1):
                    if step == 0 : input = self.util.normalise(data[step][0]).to(self.util.device)
                    else: input = output
                    self.mem_log("loading data")
                    output, gt = self.model_forward(input,data,step)
                    self.output_data += [*(output.cpu().numpy())]
           
            self.mem_log("fin",fin=True)
            if i % 100 == 0:
                system_monitor(printout=True,pids=[os.getpid()],names=["python"])
                # logging
            self.iter += 1
            self.step = self.iter*self.cfg.batch_size+len(self.dataset)*self.epoch
  

    def save_to_netcdf(self):
        '''
        Take the output_data from a inference run and save it as a netcdf file that conforms with the weatherbench2 forcast format, with coordinates:
        latitude: float64, level: int32, longitude: float64, prediction_timedelta: timedelta64[ns], time: datetime64[ns]
        '''
        data_vars = {}
        for i in range(0):
        # dataset = xr.Dataset(
        #     data
        # )
            pass
                
    def test_model_speed(self):
        with Timer("Model speed test"):
            for i in range(100):
                # data_era5 = torch.randn(1,2,721,1440)
                if self.cfg.model_type == "mae":
                    if self.cfg.model_version == "lin-probe":
                        nino = torch.randn((self.cfg.batch_size,1))
                        cls = torch.randn((self.cfg.batch_size,512))
                        data = torch.tensor([cls,nino])
                    else:
                        data = torch.randn(1,2,721,1440)
                        data_sst = torch.randn(1,2,721,1440)
                self.model_forward(data[0][0],data,0)
    
    def gen_test_data(self,num):
        # nees multistep support
        # create all data at once no for loops and directly on device
        if self.cfg.model_type == "mae":
            if self.cfg.model_version == "lin-probe":
                dataset = []
                for i in range(num):
                    nino = torch.randn((self.cfg.batch_size,1))
                    cls = torch.randn((self.cfg.batch_size,512))
                    data = [[cls],[nino]]
                    dataset.append(data)
                return dataset
            elif self.cfg.model_type == "sfno":
                if self.cfg.film_gen_type == "mae" and self.cfg.cls is not None:
                    dataset = []
                    for i in range(num):
                        data = []
                        for i in range(0, self.auto_regressive_steps+2):
                            era5 = torch.randn((self.cfg.batch_size,73,180,360))
                            cls = torch.randn((self.cfg.batch_size,512))
                            data.append([era5, cls])
                        dataset.append(data)
            else:
                data = torch.randn(1,2,721,1440)
                data_sst = torch.randn(1,2,721,1440)
    
    def save_data(self):
        self.set_dataloader()
        print("saving data from dataloader")
        dataset = []
        percent = 0
        for i, data in enumerate(self.training_loader):
            if i % (len(self.training_loader)//10) == 0:
                print("done ",percent," %")
                percent+=10
            dataset.append(data[0][0].item())
        print("done dataloader test")
        np.save(os.path.join(self.cfg.save_path,"oni.npy"),np.array(dataset))
        return
    
    def test_dataloader_speed(self):
        self.set_dataloader()
        print("testing dataloader speed")
        with Timer("Dataloader speed test",divisor=self.cfg.num_iterations):
            for i, data in enumerate(self.training_loader):
                if i > self.cfg.num_iterations: break
        print("done dataloader test")
        return
    
    def test_performance(self):
        self.test_batch_size(self.cfg.num_iterations,self.cfg.batch_size_step)
        # self.test_model_speed()
        self.test_dataloader_speed()

    def test_batch_size(self,
        num_iterations: int = 5,
        batch_size_step: int = 1,
    ) -> int:
        # resetting batch size in cfg is an issue for GCNs
        print("Test batch size")
        self.cfg.validation_interval = -1
        self.cfg.save_checkpoint_interval = -1
        self.set_logger()
        self.create_loss()
        self.create_optimizer()
        self.create_sheduler()
        self.ready_model()
        while True:
            try:
                self.training_loader = self.gen_test_data(num_iterations)
                self.dataset = self.training_loader
                print(f"Testing batch size {self.cfg.batch_size}")
                with Timer("\truntime "+str(self.cfg.batch_size),divisor=num_iterations):
                    self.train_epoch()
                self.cfg.batch_size += batch_size_step
                batch_size_step = batch_size_step*2
            except RuntimeError:
                print(f"\tOOM at batch size {self.cfg.batch_size}")
                batch_size_step = batch_size_step//2
                break
            if batch_size_step == 0:
                break
        torch.cuda.empty_cache()
        print(f"Final batch size {self.cfg.batch_size}")
        return self.cfg.batch_size



    # only plots MAE results at the moment
    def evaluate_model(self, checkpoint_list,save_path):
        """Evaluate model using checkpoint list"""
        # self.cfg.batch_size = 1
        plot = False
        with torch.no_grad():
            self.set_dataloader()
            self.util.load_statistics()
            self.util.set_seed(42)  
            for cp_idx, checkpoint in enumerate(checkpoint_list):
                cp_name = checkpoint.split("/")[-1].split(".")[0]
                cp = self.util.load_model(checkpoint)
                # self.save_path = save_path
                for i, data in enumerate(self.validation_loader):
                    for step in range(1): #range(self.cfg.multi_step_validation+1):
                        if step == 0 : input = self.util.normalise(data[step][0]).to(self.util.device)
                        else: input = output
                        output, gt = self.model_forward(input,data,step)
                        if plot:
                            self.util.plot(output, gt, int(cp["iter"])*int(cp["epoch"]),cp_name,save_path)
                            break
                    if plot: break
                
            print("done")
    
    
    
#     def evaluate_mae_cls():

# class MAETrainer(Trainer):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)









































def train_test(kwargs):
    # model = get_model(kwargs)

    # dataset = ERA5_galvani(
    #     params,
    #     path=kwargs["trainingdata_path"], 
    #     start_year=kwargs["trainingset_start_year"],
    #     end_year=kwargs["trainingset_end_year"])

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

# def test(kwargs):

#     dataset_masked = ERA5_galvani(
#         params,
#         start_year=2000,
#         end_year=2010,
#     )
#     # dataset_coarsen = ERA5_galvani_coarsen(
#     #     params,
#     #     start_year=1990,
#     #     end_year=2000,
#     # )
#     for i in range(0,10):
#         print("--- Workers: ", i, " ---")
#         # coarsen_loader = DataLoader(dataset_coarsen,shuffle=True,num_workers=i, batch_size=kwargs["batch_size"])
#         masked_loader = DataLoader(dataset_masked,shuffle=True,num_workers=i, batch_size=kwargs["batch_size"])
