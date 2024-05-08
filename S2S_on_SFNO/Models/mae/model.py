

from ..models import Model
from .maenet import ContextCast
from torch.utils.data import DataLoader
import torch.cuda.amp as amp
import logging
import wandb
import os
import numpy as np
import sys
import matplotlib.pyplot as plt

ultra_advanced_logging=False
local_logging = False

import torch

from ..train import SST_galvani
from ..losses import CosineMSELoss, L2Sphere, NormalCRPS

LOG = logging.getLogger('S2S_on_SFNO')

class MAE(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # init model
        self.model = ContextCast(data_dim=1,**kwargs)
        # self.params = kwargs
        # self.timestr = kwargs["timestr"]
        # self.assets = kwargs["assets"]
        # self.save_path = kwargs["save_path"]

        if self.resume_checkpoint:
            self.checkpoint_path = self.resume_checkpoint
        else:
            self.checkpoint_path = None
    
    def load_model(self, checkpoint_file):
        
        if checkpoint_file is not None:
            checkpoint = torch.load(checkpoint_file)
            if "model_state" in checkpoint.keys(): weights = checkpoint["model_state"]
            else: weights = checkpoint
            try:
                self.model.load_state_dict(weights)
            except RuntimeError as e:
                LOG.error(e)
                print("--- !! ---")
                print("loading state dict with strict=False, please verify if the right model is loaded and strict=False is desired")
                print("--- !! ---")
                self.model.load_state_dict(weights,strict=False)
        self.model.eval()
        self.model.zero_grad()
        self.model.to(self.device)
        return checkpoint if checkpoint_file is not None else None

    ## common

    def load_statistics(self):
            
        self.means_film = np.load(os.path.join(self.assets, "global_means_sst.npy"))
        self.means_film = self.means_film.astype(np.float32)
        self.stds_film = np.load(os.path.join(self.assets, "global_stds_sst.npy"))
        self.stds_film = self.stds_film.astype(np.float32)
    
    def normalise(self, data, reverse=False):
        """Normalise data using pre-saved global statistics"""
        if reverse:
            new_data = data * self.stds_film + self.means_film
        else:
            new_data = (data - self.means_film) / self.stds_film
        return new_data

    def evaluate_model(self, checkpoint_list,save_path):
        """Evaluate model using checkpoint list"""
        for cp_idx, checkpoint in enumerate(checkpoint_list):
            self.checkpoint_path = checkpoint
            model = self.load_model(self.checkpoint_path)
            model.eval()
            model.to(self.device)
            self.save_path = save_path
            self.validation()
            model.train()

    def plot(self, data, gt, training_examples,checkpoint):
        """Plot data using matplotlib"""
        pred = data[0][0].cpu().numpy().squeeze()
        gt = gt.cpu().numpy().squeeze()
        std = data[0][1].cpu().numpy().squeeze()
        mask = data[1].cpu().numpy()
        vmin = np.min((pred,gt))
        vmax = np.max((pred,gt))
        for time in range(pred.shape[0]):
            fig, ax = plt.subplots(2, 2, figsize=(10, 10))
            ax[0][0].imshow(pred[time],vmin=vmin, vmax=vmax,)
            ax[0][0].set_title("Predicted SST")
            im_gt = ax[0][1].imshow(gt[time],vmin=vmin, vmax=vmax,)
            ax[0][1].set_title("Ground Truth SST")
            ax[1][0].imshow(mask[time])
            ax[1][0].set_title("Mask")
            ax[1][1].imshow(std[time])
            img_std = ax[1][1].set_title("Predicted std")
            
            fig.colorbar(im_gt, ax=ax[0],shrink=0.7)
            fig.colorbar(img_std, ax=ax[1],shrink=0.7) 
            fig.suptitle("MAE reconstruction after ("+str(training_examples)+" training examples)")
            plt.savefig(os.path.join(self.save_path,'figures','MAE_',checkpoint+"_time_{}.pdf".format(time)))

        

    def run(self):
        raise NotImplementedError("Filmed model run not implemented yet. Needs to considder sst input.")

    def get_parameters(self):
        return self.model.parameters()
    
    def training(self,wandb_run=None,**kwargs):
        self.load_statistics()
        self.set_seed(42) #torch.seed()
        LOG.info("Save path: %s", self.save_path)

        if self.enable_amp == True:
            self.gscaler = amp.GradScaler()
        
        print("Trainig Data:")
        dataset = SST_galvani(
            self,
            path=kwargs["trainingdata_path"], 
            start_year=kwargs["trainingset_start_year"],
            end_year=kwargs["trainingset_end_year"],
            temporal_step=kwargs["multi_step_training"]
        )
        print("Validation Data:")
        dataset_validation = SST_galvani(
            self,
            path=kwargs["trainingdata_path"], 
            start_year=kwargs["validationset_start_year"],
            end_year=kwargs["validationset_end_year"],
            temporal_step=kwargs["multi_step_validation"])

        if kwargs["advanced_logging"] : 
            mem_log_not_done = True
            print(" ~~~ The GPU Memory will be logged for the first optimization run ~~~")
            print("mem after initialising model : ",round(torch.cuda.memory_allocated(self.device)/10**9,2)," GB")
        model = self.load_model(self.checkpoint_path)
        model.train()

        # optimizer = torch.optim.SGD(model.get_film_params(), lr=kwargs["learning_rate"], momentum=0.9)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=kwargs["learning_rate"])

        # Scheduler
        if kwargs["scheduler_type"] == 'ReduceLROnPlateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.2, patience=5, mode='min')
        elif kwargs["scheduler_type"] == 'CosineAnnealingLR':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=kwargs["scheduler_horizon"])
        elif kwargs["scheduler_type"] == 'CosineAnnealingWarmRestarts':
            self.scheduler =  torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer,T_0=kwargs["scheduler_horizon"])
        else:
            self.scheduler = None
        
        # store the optimizer and scheduler in the model class
        optimizer = self.optimizer 
        scheduler = self.scheduler 
        
        #Loss
        if kwargs["loss_fn"] == "CosineMSE":
            self.loss_fn = CosineMSELoss(reduction='mean')
        elif kwargs["loss_fn"] == "L2Sphere":
            self.loss_fn = L2Sphere(relative=True, squared=True)
        elif kwargs["loss_fn"] == "CRPS":
            self.loss_fn = NormalCRPS(relative=True, squared=True)
        else:
            self.loss_fn = torch.nn.MSELoss()

        if kwargs["advanced_logging"] and mem_log_not_done : 
            print("mem after init optimizer and scheduler and loading weights : ",round(torch.cuda.memory_allocated(self.device)/10**9,2)," GB")

        training_loader = DataLoader(dataset,shuffle=True,num_workers=kwargs["training_workers"], batch_size=kwargs["batch_size"])
        self.validation_loader = DataLoader(dataset_validation,shuffle=True,num_workers=kwargs["training_workers"], batch_size=kwargs["batch_size"])

        scale = 0.0005#1e-5
        # for logging to local file (no wandb)
        self.val_means = [[]] * (kwargs["multi_step_validation"]+1)
        self.val_stds  = [[]] * (kwargs["multi_step_validation"]+1)
        self.losses    = []
        self.epoch = 0
        self.iter = 0
        batch_loss = 0

        # to debug training don't start with validation, actually never start training with validation, we do not have space on the cluster
        if True: #self.debug:
            start_valid = 1
        else:
            start_valid = 1

        for i, data in enumerate(training_loader):

            # Validation
            if (i+start_valid) % (kwargs["validation_interval"]*(self.accumulation_steps + 1)) == 0: # +1 to skip sfno eval but then scale needs to be higher not 0 for sfno
               self.validation(mem_log_not_done,**kwargs) 
            # Training  
            # loss = []
            discount_factor = 1
            with amp.autocast(self.enable_amp):
                
                if kwargs["advanced_logging"] and mem_log_not_done : 
                    print("mem before loading data : ",round(torch.cuda.memory_allocated(self.device)/10**9,2)," GB")
                input_sst  = self.normalise_film(data).to(self.device)
                
                if kwargs["advanced_logging"] and mem_log_not_done : 
                    print("mem before exec model : ",round(torch.cuda.memory_allocated(self.device)/10**9,2)," GB")
                output = model(input_sst, np.random.uniform(0.4,0.8))
                loss = self.loss_fn(output, input_sst) 
                
            loss = loss / (self.accumulation_steps+1)
            batch_loss += loss.item()
            if kwargs["advanced_logging"] and mem_log_not_done : 
                print("mem before backward : ",round(torch.cuda.memory_allocated(self.device)/10**9,2)," GB")
            
            #backward
            if self.enable_amp:
                self.gscaler.scale(loss).backward()
            else:
                loss.backward()

            # turn of memory logging
            if kwargs["advanced_logging"] and mem_log_not_done : 
                print("mem after backward : ",round(torch.cuda.memory_allocated(self.device)/10**9,2)," GB")
                mem_log_not_done = False

            # Adjust learning weights
            if ((i + 1) % (self.accumulation_steps + 1) == 0) or (i + 1 == len(training_loader)):
                # Update Optimizer
                if self.enable_amp:
                    self.gscaler.step(self.optimizer)
                    self.gscaler.update()
                else:
                    optimizer.step()
                model.zero_grad()

                # logging
                self.iter += 1
                if self.local_logging : self.losses.append(round(batch_loss,5))
                if self.wandb_run is not None:
                    wandb.log({"loss": round(batch_loss,5) })
                if kwargs["advanced_logging"]:
                    print("Iteration: ", i, " Loss: ", round(batch_loss,5)," - scale: ",round(scale,5))
                batch_loss = 0
            else:
                if kwargs["advanced_logging"] and ultra_advanced_logging:
                    print("skipping optimizer step, accumulate gradients")
            
        # end of epoch
        self.epoch += 1
        print("End of epoch ",self.epoch)
        self.save_checkpoint()
    
    def validation(self,mem_log_not_done,**kwargs):
        if kwargs["advanced_logging"] and mem_log_not_done : 
            print("mem before validation : ",round(torch.cuda.memory_allocated(self.device)/10**9,2)," GB")
        model = self.model
        val_loss = []
        val_log  = {}
        model.eval()
        with torch.no_grad():
            # For loop over validation dataset, calculates the validation loss mean for number of kwargs["validation_epochs"]
            for val_epoch, val_data in enumerate(self.validation_loader):
                
                val_input_sst  = self.normalise_film(val_data).to(self.device) # get gt sst from next step
                outputs = model(val_input_sst, np.random.uniform(0.4,0.8))
                val_loss.append(self.loss_fn(outputs, val_input_sst) / kwargs["batch_size"])
                
                if val_epoch > kwargs["validation_epochs"]:
                    val_loss_array  = np.array(val_loss)
                    val_log["valid loss"] = round(val_loss_array.mean(),5)
                    val_log["valid loss std "] = round(val_loss_array.std(),5)
                    break
            
            #scheduler
            if kwargs["scheduler_type"] == 'ReduceLROnPlateau':
                self.scheduler.step(val_log["valid loss"])
            elif kwargs["scheduler_type"] == 'CosineAnnealingLR':
                self.scheduler.step()
                if self.epoch >= kwargs["scheduler_horizon"]:
                    LOG.info("Terminating training after reaching params.max_epochs while LR scheduler is set to CosineAnnealingLR") 
            elif kwargs["scheduler_type"] == 'CosineAnnealingWarmRestarts':
                self.scheduler.step(self.iter)
            if self.scheduler is not None and self.scheduler != "None": 
                lr = self.scheduler.get_last_lr()[0]
                val_log["learning rate"] = lr

            # little complicated console logging - looks nicer than LOG.info(str(val_log))
            print("-- validation after ",self.iter*kwargs["batch_size"], "training examples")
            val_log_keys = list(val_log.keys())
            for log_idx in range(0,kwargs["multi_step_validation"]*2+1,2): 
                LOG.info(val_log_keys[log_idx] + " : " + str(val_log[val_log_keys[log_idx]]) 
                            + " +/- " + str(val_log[val_log_keys[log_idx+1]]))
            if self.wandb_run :
                wandb.log(val_log,commit=False)
        # save model and training statistics for checkpointing
        if (self.iter+1) % (kwargs["validation_interval"]*kwargs["save_checkpoint_interval"]) == 0:
            save_file ="checkpoint_"+kwargs["model_type"]+"_"+kwargs["model_version"]+"_"+"_epoch={}.pkl".format(i)
            self.save_checkpoint(save_file)
        model.train()
        if kwargs["advanced_logging"] and mem_log_not_done : 
            print("mem after validation : ",round(torch.cuda.memory_allocated(self.device)/10**9,2)," GB")
    
def get_model(**kwargs):
    models = {
        "latest": MAE,
    }
    return models[kwargs["model_version"]](**kwargs)