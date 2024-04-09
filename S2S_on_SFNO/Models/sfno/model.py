# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
import os
from time import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from ..models import Model
import datetime

import climetlab as cml
import wandb

# import ai_models_fourcastnetv2.fourcastnetv2 as nvs
from .sfnonet import FourierNeuralOperatorNet
from .sfnonet import FourierNeuralOperatorNet_Filmed
from ..train import ERA5_galvani

LOG = logging.getLogger(__name__)
local_logging = True

class FourCastNetv2(Model):
    # Download
    download_url = "https://get.ecmwf.int/repository/test-data/ai-models/fourcastnetv2/small/{file}"
    download_files = ["weights.tar", "global_means.npy", "global_stds.npy"]

    # Input
    area = [90, 0, -90, 360 - 0.25]
    grid = [0.25, 0.25]

    param_sfc = ["10u", "10v", "2t", "sp", "msl", "tcwv", "100u", "100v"]

    # param_level_pl = (
    #     [ "u", "v", "z", "t", "r"],
    #     [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50],
    # )

    # param_sfc = ["u10", "v10", "t2m", "sp", "msl", "tcwv", "u100", "v100"] # don't know why I would change the names like this, maybe for combatibility with other variables. Like this it can't be read by coperincus .

    param_level_pl = (
        [ "u", "v", "z", "t", "r"],
        [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50],
    )

    # for grib data from copernicus the variable name is different (10u) from what will be used if converted to xarray. Xarray uses the cfVarName from the metadata (u10)
    # if an xarray dataset is used as input the ordering list musst be adapted to the cfVarName
    # 73 Variables
    ordering = [
        "10u",
        "10v",
        "100u",
        "100v",
        "2t",
        "sp",
        "msl",
        "tcwv",
        "u50",
        "u100",
        "u150",
        "u200",
        "u250",
        "u300",
        "u400",
        "u500",
        "u600",
        "u700",
        "u850",
        "u925",
        "u1000",
        "v50",
        "v100",
        "v150",
        "v200",
        "v250",
        "v300",
        "v400",
        "v500",
        "v600",
        "v700",
        "v850",
        "v925",
        "v1000",
        "z50",
        "z100",
        "z150",
        "z200",
        "z250",
        "z300",
        "z400",
        "z500",
        "z600",
        "z700",
        "z850",
        "z925",
        "z1000",
        "t50",
        "t100",
        "t150",
        "t200",
        "t250",
        "t300",
        "t400",
        "t500",
        "t600",
        "t700",
        "t850",
        "t925",
        "t1000",
        "r50",
        "r100",
        "r150",
        "r200",
        "r250",
        "r300",
        "r400",
        "r500",
        "r600",
        "r700",
        "r850",
        "r925",
        "r1000",
    ]

    levels_per_pl = {"u_component_of_wind":[1000,925,850,700,600,500,400,300,250,200,150,100,50],
                     "v_component_of_wind":[1000,925,850,700,600,500,400,300,250,200,150,100,50],
                     "geopotential":[1000,925,850,700,600,500,400,300,250,200,150,100,50],
                     "temperature":[1000,925,850,700,600,500,400,300,250,200,150,100,50],
                     "relative_humidity":[1000,925,850,700,600,500,400,300,250,200,150,100,50]}

    param_sfc_ERA5 = ["10m_u_component_of_wind", "10m_v_component_of_wind", "2m_temperature", "surface_pressure", "mean_sea_level_pressure", "total_column_water_vapour"]

        # u50     v50   z50     t50     r50
        # u100    v100  z100    t100    r100
        # u150    v150  z150    t150    r150
        # u200    v200  z200    t200    r200
        # u250    v250  z250    t250    r250
        # u300    v300  z300    t300    r300
        # u400    v400  z400    t400    r400
        # u500    v500  z500    t500    r500
        # u600    v600  z600    t600    r600
        # u700    v700  z700    t700    r700
        # u850    v850  z850    t850    r850
        # u925    v925  z925    t925    r925
        # u1000   v1000 z1000   t1000   r1000

    # Output
    expver = "sfno"

    def __init__(self, precip_flag=False, **kwargs):
        super().__init__(**kwargs)

        self.n_lat = 721
        self.n_lon = 1440
        self.hour_steps = 6
        self.input_type = kwargs["input"]

        self.backbone_channels = len(self.ordering)

        if self.sfno_weights:
            self.checkpoint_path = self.sfno_weights
        else:
            self.checkpoint_path = os.path.join(self.assets, "weights.tar")

        if kwargs["film_weights"]:
            self.checkpoint_path_film =self.film_weights
        else:
            self.checkpoint_path_film = None

        # create model
        self.model = FourierNeuralOperatorNet(**kwargs)

    def load_statistics(self, film_gen_type=None):
        path = os.path.join(self.assets, "global_means.npy")
        LOG.info("Loading %s", path)
        self.means = np.load(path)
        self.means = self.means[:, : self.backbone_channels, ...]
        self.means = self.means.astype(np.float32)

        path = os.path.join(self.assets, "global_stds.npy")
        LOG.info("Loading %s", path)
        self.stds = np.load(path)
        self.stds = self.stds[:, : self.backbone_channels, ...]
        self.stds = self.stds.astype(np.float32)

        if film_gen_type is not None:
            self.means_film = np.load(os.path.join(self.assets, "global_means_sst.npy"))
            self.means_film = self.means_film.astype(np.float32)
            self.stds_film = np.load(os.path.join(self.assets, "global_stds_sst.npy"))
            self.stds_film = self.stds_film.astype(np.float32)

    def load_model(self, checkpoint_file):
        # model = nvs.FourierNeuralOperatorNet()
        # model = FourierNeuralOperatorNet()
        model = self.model

        model.zero_grad()
        # Load weights

        checkpoint = torch.load(checkpoint_file, map_location=self.device)

        if "model_state" in checkpoint.keys(): weights = checkpoint["model_state"]
        else: weights = checkpoint
        drop_vars = ["module.norm.weight", "module.norm.bias"]
        weights = {k: v for k, v in weights.items() if k not in drop_vars}

        # Make sure the parameter names are the same as the checkpoint
        # need to use strict = False to avoid this error message when
        # using sfno_76ch::
        # RuntimeError: Error(s) in loading state_dict for Wrapper:
        # Missing key(s) in state_dict: "module.trans_down.weights",
        # "module.itrans_up.pct",
        try:
            # Try adding model weights as dictionary
            new_state_dict = dict()
            # for k, v in checkpoint["model_state"].items():
            for k, v in weights.items():
                name = k[7:]
                if name != "ged":
                    new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
        except Exception:
            # model.load_state_dict(checkpoint["model_state"])
             model.load_state_dict(weights)

        # Set model to eval mode and return
        model.eval()
        model.to(self.device)

        return model

    def normalise(self, data, reverse=False):
        """Normalise data using pre-saved global statistics"""
        if reverse:
            new_data = data * self.stds + self.means
        else:
            new_data = (data - self.means) / self.stds
        return new_data

    def normalise_film(self, data, reverse=False):
        """Normalise data using pre-saved global statistics"""
        if reverse:
            new_data = data * self.stds_film + self.means_film
        else:
            new_data = (data - self.means_film) / self.stds_film
        return new_data

    def run(self):
        self.load_statistics()

        if self.input_type == "localERA5":
            all_fields = self.all_fields()
        else:
            all_fields = self.all_fields
        
        # all_fields.save("/home/lenny/test_copernicus20210101.grib")
        # output = cml.new_grib_output("/home/lenny/test1_copernicus.grib")
        # output.write(all_fields.to_numpy(),all_fields[0])
        # output.close()
        
        all_fields = all_fields.sel(
            param_level=self.ordering, remapping={"param_level": "{param}{levelist}"}
        )
        all_fields = all_fields.order_by(
            {"param_level": self.ordering},
            remapping={"param_level": "{param}{levelist}"},
        )

        all_fields_numpy = all_fields.to_numpy(dtype=np.float32) # machine precision 1.1920929e-07 , 6 accurate decimals

        all_fields_numpy = self.normalise(all_fields_numpy)

        model = self.load_model(self.checkpoint_path)

        # Run the inference session
        input_iter = torch.from_numpy(all_fields_numpy).to(self.device)

        sample_sfc = all_fields.sel(param="2t")[0]

        torch.set_grad_enabled(False)

        # Test
        dataset = ERA5_galvani(self,start_year=2019,end_year=2020)
        data = dataset[0]

        with self.stepper(self.hour_steps) as stepper:
            for i in range(self.lead_time // self.hour_steps):
                output = model(input_iter)

                input_iter = output
                if i == 0 and LOG.isEnabledFor(logging.DEBUG):
                    LOG.debug("Mean/stdev of normalised values: %s", output.shape)

                    for j, name in enumerate(self.ordering):
                        LOG.debug(
                            "    %s %s %s %s %s",
                            name,
                            np.mean(output[:, j].cpu().numpy()),
                            np.std(output[:, j].cpu().numpy()),
                            np.amin(output[:, j].cpu().numpy()),
                            np.amax(output[:, j].cpu().numpy()),
                        )

                # Save the results
                step = (i + 1) * self.hour_steps
                output = self.normalise(output.cpu().numpy(), reverse=True)

                if i == 0 and LOG.isEnabledFor(logging.DEBUG):
                    LOG.debug("Mean/stdev of denormalised values: %s", output.shape)

                    for j, name in enumerate(self.ordering):
                        LOG.debug(
                            "    %s mean=%s std=%s min=%s max=%s",
                            name,
                            np.mean(output[:, j]),
                            np.std(output[:, j]),
                            np.amin(output[:, j]),
                            np.amax(output[:, j]),
                        )

                self.write(
                    output[0],
                    check_nans=True,
                    template=all_fields,
                    step=step,
                    param_level_pl=self.param_level_pl,
                    param_sfc=self.param_sfc,
                    precip_output=None
                )

                stepper(i, step)
    def training(self,wandb_run=None,**kwargs):
        self.load_statistics()
        
        print("Trainig Data:")
        dataset = ERA5_galvani(
            self,
            path=kwargs["trainingdata_path"], 
            start_year=kwargs["trainingset_start_year"],
            end_year=kwargs["trainingset_end_year"],
            sst=False
        )
        print("Validation Data:")
        dataset_validation = ERA5_galvani(
            self,
            path=kwargs["trainingdata_path"], 
            start_year=kwargs["validationset_start_year"],
            end_year=kwargs["validationset_end_year"],
            auto_regressive_steps=kwargs["autoregressive_steps"],
            sst=False)
        
        model = self.load_model(self.checkpoint_path)
        model.train()

        # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        optimizer = torch.optim.Adam(model.parameters(), lr=2*0.001)
        scheduler =  torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=kwargs["scheduler_horizon"])
        loss_fn = torch.nn.MSELoss()

        training_loader = DataLoader(dataset,shuffle=True,num_workers=kwargs["training_workers"], batch_size=kwargs["batch_size"])
        validation_loader = DataLoader(dataset_validation,shuffle=True,num_workers=kwargs["training_workers"], batch_size=kwargs["batch_size"])
        
        ## for logging offline to local file (no wandb)
        self.val_means = [[]] * (kwargs["autoregressive_steps"]+1)
        self.val_stds  = [[]] * (kwargs["autoregressive_steps"]+1)
        self.losses    = []

        for i, (input, g_truth) in enumerate(training_loader):

            # Validation
            if i % kwargs["validation_interval"] == 0:
                val_loss = {}
                val_log  = {}
                model.eval()
                with torch.no_grad():
                    # For loop over validation dataset, calculates the validation loss mean for number of kwargs["validation_epochs"]
                    for val_epoch, val_data in enumerate(validation_loader):
                        # Calculates the validation loss for autoregressive model evaluation
                        # if self.auto_regressive_steps = 0 the dataloader only outputs 2 datapoint 
                        # and the for loop only runs once (calculating the ordinary validation loss with no auto regressive evaluation
                        val_input_era5 = None
                        for val_idx in range(len(val_data)-1):
                            if val_input_era5 is None: val_input_era5 = self.normalise(val_data[val_idx]).to(self.device)
                            else: val_input_era5 = outputs
                            val_g_truth_era5 = self.normalise(val_data[val_idx+1]).to(self.device)
                            outputs = model(val_input_era5)
                            val_loss_value = loss_fn(outputs, val_g_truth_era5) / kwargs["batch_size"]
                            if val_epoch == 0: 
                                val_loss["validation loss (n={}, autoregress={})".format(
                                    kwargs["validation_epochs"],val_idx)] = [val_loss_value.cpu()]
                            else:
                                val_loss["validation loss (n={}, autoregress={})".format(
                                    kwargs["validation_epochs"],val_idx)].append(val_loss_value.cpu())

                        # end of validation 
                        if val_epoch > kwargs["validation_epochs"]:
                            for k in val_loss.keys():
                                val_loss_array      = np.array(val_loss[k])
                                val_log[k]          = round(val_loss_array.mean(),5)
                                val_log["std " + k] = round(val_loss_array.std(),5)
                            break
                    
                    if scheduler: 
                        lr = scheduler.get_last_lr()[0]
                        val_log["learning rate"] = lr
                        scheduler.step(i)
                   
                    # LOG.info("Validation loss: "+str(mean_val_loss)+" +/- "+str(std_val_loss)+" (n={})".format(kwargs["validation_epochs"]))

                    # little complicated console logging - looks nicer
                    print("-- validation after ",i*kwargs["batch_size"], "training examples")
                    val_log_keys = list(val_log.keys())
                    for log_idx in range(0,kwargs["autoregressive_steps"]*2+1,2):
                        # log to console
                        LOG.info(val_log_keys[log_idx] + " : " + str(val_log[val_log_keys[log_idx]]) 
                                 + " +/- " + str(val_log[val_log_keys[log_idx+1]]))
                        # log to local file
                        self.val_means[log_idx].append(val_log[val_log_keys[log_idx]])
                        self.val_stds[log_idx].append(val_log[val_log_keys[log_idx+1]])
                    if wandb_run :
                        wandb.log(val_log)
                if i % (kwargs["validation_interval"]*kwargs["save_checkpoint_interval"]) == 0:
                    self.save_and_exit()
                    save_file ="checkpoint_"+kwargs["model"]+"_"+kwargs["model_version"]+"_epoch={}.pkl".format(i)
                    torch.save(model.state_dict(), os.path.join( kwargs["save_path"],save_file))
                model.train()
            
            # Training  
            input_era5 = self.normalise(input).to(self.device)
            g_truth_era5 = self.normalise(g_truth).to(self.device)
            
            optimizer.zero_grad()

            # Make predictions for this batch
            outputs = model(input_era5)

            # Compute the loss and its gradients
            loss = loss_fn(outputs, g_truth_era5)
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            # logging
            loss_value = round(loss.item(),5)
            if local_logging : self.losses.append(loss_value)
            if wandb_run is not None:
                wandb.log({"loss": loss_value })
            if kwargs["debug"]:
                print("Epoch: ", i, " Loss: ", loss_value)
        
        self.save_and_exit()
    
    # needed only for offline logging, commented out atm
    def save_and_exit(self):
        if local_logging : 
            print(" -> saving to : ",self.save_path)
            np.save(os.path.join( self.save_path,"val_means.npy"),self.val_means)
            np.save(os.path.join( self.save_path,"val_stds.npy"),self.val_stds)
            np.save(os.path.join( self.save_path,"losses.npy"),self.losses)
        # self.model is the trained model?? or self.state_dict()??
        # save_file ="checkpoint_"+self.timestr+"_final.pkl"
        # torch.save(self.model.state_dict(), os.path.join( self.save_path,save_file))


class FourCastNetv2_filmed(FourCastNetv2):
    def __init__(self, precip_flag=False, **kwargs):
        super().__init__(precip_flag, **kwargs)

        # init model
        self.model = FourierNeuralOperatorNet_Filmed(self.device,**kwargs)
    
    def load_model(self, checkpoint_file):
        
        model = self.model
        model.zero_grad()

        #  Load Filmed weights
        if self.checkpoint_path_film:
            checkpoint_film = torch.load(self.checkpoint_file_film, map_location=self.device)
            # model.film_gen.load_state_dict(checkpoint_film["model_state"])
            model.film_gen.load_state_dict(checkpoint_film)
        else:
            pass
        
        # Load SFNO weights
        checkpoint_sfno = torch.load(checkpoint_file, map_location=self.device)
        if "model_state" in checkpoint_sfno.keys(): weights = checkpoint_sfno["model_state"]
        else: weights = checkpoint_sfno
        drop_vars = ["module.norm.weight", "module.norm.bias"]
        weights = {k: v for k, v in weights.items() if k not in drop_vars}

        # Make sure the parameter names are the same as the checkpoint
        # need to use strict = False to avoid this error message when
        # using sfno_76ch::
        # RuntimeError: Error(s) in loading state_dict for Wrapper:
        # Missing key(s) in state_dict: "module.trans_down.weights",
        # "module.itrans_up.pct",

        # Load SFNO weights
        try:
            # Try adding model weights as dictionary
            new_state_dict = dict()
            # for k, v in checkpoint_sfno["model_state"].items():
            for k, v in weights.items():
                name = k[7:]
                if name != "ged":
                    new_state_dict[name] = v
            model.load_state_dict(new_state_dict,strict=False)
        except Exception:
            # model.load_state_dict(checkpoint_sfno["model_state"])
            model.load_state_dict(weights)

        # Set model to eval mode and return
        model.eval()
        model.to(self.device)

        return model

    def run(self):
        raise NotImplementedError("Filmed model run not implemented yet. Needs to considder sst input.")

    def training(self,wandb_run=None,**kwargs):
        self.load_statistics(kwargs["film_gen_type"])
        
        print("Trainig Data:")
        dataset = ERA5_galvani(
            self,
            path=kwargs["trainingdata_path"], 
            start_year=kwargs["trainingset_start_year"],
            end_year=kwargs["trainingset_end_year"]
        )
        print("Validation Data:")
        dataset_validation = ERA5_galvani(
            self,
            path=kwargs["trainingdata_path"], 
            start_year=kwargs["validationset_start_year"],
            end_year=kwargs["validationset_end_year"],
            auto_regressive_steps=kwargs["autoregressive_steps"])
        
        model = self.load_model(self.checkpoint_path)
        model.train()

        # optimizer = torch.optim.SGD(model.get_film_params(), lr=0.001, momentum=0.9)
        optimizer = torch.optim.Adam(model.get_film_params(), lr=2*0.001)
        scheduler =  torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=kwargs["scheduler_horizon"])
        
        loss_fn = torch.nn.MSELoss()

        training_loader = DataLoader(dataset,shuffle=True,num_workers=kwargs["training_workers"], batch_size=kwargs["batch_size"])
        validation_loader = DataLoader(dataset_validation,shuffle=True,num_workers=kwargs["training_workers"], batch_size=kwargs["batch_size"])

        scale = 0.0
        # for logging to local file (no wandb)
        self.val_means = [[]] * (kwargs["autoregressive_steps"]+1)
        self.val_stds  = [[]] * (kwargs["autoregressive_steps"]+1)
        self.losses    = []

        for i, (input, g_truth) in enumerate(training_loader):

            # Validation
            if i % kwargs["validation_interval"] == 0:
                val_loss = {}
                val_log  = {}
                model.eval()
                with torch.no_grad():
                    # For loop over validation dataset, calculates the validation loss mean for number of kwargs["validation_epochs"]
                    for val_epoch, val_data in enumerate(validation_loader):
                        # Calculates the validation loss for autoregressive model evaluation
                        # if self.auto_regressive_steps = 0 the dataloader only outputs 2 datapoint 
                        # and the for loop only runs once (calculating the ordinary validation loss with no auto regressive evaluation
                        val_input_era5 = None
                        for val_idx in range(len(val_data)-1):
                            if val_input_era5 is None: val_input_era5 = self.normalise(val_data[val_idx][0]).to(self.device)
                            else: val_input_era5 = outputs
                            val_input_sst  = self.normalise_film(val_data[val_idx][1]).to(self.device)
                            val_g_truth_era5 = self.normalise(val_data[val_idx+1][0]).to(self.device)
                            outputs = model(val_input_era5,val_input_sst,scale)
                            val_loss_value = loss_fn(outputs, val_g_truth_era5) / kwargs["batch_size"]
                            if val_epoch == 0: 
                                val_loss["validation loss (n={}, autoregress={})".format(
                                    kwargs["validation_epochs"],val_idx)] = [val_loss_value.cpu()]
                            else:
                                val_loss["validation loss (n={}, autoregress={})".format(
                                    kwargs["validation_epochs"],val_idx)].append(val_loss_value.cpu())

                        # end of validation 
                        if val_epoch > kwargs["validation_epochs"]:
                            for k in val_loss.keys():
                                val_loss_array      = np.array(val_loss[k])
                                val_log[k]          = round(val_loss_array.mean(),5)
                                val_log["std " + k] = round(val_loss_array.std(),5)
                            break
                    
                    if scheduler: 
                        lr = scheduler.get_last_lr()[0]
                        val_log["learning rate"] = lr
                        scheduler.step(i)
                    # change scale value based on validation loss
                    if list(val_log.values())[0] < kwargs["val_loss_threshold"] and scale < 1.0:
                        val_log["scale"] = lr
                        scale = scale + 0.05

                    # little complicated console logging - looks nicer than LOG.info(str(val_log))
                    print("-- validation after ",i*kwargs["batch_size"], "training examples")
                    val_log_keys = list(val_log.keys())
                    for log_idx in range(0,kwargs["autoregressive_steps"]*2+1,2): 
                        LOG.info(val_log_keys[log_idx] + " : " + str(val_log[val_log_keys[log_idx]]) 
                                 + " +/- " + str(val_log[val_log_keys[log_idx+1]]))
                        # log to local file
                        self.val_means[log_idx].append(val_log[val_log_keys[log_idx]])
                        self.val_stds[log_idx].append(val_log[val_log_keys[log_idx+1]])
                    if wandb_run :
                        wandb.log(val_log)
                # save model and training statistics for checkpointing
                if i % (kwargs["validation_interval"]*kwargs["save_checkpoint_interval"]) == 0:
                    save_file ="checkpoint_"+kwargs["model"]+"_"+kwargs["model_version"]+"_"+kwargs["film_gen_type"]+"_epoch={}.pkl".format(i)
                    torch.save(model.state_dict(), os.path.join( kwargs["save_path"],save_file))
                model.train()
            
            # Training  
            input_era5, input_sst = self.normalise(input[0]).to(self.device), self.normalise_film(input[1]).to(self.device)
            g_truth_era5, g_truth_sst = self.normalise(g_truth[0]).to(self.device), self.normalise_film(g_truth[1]).to(self.device)
            
            optimizer.zero_grad()

            # Make predictions for this batch
            outputs = model(input_era5,input_sst,scale)

            # Compute the loss and its gradients
            loss = loss_fn(outputs, g_truth_era5)
            loss.backward()

            # Adjust learning weights
            optimizer.step()
            scheduler.step(i)

            # logging
            loss_value = round(loss.item(),5)
            if local_logging : self.losses.append(loss_value)
            if wandb_run is not None:
                wandb.log({"loss": loss_value })
            if kwargs["debug"]:
                print("Epoch: ", i, " Loss: ", loss_value," - scale: ",scale)

        self.save_and_exit()
        
    def test_training(self,**kwargs):
        dataset = ERA5_galvani(
            self,
            path=kwargs["trainingdata_path"], 
            start_year=kwargs["trainingset_start_year"],
            end_year=kwargs["trainingset_end_year"])
        model = self.load_model(self.checkpoint_path)
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        loss_fn = torch.nn.MSELoss()
        training_loader = DataLoader(dataset,shuffle=True,num_workers=kwargs["training_workers"], batch_size=kwargs["batch_size"])
        mean_batch_time = 0
        l1 = time()
        mean_model_time = 0
        for i, data in enumerate(training_loader):
            input, g_truth = data
            optimizer.zero_grad()
            outputs = model(input[0],input[1])
            loss = loss_fn(outputs, g_truth[0])
            loss.backward()
            optimizer.step()
            l2 = time()
            tb = l2-l1
            mean_batch_time = mean_batch_time+(tb - mean_batch_time)/(i+1)
            print("Time for epoch: ", tb , " mean : ", mean_batch_time) 
            t = torch.cuda.get_device_properties(0).total_memory
            r = torch.cuda.memory_reserved(0)
            a = torch.cuda.memory_allocated(0)
            f = r-a  # free inside reserved
            print("total gpu mem:",t)
            print("allocated:",a)
            print("free:",f)    




def get_model(**kwargs):
    models = {
        "0": FourCastNetv2,
        "small": FourCastNetv2,
        "release": FourCastNetv2,
        "latest": FourCastNetv2,
        "film": FourCastNetv2_filmed,
    }
    return models[kwargs["model_version"]](**kwargs)
