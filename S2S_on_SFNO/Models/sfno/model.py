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

        self.checkpoint_path = os.path.join(self.assets, "weights.tar")

        if kwargs["assets_film"]:
            self.checkpoint_path_film = os.path.join(self.assets_film)
        else:
            self.checkpoint_path_film = None

        # create model
        self.model = FourierNeuralOperatorNet(**kwargs)

    def load_statistics(self):
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

    def load_model(self, checkpoint_file):
        # model = nvs.FourierNeuralOperatorNet()
        # model = FourierNeuralOperatorNet()
        model = self.model

        model.zero_grad()
        # Load weights

        checkpoint = torch.load(checkpoint_file, map_location=self.device)

        weights = checkpoint["model_state"]
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
            for k, v in checkpoint["model_state"].items():
                name = k[7:]
                if name != "ged":
                    new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
        except Exception:
            model.load_state_dict(checkpoint["model_state"])

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
            model.film_gen.load_state_dict(checkpoint_film["model_state"])
        else:
            pass
        
        # Load SFNO weights
        checkpoint_sfno = torch.load(checkpoint_file, map_location=self.device)
        weights = checkpoint_sfno["model_state"]
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
            for k, v in checkpoint_sfno["model_state"].items():
                name = k[7:]
                if name != "ged":
                    new_state_dict[name] = v
            model.load_state_dict(new_state_dict,strict=False)
        except Exception:
            model.load_state_dict(checkpoint_sfno["model_state"])

        # Set model to eval mode and return
        model.eval()
        model.to(self.device)

        return model

    def run(self):
        raise NotImplementedError("Filmed model run not implemented yet. Needs to considder sst input.")

    def training(self,wandb_run=None,**kwargs):

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
            end_year=kwargs["validationset_end_year"])
        
        model = self.load_model(self.checkpoint_path)
        model.train()

        optimizer = torch.optim.SGD(model.get_film_params(), lr=0.001, momentum=0.9)
        loss_fn = torch.nn.MSELoss()

        training_loader = DataLoader(dataset,shuffle=True,num_workers=kwargs["training_workers"], batch_size=kwargs["batch_size"])
        validation_loader = DataLoader(dataset_validation,shuffle=True,num_workers=kwargs["training_workers"], batch_size=kwargs["batch_size"])

        scale = 0.0

        for i, (input, g_truth) in enumerate(training_loader):

            # Validation
            if i % kwargs["validation_interval"] == 0:
                val_loss = []
                model.eval()
                with torch.no_grad():
                    for val_epoch, (val_input, val_g_truth) in enumerate(validation_loader):
                        s = time()
                        val_input_era5, val_input_sst = self.normalise(val_input[0].to(self.device)), val_input[1].to(self.device)
                        val_g_truth_era5, val_g_truth_sst = self.normalise(val_g_truth[0].to(self.device)), val_g_truth[1].to(self.device)
                        outputs = model(val_input_era5,val_input_sst,scale)
                        val_loss.append( loss_fn(outputs, val_g_truth_era5) / kwargs["batch_size"])
                        e = time()
                        print("run time for validation batch: ", e-s)
                        if val_epoch > kwargs["validation_epochs"]:
                            break
                    val_loss_pt = torch.tensor(val_loss)
                    mean_val_loss = val_loss_pt.mean()
                    std_val_loss = val_loss_pt.std()
                    # change scale value based on validation loss
                    if mean_val_loss > kwargs["val_loss_threshold"]:
                        scale = scale + 0.05
                    print("Validation loss: ", mean_val_loss, " +/- ", std_val_loss)
                    if wandb_run :
                        wandb.log({"validation_loss": mean_val_loss})
                save_file ="checkpoint_"+kwargs["model"]+"_"+kwargs["model_version"]+"_epoch={}".format(i)
                if wandb_run:
                    save_file = save_file +"_"+ wandb_run.name + ".pkl"
                else:
                     save_file = save_file + kwargs["timestr"] + ".pkl"
                torch.save(model.state_dict(), os.path.join( kwargs["save_path"],save_file))
                model.train()
            
            # Training  
            input_era5, input_sst = input[0].to(self.device), input[1].to(self.device)
            g_truth_era5, g_truth_sst = g_truth[0].to(self.device), g_truth[1].to(self.device)
            
            optimizer.zero_grad()

            # Make predictions for this batch
            outputs = model(input_era5,input_sst,scale)

            # Compute the loss and its gradients
            loss = loss_fn(outputs, g_truth_era5)
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            # logging
            if wandb_run is not None:
                wandb.log({"loss": loss })
            if kwargs["debug"]:
                print("Epoch: ", i, " Loss: ", loss)
            
    
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
