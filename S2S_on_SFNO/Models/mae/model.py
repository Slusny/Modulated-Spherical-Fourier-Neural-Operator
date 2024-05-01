

from ..models import Model
from .maenet import ContextCast

class MAE(Model):
    def __init__(self, **kwargs):
        super()
        # init model
        self.model = ContextCast()
    
    def load_model(self, checkpoint_file):
        
        # model = self.model
        # model.zero_grad()

        # # Load SFNO weights
        # checkpoint_sfno = torch.load(checkpoint_file)
        # if "model_state" in checkpoint_sfno.keys(): weights = checkpoint_sfno["model_state"]
        # else: weights = checkpoint_sfno
        # drop_vars = ["module.norm.weight", "module.norm.bias"] # no checkpoint has that layer, probably lecacy from ai-model dev
        # weights = {k: v for k, v in weights.items() if k not in drop_vars}

        # # print state of loaded model:
        # if self.advanced_logging and 'hyperparameters' in checkpoint_sfno.keys():
        #     print("loaded model with following hyperparameters:")
        #     print("    iter:",checkpoint_sfno["iter"])
        #     for k,v in checkpoint_sfno['hyperparameters'].items():print("    ",k,":",v)

        # # Make sure the parameter names are the same as the checkpoint
        # # need to use strict = False to avoid this error message when
        # # using sfno_76ch::
        # # RuntimeError: Error(s) in loading state_dict for Wrapper:
        # # Missing key(s) in state_dict: "module.trans_down.weights",
        # # "module.itrans_up.pct",

        # # Load SFNO weights
        # if list(weights.keys())[0][0:7] == 'module.':
        #     # Try adding model weights as dictionary
        #     new_state_dict = dict()
        #     # for k, v in checkpoint_sfno["model_state"].items():
        #     for k, v in weights.items():
        #         name = k[7:]
        #         if name != "ged":
        #             new_state_dict[name] = v
        #     try:
        #         model.load_state_dict(new_state_dict)
        #     except RuntimeError as e:
        #         LOG.error(e)
        #         print("--- !! ---")
        #         print("loading state dict with strict=False, please verify if the right model is loaded and strict=False is desired")
        #         print("--- !! ---")
        #         model.load_state_dict(new_state_dict,strict=False)

        # else:
        #     try:
        #         model.load_state_dict(weights)
        #     except RuntimeError as e:
        #         LOG.error(e)
        #         print("--- !! ---")
        #         print("loading state dict with strict=False, please verify if the right model is loaded and strict=False is desired")
        #         print("--- !! ---")
        #         model.load_state_dict(weights,strict=False)

        # #  Load Filmed weights
        # if self.checkpoint_path_film:
        #     checkpoint_film = torch.load(self.checkpoint_file_film)
        #     print("not yet implemented")
        #     sys.exit()
        #     # needs to extract only film_gen weights if the whole model was saved
        #     # model.film_gen.load_state_dict(checkpoint_film["model_state"])
        #     model.film_gen.load_state_dict(checkpoint_film)
        #     del checkpoint_film
        # else:
        #     pass

        # # Set model to eval mode and return
        # model.eval()
        # model.to(self.device)

        # free VRAM
        # del checkpoint_sfno

        # # disable grad for sfno
        # #model.requires_grad = False
        # for name, param in model.named_parameters():
        #     if not "film_gen" in name:
        #         param.requires_grad = False 
        #     # if "film_gen" in name:
        #         # param.requires_grad = True
        #     # param.requires_grad = False 

        return model

    def run(self):
        raise NotImplementedError("Filmed model run not implemented yet. Needs to considder sst input.")

    def training(self,wandb_run=None,**kwargs):
        self.load_statistics(kwargs["film_gen_type"])
        self.set_seed(42) #torch.seed()
        LOG.info("Save path: %s", self.save_path)

        if self.enable_amp == True:
            self.gscaler = amp.GradScaler()
        
        print("Trainig Data:")
        dataset = ERA5_galvani(
            self,
            path=kwargs["trainingdata_path"], 
            start_year=kwargs["trainingset_start_year"],
            end_year=kwargs["trainingset_end_year"],
            auto_regressive_steps=kwargs["multi_step_training"]
        )
        print("Validation Data:")
        dataset_validation = ERA5_galvani(
            self,
            path=kwargs["trainingdata_path"], 
            start_year=kwargs["validationset_start_year"],
            end_year=kwargs["validationset_end_year"],
            auto_regressive_steps=kwargs["multi_step_validation"])

        if kwargs["advanced_logging"] : 
            mem_log_not_done = True
            print(" ~~~ The GPU Memory will be logged for the first optimization run ~~~")
            print("mem after initialising model : ",round(torch.cuda.memory_allocated(self.device)/10**9,2)," GB")
        model = self.load_model(self.checkpoint_path)
        model.film_gen.train()

        # optimizer = torch.optim.SGD(model.get_film_params(), lr=kwargs["learning_rate"], momentum=0.9)
        self.optimizer = torch.optim.Adam(model.get_film_params(), lr=kwargs["learning_rate"])

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
            loss_fn = CosineMSELoss(reduction='mean')
        elif kwargs["loss_fn"] == "L2Sphere":
            loss_fn = L2Sphere(relative=True, squared=True)
        else:
            loss_fn = torch.nn.MSELoss()

        if kwargs["advanced_logging"]: loss_fn_pervar = torch.nn.MSELoss(reduction='none')

        if kwargs["advanced_logging"] and mem_log_not_done : 
            print("mem after init optimizer and scheduler and loading weights : ",round(torch.cuda.memory_allocated(self.device)/10**9,2)," GB")

        training_loader = DataLoader(dataset,shuffle=True,num_workers=kwargs["training_workers"], batch_size=kwargs["batch_size"])
        validation_loader = DataLoader(dataset_validation,shuffle=True,num_workers=kwargs["training_workers"], batch_size=kwargs["batch_size"])

        scale = 0.0005#1e-5
        # for logging to local file (no wandb)
        self.val_means = [[]] * (kwargs["multi_step_validation"]+1)
        self.val_stds  = [[]] * (kwargs["multi_step_validation"]+1)
        self.losses    = []
        self.epoch = 0
        self.iter = 0
        batch_loss = 0

        # to debug training don't start with validation, actually never start training with validation, we do not have space on the cluster
        if self.debug:
            start_valid = 1
        else:
            start_valid = 1

        for i, data in enumerate(training_loader):

            # Validation
            if (i+start_valid) % (kwargs["validation_interval"]*(self.accumulation_steps + 1)) == 0: # +1 to skip sfno eval but then scale needs to be higher not 0 for sfno
                if kwargs["advanced_logging"] and mem_log_not_done : 
                    print("mem before validation : ",round(torch.cuda.memory_allocated(self.device)/10**9,2)," GB")
                val_loss = {}
                val_log  = {}
                model.eval()
                with torch.no_grad():
                    # For loop over validation dataset, calculates the validation loss mean for number of kwargs["validation_epochs"]
                    if kwargs["advanced_logging"] and mse_all_vars: loss_pervar_list = []
                    for val_epoch, val_data in enumerate(validation_loader):
                        # Calculates the validation loss for autoregressive model evaluation
                        # if self.auto_regressive_steps = 0 the dataloader only outputs 2 datapoint 
                        # and the for loop only runs once (calculating the ordinary validation loss with no auto regressive evaluation
                        val_input_era5 = None
                        for val_idx in range(kwargs["multi_step_validation"]+1):
                            if val_input_era5 is None: val_input_era5 = self.normalise(val_data[val_idx][0]).to(self.device)
                            else: val_input_era5 = outputs
                            val_input_sst  = self.normalise_film(val_data[val_idx+1][1]).to(self.device) # get gt sst from next step
                            
                            val_g_truth_era5 = self.normalise(val_data[val_idx+1][0]).to(self.device)
                            outputs = model(val_input_era5,val_input_sst,scale)
                            val_loss_value = loss_fn(outputs, val_g_truth_era5) / kwargs["batch_size"]

                            # loss for each variable
                            if kwargs["advanced_logging"] and mse_all_vars  and val_idx == 0: # !! only for first multi validation step, could include next step with -> ... -> ... in print statement on same line
                                loss_pervar_list.append(loss_fn_pervar(outputs, val_g_truth_era5).mean(dim=(0,2,3)) / kwargs["batch_size"])
                            
                            if val_epoch == 0: 
                                val_loss["validation loss step={}".format(val_idx)] = [val_loss_value.cpu()] #kwargs["validation_epochs"]
                            else:
                                val_loss["validation loss step={}".format(val_idx)].append(val_loss_value.cpu())

                        # end of validation 
                        if val_epoch > kwargs["validation_epochs"]:
                            for k in val_loss.keys():
                                val_loss_array      = np.array(val_loss[k])
                                val_log[k]          = round(val_loss_array.mean(),5)
                                val_log["std " + k] = round(val_loss_array.std(),5)
                            break
                    
                    #scheduler
                    valid_mean = list(val_log.values())[0]
                    if kwargs["scheduler_type"] == 'ReduceLROnPlateau':
                        self.scheduler.step(valid_mean)
                    elif kwargs["scheduler_type"] == 'CosineAnnealingLR':
                        self.scheduler.step()
                        if self.epoch >= kwargs["scheduler_horizon"]:
                            LOG.info("Terminating training after reaching params.max_epochs while LR scheduler is set to CosineAnnealingLR") 
                    elif kwargs["scheduler_type"] == 'CosineAnnealingWarmRestarts':
                        self.scheduler.step(i)
                    if scheduler is not None and scheduler != "None": 
                        lr = scheduler.get_last_lr()[0]
                        val_log["learning rate"] = lr

                    # change scale value based on validation loss
                    # if valid_mean < kwargs["val_loss_threshold"] and scale < 1.0:
                    if scale < 1.0:
                        val_log["scale"] = scale
                        scale = scale + 0.002 # 5e-5 #

                    # little complicated console logging - looks nicer than LOG.info(str(val_log))
                    print("-- validation after ",i*kwargs["batch_size"], "training examples")
                    val_log_keys = list(val_log.keys())
                    for log_idx in range(0,kwargs["multi_step_validation"]*2+1,2): 
                        LOG.info(val_log_keys[log_idx] + " : " + str(val_log[val_log_keys[log_idx]]) 
                                 + " +/- " + str(val_log[val_log_keys[log_idx+1]]))
                        # log to local file
                        # self.val_means[log_idx].append(val_log[val_log_keys[log_idx]]) ## error here
                        # self.val_stds[log_idx].append(val_log[val_log_keys[log_idx+1]]) 
                    if kwargs["advanced_logging"] and mse_all_vars:
                        print("MSE for each variable:")
                        val_loss_value_pervar = torch.stack(loss_pervar_list).mean(dim=0)
                        for idx_var,var_name in enumerate(self.ordering):
                            print("    ",var_name," = ",round(val_loss_value_pervar[idx_var].item(),5))
                        gamma_np = model.gamma.cpu().numpy()
                        beta_np  = model.beta.cpu().numpy()
                        print("gamma values mean : ",round(gamma_np.mean(),5),"+/-",round(gamma_np.std(),5))
                        print("beta values mean  : ",round(beta_np.mean(),5),"+/-",round(beta_np.std(),5))
                    if wandb_run :
                        wandb.log(val_log,commit=False)
                # save model and training statistics for checkpointing
                if (i+1) % (kwargs["validation_interval"]*kwargs["save_checkpoint_interval"]) == 0:
                    save_file ="checkpoint_"+kwargs["model_type"]+"_"+kwargs["model_version"]+"_"+kwargs["film_gen_type"]+"_epoch={}.pkl".format(i)
                    self.save_checkpoint(save_file)
                    if self.params["advanced_logging"]:
                        gamma_np = model.gamma.cpu().numpy()
                        beta_np  = model.beta.cpu().numpy()
                        np.save(os.path.join( self.save_path,"gamma_{}.npy".format(i)),gamma_np)
                        np.save(os.path.join( self.save_path,"beta_{}.npy".format(i)),beta_np)
                        print("gamma values mean : ",round(gamma_np.mean(),5),"+/-",round(gamma_np.std(),5))
                        print("beta values mean  : ",round(beta_np.mean(),5),"+/-",round(beta_np.std(),5))
                model.film_gen.train()
                if kwargs["advanced_logging"] and mem_log_not_done : 
                    print("mem after validation : ",round(torch.cuda.memory_allocated(self.device)/10**9,2)," GB")
            
            # Training  
            # loss = []
            loss = 0
            discount_factor = 1
            with amp.autocast(self.enable_amp):
                for step in range(kwargs["multi_step_training"]+1):
                    #print(" - step : ", step) ## Log multistep loss better
                    if kwargs["advanced_logging"] and mem_log_not_done : 
                        print("mem before loading data : ",round(torch.cuda.memory_allocated(self.device)/10**9,2)," GB")
                    if step == 0 : input_era5 = self.normalise(data[step][0]).to(self.device)
                    else: input_era5 = outputs
                    input_sst  = self.normalise_film(data[step+1][1]).to(self.device) # get gt sst from next step
                    g_truth_era5 = self.normalise(data[step+1][0]).to(self.device)
                    
                    if kwargs["advanced_logging"] and mem_log_not_done : 
                        print("mem before exec model : ",round(torch.cuda.memory_allocated(self.device)/10**9,2)," GB")
                    outputs = model(input_era5,input_sst,scale)
                    # outputs = outputs.detach()
                    # loss.append(loss_fn(outputs, g_truth_era5))#*discount_factor**step
                    if step % (kwargs["training_step_skip"]+1) == 0:
                        if kwargs["advanced_logging"] and ultra_advanced_logging: print("calculating loss for step ",step)
                        if kwargs["advanced_logging"] and mem_log_not_done : 
                            print("mem before loss : ",round(torch.cuda.memory_allocated(self.device)/10**9,2)," GB")
                        loss = loss + loss_fn(outputs, g_truth_era5)/(kwargs["multi_step_training"]+1) #*discount_factor**step
                    else:
                        if kwargs["advanced_logging"] and ultra_advanced_logging : print("skipping step",step)
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
                if local_logging : self.losses.append(round(batch_loss,5))
                if wandb_run is not None:
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


def get_model(**kwargs):
    models = {
        "latest": MAE,
    }
    return models[kwargs["model_version"]](**kwargs)