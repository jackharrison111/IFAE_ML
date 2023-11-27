#Steps are: 

#/nfs/pic.es/user/j/jharriso/IFAE_ML/results/TestRun/1Z_0b_2SFOS_NFs/Run_0057-13-05-2023

# Using a config input:

#Read data

#Preprocess (select variables of interest) + remove duplicates + nans

#Calculate weights ( + remove negative or 0 weights)

#Scale the input features and save the scalers

#Train test split (per sample), can be tied into making the datasets

#USE DATASET_IO CLASS FOR ALL OF THE ABOVE

#Read model (from config)

#Training loop
# - evalutate on validation stuff
# - add option for CV ? 

#Save outputs + trained model

#Separate above and below 
#---------------------------------------

#Test the model on new signal data
# - produce AD score
# - Save all relevant variables
# - Calculate separation from the Sig vs Bkg
# - need to separate the histogram production from the plotting

#Put the anomaly score back into the ROOT files


#------------------------------------------------------------------
#------------------------------------------------------------------

#############################################
#
# Classes for running the training pipeline
# of a VAE.
#
# Intentions to make this general enough to 
# be able to use on any model.
#
# Jack Harrison 21/08/2022
#############################################


import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

import os
import datetime as dt
from tqdm.auto import tqdm
from time import perf_counter
import yaml
import pickle

import copy

#Utils
from utils._utils import load_yaml_config, get_reversed_groupings, make_output_folder


#Dataset includes
from preprocessing.dataset_io import DatasetHandler
from preprocessing.dataset import data_set
from torch.utils.data import DataLoader

#Plotting
from plotting.plot_results import Plotter


class Trainer():
    
    def __init__(self, model, config, sample_groupings={}, output_dir=None):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        
        self.config = load_yaml_config(config)
        
        if not output_dir:
            self.output_dir = make_output_folder(config)
        else:
            self.output_dir = output_dir
        
        self.reversed_groupings = get_reversed_groupings(self.config['groupings'])
        
        self.make_optimizer(learning_rate=self.config['learning_rate'], weight_decay=self.config.get('weight_decay', None))
        self.p = Plotter()
        
        
    def make_optimizer(self, learning_rate=1e-2, weight_decay=None):
        
        if self.config['optimizer'] == 'Adam':
            if weight_decay:
                self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate,weight_decay=weight_decay)
            else:
                self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        else:
            self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        
        
    def train_model(self, dataloader, val_loader=None):
        
        print(f"Using available: {self.device}")
        
        epoch_losses, epoch_mses, epoch_klds = [], [], []
        weighted_epoch_losses = []
        
        self.best_val_loss = None  #float('inf')
        self.best_val_epoch = None
        
        epoch_losses = {}
        weighted_losses = {}
        epoch_val_losses, epoch_val_losses_avg, epoch_val_weighted_losses, sum_val_losses = {}, {}, {}, {}
        sum_weighted_val_losses = {}
        
        weight_loss = self.config['weight_loss']
        use_abs_weights = self.config.get('absolute_weights',True)
        use_scaled = self.config.get('use_scaled', True)

        trim_weight = self.config.get('trim_weight', False)
        stopping_delta = self.config.get('stopping_delta', 0)
        early_stopping = self.config.get('early_stopping', True)
        patience = self.config.get('patience', 10)
        
        s = perf_counter()

        early_stopping_counter = 0
        stop_early = False

        print("Running training for: ", self.config['num_epochs'], " epochs.")
        for epoch in range(self.config['num_epochs']):
            
            if stop_early:
                break
            l1 = perf_counter()
            print(f"Epoch {epoch}:")
            epoch_info, weighted_epoch_info = {}, {}

            
            self.model.train()
            for idx, data_dict in enumerate(dataloader):
                
                data = data_dict['data']
                
                if use_scaled:
                    weight = data_dict['scaled_weight']
                else:
                    weight = data_dict['weight']
                
                data = data.to(self.device)
                weight = weight.to(self.device)
                
                self.optimizer.zero_grad()
                
                #Feed data into model
                outputs = self.model(data)
                
                #Get the loss
                losses = self.model.loss_function(**outputs)
                loss = losses['loss']
                
                epoch_info['loss'] = epoch_info.get('loss', 0) + torch.sum(loss).item()
                
                #TODO: INCLUDE WEIGHT TRIMMING TO 0.5
                
                #Multiply the loss by the weights when updating
                if weight_loss and not use_abs_weights:
                    loss = torch.dot(self.config['added_weight_factor']*weight, loss) 
                
                elif use_abs_weights and trim_weight: 
                    tmp_loss = copy.deepcopy(loss.detach().numpy())
                    loss = torch.dot(torch.abs(torch.clip(weight, min=-0.5,max=0.5)),loss)

                elif use_abs_weights and not trim_weight:   #Should be default, weight trimming done in scaled weight making
                    loss = torch.dot(torch.abs(weight),loss)
                else:
                    loss = torch.dot(weight, loss)
                
                #Take the abs of weights
                weighted_epoch_info['loss'] = weighted_epoch_info.get('loss', 0) + loss.item()
                epoch_info['counts'] = epoch_info.get('counts', 0) + len(data)
                
                
                if 'mse' in losses.keys():
                    mse = torch.dot(self.config['added_weight_factor']*weight, losses['mse']) if weight_loss else torch.dot(weight, loss)
                    kld = torch.dot(self.config['added_weight_factor']*weight, losses['kld']) if weight_loss else torch.dot(weight, loss)
                    epoch_info['mse'] = epoch_info.get('mse', 0) + mse.item()
                    epoch_info['kld'] = epoch_info.get('kld', 0) + kld.item()                   
                
                # Backpropagation based on the loss
                loss.backward()
                self.optimizer.step()
            
            
            epoch_losses.setdefault('loss', []).append(epoch_info['loss']/epoch_info['counts'])
            weighted_losses.setdefault('loss', []).append(weighted_epoch_info['loss']/epoch_info['counts'])
            
            if 'mse' in epoch_info.keys():
                epoch_losses.setdefault('mse', []).append(epoch_info['mse']/epoch_info['counts'])
                epoch_losses.setdefault('kld', []).append(epoch_info['kld']/epoch_info['counts'])
                
            
            ###############################################################
            ### Print output info for tracking

            print(f"      Loss: {epoch_losses['loss'][-1]}")
            print(f"      Weighted: {weighted_losses['loss'][-1]}")
            if 'mse' in epoch_losses.keys():
                    print(f"      MSE: {epoch_losses['mse'][-1]}")
                    print(f"      KLD: {epoch_losses['kld'][-1]}")
            ################################################################
        
            
            #Epoch validation
            #Output a validation loss per sample (per group)
            if val_loader and self.config['val_frequency'] > 0:
                if epoch % self.config['val_frequency'] == 0:
                    
                    
                    
                    self.model.eval()

                    val_losses, val_losses_weighted, val_counts = {}, {}, {}
                    sum_val_loss = 0
                    weighted_sum_val_loss = 0
                    sum_val_counts = 0
                    for idx, data_dict in enumerate(val_loader):
                
                        data = data_dict['data']
                        if use_scaled:
                            weight = data_dict['scaled_weight']
                        else:
                            weight = data_dict['weight']
                        
                        samples = data_dict['sample']
                        data = data.to(self.device)
                        
                        #Feed data into model
                        outputs = self.model(data)

                        #Get the loss
                        losses = self.model.loss_function(**outputs)
                    
                        #TODO: Figure out how to do this in batches
                        loss = losses['loss']
                        weighted_loss = self.config['added_weight_factor']*weight*loss if weight_loss else weight*loss
                        
                        sum_val_loss += torch.sum(loss).item()
                        sum_val_counts += len(loss)

                        #We want the weighted loss to remain per-sample
                        if weight_loss and not use_abs_weights:
                            weighted_loss = self.config['added_weight_factor']*weight*loss 
                        elif use_abs_weights and trim_weight:  #Should be default
                            weighted_loss = torch.abs(torch.clip(weight, min=-0.5,max=0.5))*loss
                        elif use_abs_weights:
                            weighted_loss = torch.abs(weight)*loss
                        else:
                            weighted_loss = weight*loss

                        weighted_sum_val_loss += torch.sum(weighted_loss).item()
                        
                        
                        if 'mse' in losses.keys():
                            mse = torch.sum(mse)
                            kld = torch.sum(kld)
                            
                        for sample in set(samples):
                        
                            #Get the indices that are this sample
                            #TODO: Add safety check for if there are no events? 
                            inds = np.where(np.isin(samples,[sample]))[0]

                            #Sum the loss of these samples
                            sample_losses = loss[inds]
                            w_sample_losses = weighted_loss[inds]    
                            
                            #Get the group that the sample is in
                            group = self.reversed_groupings.get(sample,'All')
                       
                            #Add the loss to the right sample 
                            val_losses[group] = val_losses.get(group,0) + torch.sum(sample_losses).item()
                            val_counts[group] = val_counts.get(group,0) + len(sample_losses)
                            val_losses_weighted[group] = val_losses_weighted.get(group,0) + torch.sum(w_sample_losses).item()
                
                
                    #Now for each sample that we have validated on
                    for group in val_counts.keys():
                        
                        #Average the loss over an epoch for each sample
                        val_losses[group] = val_losses[group]/val_counts[group]
                        val_losses_weighted[group] = val_losses_weighted[group]/val_counts[group]
                        
                        #Store this epoch's validation data
                        epoch_val_losses.setdefault(group, []).append(val_losses[group])
                        epoch_val_weighted_losses.setdefault(group, []).append(val_losses_weighted[group])
                        
                    sum_val_losses.setdefault('Val.', []).append(sum_val_loss/sum_val_counts)

                    sum_weighted_val_losses.setdefault('Val.', []).append(weighted_sum_val_loss/sum_val_counts)
                    
                    print(f"      Validation Loss: {sum_val_loss/sum_val_counts}")
                    print(f"      Validation weighted: {weighted_sum_val_loss/sum_val_counts}")
                    print(f"      Validation Loss/Sample: {val_losses}")
                    print(f"      Validation weighted/Sample: {val_losses_weighted}")
                
                    #Changed to use weighted loss...
                    #if self.best_val_loss is None or sum_val_loss/sum_val_counts < self.best_val_loss:
                    if self.best_val_loss is None or weighted_sum_val_loss/sum_val_counts < self.best_val_loss:
                        self.best_model = copy.deepcopy(self.model)
                        self.best_val_epoch = epoch
                        #self.best_val_loss = sum_val_loss/sum_val_counts
                        self.best_val_loss = weighted_sum_val_loss/sum_val_counts
                        self.best_optimizer = copy.deepcopy(self.optimizer)


                    #Decide on early stopping condition
                    #Add a patience and then a checking condition on the validation loss
                    if early_stopping:

                        #Check that the validation loss hasnt increased since X
                        if len(sum_weighted_val_losses['Val.']) > 2:

                            if sum_weighted_val_losses['Val.'][-1] > sum_weighted_val_losses['Val.'][-2] + stopping_delta:
                                early_stopping_counter += 1
                                print(f"Found early stopping increment:\n   Previous val loss: {sum_weighted_val_losses['Val.'][-2]} , Current val loss: {sum_weighted_val_losses['Val.'][-1]}")
                            else:
                                early_stopping_counter = 0
                                
                        if early_stopping_counter > patience:
                            stop_early = True
                            print(f"Stopping early on epoch {epoch} as patience of {patience} has been met...")


                    
            l2 = perf_counter()
            print(f"Finished epoch... time taken: {round(l2-l1,2)}s.")

            
        print(f"Finished training... time taken: {round((l2-s),2)/60}mins.")
        
        return {'epoch_losses': epoch_losses,
                'weighted_losses' : weighted_losses,
                'epoch_val_losses' : epoch_val_losses,
                'epoch_val_weighted_losses' : epoch_val_weighted_losses,
                'sum_val_losses' : sum_val_losses,
                'sum_weighted_val_losses' : sum_weighted_val_losses}
                
    
    #TODO: DO THIS IN A WAY THAT MAKES IT LOOK NICE
    #USE MPLHEP
    def make_training_outputs(self, **results):
        
        
        epoch_axis = [i for i in range(len(results['epoch_losses']['loss']))]
        
       
        #Plot Loss vs Epoch
        self.p.plot_loss(epoch_axis, results['epoch_losses']['loss'], xlab='Epoch', ylab='Loss',
                   save_name=os.path.join(self.output_dir,'Epoch_losses.png'))
        
        #Plot Logloss vs Epoch
        epoch_loglosses = np.log(results['epoch_losses']['loss'])
        self.p.plot_loss(epoch_axis, epoch_loglosses, xlab='Epoch', ylab='logloss',
                   save_name=os.path.join(self.output_dir,'Epoch_loglosses.png'))
        
        #Plot VAE split losses
        if 'mse' in results['epoch_losses'].keys():
            
            loss_parts = {'MSE loss' : results['epoch_losses']['mse'], 
                          'K-L Divergence' : results['epoch_losses']['kld'], 
                          'Total' : results['epoch_losses']['loss']}
            
            self.p.plot_loss_overlay(loss_parts, xlab='Epoch', ylab='Loss',
                                  save_name=os.path.join(self.output_dir,'Epoch_losses_Separated.png'),
                                   val_frequency=1)
    
        #Plot validation unweighted losses
        results['epoch_val_losses']['Train'] = results['epoch_losses']['loss']
        self.p.plot_loss_overlay(results['epoch_val_losses'], xlab='Epoch', ylab='loss',
                              save_name=os.path.join(self.output_dir,'Epoch_losses_Val.png'),
                               val_frequency=self.config['val_frequency'])
        
        #Plot validation unweighted loglosses
        val_loglosses = {}
        for key, val in results['epoch_val_losses'].items():
            val_loglosses[key] = np.log(np.array(val))
        self.p.plot_loss_overlay(val_loglosses, xlab='Epoch', ylab="logloss",
                              save_name=os.path.join(self.output_dir,'Epoch_LogLosses_Val.png'),
                               val_frequency=self.config['val_frequency'])
            
        #Plot validation unweighted as one group
        train_vs_val = {}
        train_vs_val['Train'] = results['epoch_losses']['loss']
        train_vs_val['Val.'] = results['sum_val_losses']['Val.']
        self.p.plot_loss_overlay(train_vs_val, 
                                 xlab='Epoch', ylab='Loss',
                            save_name=os.path.join(self.output_dir,'Epoch_losses_Train_Val.png'),
                               val_frequency=self.config['val_frequency'])
        
        #Plot weighted loss per sample
        results['epoch_val_weighted_losses']['Train'] = results['weighted_losses']['loss']
        self.p.plot_loss_overlay(results['epoch_val_weighted_losses'], 
                                 xlab='Epoch', ylab='Weighted loss',
                            save_name=os.path.join(self.output_dir,'Weighted_epoch_losses_Val.png'),
                               val_frequency=self.config['val_frequency'])
        
        #Plot weighted logloss per sample
        weighted_val_loglosses = {}
        for key, val in results['epoch_val_weighted_losses'].items():
            weighted_val_loglosses[key] = np.log(val)
        self.p.plot_loss_overlay(weighted_val_loglosses, 
                                 xlab='Epoch', ylab='Weighted logloss',
                        save_name=os.path.join(self.output_dir,'Weighted_epoch_LogLosses_Val.png'),
                               val_frequency=self.config['val_frequency'])
        
        train_vs_val = {}
        train_vs_val['Train'] = results['weighted_losses']['loss']
        train_vs_val['Val.'] = results['sum_weighted_val_losses']['Val.']
        self.p.plot_loss_overlay(train_vs_val, 
                                 xlab='Epoch', ylab='Loss',
                            save_name=os.path.join(self.output_dir,'Weighted_Epoch_losses_Train_Val.png'),
                               val_frequency=self.config['val_frequency'])
        
        
        #Any other plots needed?
        ...
    
    
    def save_training(self, train_results, save_best=True):
        
        #Save the model and optimizer
        

        if save_best:
            torch.save(self.best_model.state_dict(), os.path.join(self.output_dir, 'model_state_dict.pt'))
            torch.save(self.best_optimizer.state_dict(), os.path.join(self.output_dir, 'optimizer_state_dict.pt'))
            
            summary = []
            summary.append(f"Saved best model at epoch {self.best_val_epoch} with validation loss: {self.best_val_loss}\n")
            with open(os.path.join(self.output_dir, 'training_run_result.txt'), 'w') as f:
                f.writelines(summary)
                
            self.model = self.best_model
                
        else:
            torch.save(self.model.state_dict(), os.path.join(self.output_dir, 'model_state_dict.pt'))
            torch.save(self.optimizer.state_dict(), os.path.join(self.output_dir, 'optimizer_state_dict.pt'))
        
        with open(os.path.join(self.output_dir, 'saved_train_results.pkl'), 'wb') as f:
            pickle.dump(train_results, f)
            
        
        
                        
    def train(self, train_loader, val_loader):
        

        train_results = self.train_model(train_loader, val_loader)
        
        self.save_training(train_results)
        self.make_training_outputs(**train_results)
        
        print("Finished running training.")
        return self.model

        
    
                    
                
if __name__ == '__main__':
    
    #Read config input
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config",default="configs/training_config.yaml", help="Choose the master config to use")
    args = parser.parse_args()
    
    #Get the model
    from model.model_getter import get_model
    from preprocessing.dataset_io import DatasetHandler
    import os 
    from preprocessing.dataset import data_set
    from torch.utils.data import DataLoader
    
    with open(args.config,'r') as f:
        conf = yaml.safe_load(f)

    model = get_model(conf)
    
    
    dh = DatasetHandler(conf, job_name='TestRun')
    train, val, test = dh.split_dataset(use_val=dh.config['validation_set'], 
                use_eventnumber=dh.config.get('use_eventnumber',None))
    
    train_data = data_set(train)
    test_data = data_set(test)
    train_loader = DataLoader(train_data, batch_size=dh.config['batch_size'], shuffle=True, drop_last=True)    
    test_loader = DataLoader(test_data, batch_size=1)
    
    if len(val) != 0:
        val_data = data_set(val)
        val_loader = DataLoader(val_data, batch_size=2056)
    else:
        val_loader=None
    
    
    #Train the model
    t = Trainer(model, config=conf, output_dir=dh.output_dir)
    model = t.train(train_loader, val_loader=val_loader)
    
    
    

    

    
     