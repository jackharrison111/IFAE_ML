#Steps are: 

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
        
        self.make_optimizer(learning_rate=self.config['learning_rate'])
        self.p = Plotter()
        
        
    def make_optimizer(self, learning_rate=1e-2):
        
        if self.config['optimizer'] == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        else:
            self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        
        
    def train_model(self, dataloader, val_loader=None):
        
        print(f"Using available: {self.device}")
        
        epoch_losses, epoch_mses, epoch_klds, epoch_loglosses = [], [], [], []
        weighted_epoch_losses, weighted_epoch_loglosses = [], []
        validation_losses, validation_loglosses = {}, {}
        validation_losses_weighted, validation_loglosses_weighted = {}, {}

        
        weight_loss = self.config['weight_loss']
        s = perf_counter()
        for epoch in range(self.config['num_epochs']):
            
            l1 = perf_counter()
            epoch_loss, epoch_counts, epoch_kld, epoch_mse = 0,0,0,0
            weighted_epoch_loss = 0
            
            self.model.train()

            for idx, data_dict in enumerate(dataloader):
                
                data = data_dict['data']
                sc_weight = data_dict['scaled_weight']
                
                data = data.to(self.device)
                sc_weight = sc_weight.to(self.device)
                self.optimizer.zero_grad()
                
                #Feed data into model
                outputs = self.model(data)
                
                #Get the loss
                losses = self.model.loss_function(outputs)
            
                
                # Feeding a batch into the network to obtain the output image, mu, and logVar
                out, mu, logVar = self.model(data)
                loss, mse, kld = self.model.loss_function(out, data, mu, logVar, variational=self.model.variational, beta=self.config["beta"])
                num_examples = loss.shape[0]
                
                epoch_loss += torch.sum(loss).item()
                
                #Multiply the loss by the weights when updating
                loss = torch.dot(self.config['added_weight_factor']*sc_weight, loss) if weight_loss else torch.dot(sc_weight, loss)
                if mse is not None:
                    mse = torch.dot(self.config['added_weight_factor']*sc_weight, mse) if weight_loss else torch.dot(sc_weight, loss)
                    kld = torch.dot(self.config['added_weight_factor']*sc_weight, kld) if weight_loss else torch.dot(sc_weight, loss)

                weighted_epoch_loss += loss.item()
                epoch_counts += num_examples
                if mse is not None:
                    epoch_mse += mse.item()
                    epoch_kld += kld.item()
                
                
                # Backpropagation based on the loss
                loss.backward()
                self.optimizer.step()
                
            normalised_epoch_loss = epoch_loss/epoch_counts
            normalised_weighted_epoch_loss = weighted_epoch_loss/epoch_counts
            normalised_epoch_mse = epoch_mse/epoch_counts
            normalised_epoch_kld = epoch_kld/epoch_counts
            epoch_losses.append(normalised_epoch_loss)
            weighted_epoch_losses.append(normalised_weighted_epoch_loss)
            epoch_loglosses.append(np.log(normalised_epoch_loss))
            weighted_epoch_loglosses.append(np.log(normalised_weighted_epoch_loss))
            epoch_mses.append(normalised_epoch_mse)
            epoch_klds.append(normalised_epoch_kld)
            print(f"Epoch: {epoch}, Loss: {normalised_epoch_loss}, weighted: {normalised_weighted_epoch_loss}, MSE: {normalised_epoch_mse}, KLD: {normalised_epoch_kld}")
            
            
            self.model.eval()
            #And then output a validation loss per sample (per group)
            if val_loader and self.config['val_frequency'] > 0:
                if epoch % self.config['val_frequency'] == 0:
    
                    val_losses, val_losses_weighted, val_counts = {}, {}, {}
                    

                    for idx, data_dict in enumerate(val_loader):
                
                        data = data_dict['data']
                        sc_weights = data_dict['scaled_weight']
                        samples = data_dict['sample']
                        weights = data_dict['weight']
                        
                        data = data.to(self.device)
                        samples=samples[0]
                        out, mu, logVar = self.model(data)
                        loss, mse, kld = self.model.loss_function(out, data, mu, logVar, 
                                                            variational=self.model.variational, beta=self.config["beta"])
                        num_examples = loss.shape[0]
                        
                        weighted_loss = torch.dot(self.config['added_weight_factor']*sc_weights, loss) if weight_loss else torch.dot(sc_weights, loss)
                        
                        #Don't multiply loss by weights when testing
                        loss = torch.sum(loss)
                        num_examples=len(weights)
                        
                        if mse is not None:
                            mse = torch.sum(mse)
                            kld = torch.sum(kld)

                        group = self.reversed_groupings.get(samples,'All')
                       
                        running_sum = val_losses.get(group,0) + loss.item()
                        running_weighted_sum = val_losses_weighted.get(group,0) + weighted_loss.item()
                        running_counts = val_counts.get(group,0) + num_examples
                        
                        val_losses[group] = running_sum
                        val_losses_weighted[group] = running_weighted_sum
                        val_counts[group] = running_counts

                    for key in val_counts.keys():
                        #Average the loss per sample
                        key_loss = val_losses[key]/val_counts[key]
                        key_weighted_loss = val_losses_weighted[key]/val_counts[key]
                        val_losses[key] = key_loss
                        val_losses_weighted[key] = key_weighted_loss
                        
                        loss_store = validation_losses.get(key, [])
                        logloss_store = validation_loglosses.get(key, [])
                        logloss_store.append(np.log(key_loss))
                        loss_store.append(key_loss)
                        validation_losses[key] = loss_store
                        validation_loglosses[key] = logloss_store
                        
                        loss_store_w = validation_losses_weighted.get(key, [])
                        logloss_store_w = validation_loglosses_weighted.get(key, [])
                        logloss_store_w.append(np.log(key_weighted_loss))
                        loss_store_w.append(key_weighted_loss)
                        validation_losses_weighted[key] = loss_store_w
                        validation_loglosses_weighted[key] = logloss_store_w
                        
                        
                        
                    print(f"Epoch {epoch}: Validation Loss {val_losses}, Weighted: {val_losses_weighted}")
            l2 = perf_counter()
            print(f"Finished epoch... time taken: {round(l2-l1,2)}s.")
            
        
        
        self.save_training()
        
        epoch_axis = [i for i in range(len(epoch_losses))]
        self.p.plot_loss(epoch_axis, epoch_losses, xlab='Epoch', ylab='Loss',
                   save_name=os.path.join(self.output_dir,'Epoch_losses.png'))
        self.p.plot_loss(epoch_axis, epoch_loglosses, xlab='Epoch', ylab='Epoch logloss',
                   save_name=os.path.join(self.output_dir,'Epoch_loglosses.png'))
        
        loss_parts = {'MSE loss' : epoch_mses, 'K-L Divergence' : epoch_klds, 'Total' : epoch_losses}
        self.p.plot_loss_overlay(loss_parts, xlab='Epoch', ylab='Loss',
                              save_name=os.path.join(self.output_dir,'Epoch_losses_Separated.png'),
                               val_frequency=1)
    
    
        #Non weighted
        validation_losses['Train'] = epoch_losses
        validation_loglosses['Train'] = epoch_loglosses
        self.p.plot_loss_overlay(validation_losses, xlab='Epoch', ylab='Loss',
                              save_name=os.path.join(self.output_dir,'Epoch_losses_Val.png'),
                               val_frequency=self.config['val_frequency'])
        
        self.p.plot_loss_overlay(validation_loglosses, xlab='Epoch', ylab='Logloss',
                              save_name=os.path.join(self.output_dir,'Epoch_LogLosses_Val.png'),
                               val_frequency=self.config['val_frequency'])
            
        #Weighted  
        validation_losses_weighted['Train'] = weighted_epoch_losses
        validation_loglosses_weighted['Train'] = weighted_epoch_loglosses
        
        self.p.plot_loss_overlay(validation_losses_weighted, 
                                 xlab='Epoch', ylab='Weighted Loss',
                            save_name=os.path.join(self.output_dir,'Weighted_epoch_losses_Val.png'),
                               val_frequency=self.config['val_frequency'])
        
        self.p.plot_loss_overlay(validation_loglosses_weighted, 
                                 xlab='Epoch', ylab='Weighted Logloss',
                        save_name=os.path.join(self.output_dir,'Weighted_epoch_LogLosses_Val.png'),
                               val_frequency=self.config['val_frequency'])
            
            
            
        print(f"Finished training... time taken: {round((l2-s),2)/60}mins.")
        return epoch_losses, epoch_loglosses, validation_losses
    
    
    def save_training(self):
        
        #Save the model and optimizer
        torch.save(self.model.state_dict(), os.path.join(self.output_dir, 'model_state_dict.pt'))
        torch.save(self.optimizer.state_dict(), os.path.join(self.output_dir, 'optimizer_state_dict.pt'))
        
        
                
    def run(self, train_loader, val_loader):
        

        epoch_losses, epoch_loglosses, validation_losses = self.train_model(train_loader, val_loader)
        self.save_training()
        print("Finished training.")
        
        
        self.p.plot_loss([i for i in range(len(epoch_losses))], epoch_losses, xlab='Epoch', ylab='Epoch loss',
                   save_name=os.path.join(self.output_dir,'Epoch_losses.png'))
        
        self.p.plot_loss([i for i in range(len(epoch_losses))], epoch_loglosses, xlab='Epoch', ylab='Epoch logloss',
                   save_name=os.path.join(self.output_dir,'Epoch_Loglosses.png'))
    
        validation_losses['Train'] = epoch_losses
        self.p.plot_loss_overlay(validation_losses, xlab='Epoch', ylab='Epoch loss',
                              save_name=os.path.join(self.output_dir,'Epoch_losses_Val.png'),
                               val_frequency=self.config['val_frequency'])
        
    
                    
                
if __name__ == '__main__':
    
    #Read config input
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config",default="configs/training_config.yaml", help="Choose the master config to use")
    args = parser.parse_args()
    
    #Get the model
    from model.autoencoder import VAE, AE
    from model.norm_flow import NormFlow
    
    with open(args.config,'r') as f:
        conf = yaml.safe_load(f)
    
    
    useful_columns = [col for col in conf['training_variables'] if col not in ['sample','weight', 'scaled_weight','eventNumber']]
    enc_dim = [len(useful_columns),8,6]
    dec_dim = [6,8,len(useful_columns)]
    z_dim = 4
    model_type = conf['model_type']
    
    if model_type == 'AE':
        model = AE(enc_dim, dec_dim, z_dim)
    elif model_type == 'VAE':
        model = VAE(enc_dim, dec_dim, z_dim)
        
    elif model_type == 'norm_flow':
        model = NormFlow(len(useful_columns))
    
    
    #Train
    t = Trainer(model, config=args.config)
    t.run()
    
    

    

    
     