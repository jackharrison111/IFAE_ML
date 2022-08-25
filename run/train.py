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
import torch
import torch.optim as optim
import torch.nn.functional as F

import os
import datetime as dt
from tqdm.auto import tqdm
from time import perf_counter
import yaml

#Dataset includes
from preprocessing.dataset_io import DatasetHandler
from preprocessing.dataset import data_set
from torch.utils.data import DataLoader

#Plotting
from plotting.plot_results import Plotter

class Trainer():
    
    def __init__(self, model, config, sample_groupings={}):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        
        
        if type(config) == str:
            with open(config, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = config
        
        self.reversed_groupings = {}
        for key, values in self.config['groupings'].items():
            for val in values:
                self.reversed_groupings[val] = key
        
        date = dt.datetime.strftime(dt.datetime.now(),"%H%M-%d-%m-%Y")
        self.output_dir = 'outputs/VAE_'+date
        if self.config['test_dump']:
            self.output_dir = 'outputs/test_dump'
        
        
        self.make_optimizer()
        
        
    def get_dataset(self, config=None):
        
        if not config:
            config = self.config
        dh = DatasetHandler(config)
        data, val = dh.split_per_sample(val=True)
        
        train_data = data_set(data)
        val_data = data_set(val)
        
        train_loader = DataLoader(train_data, batch_size=dh.config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_data, batch_size=1)
        
        return train_loader, val_loader
    
        
    def loss_function(self, recon_x, x, mu, log_var, variational=True):
        #BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        MSE = F.mse_loss(x, recon_x, reduction='none')   #Works it out element-wise
        MSE = torch.mean(MSE, dim=1)   #Average across features, leaving per-example MSE

        if variational:
            KLD = 1 + log_var - mu.pow(2) - log_var.exp()   #Again worked out element-wise
            KLD = -0.5 * torch.sum(KLD, dim=1)   #Sum across features, leaving per-example KLD
            return MSE + KLD, MSE, KLD
        else:
            return MSE, None, None
        
        
    def make_optimizer(self, learning_rate=1e-2):
        
        #optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        
        
    def train_vae(self, dataloader, val_loader=None):
        
        print(f"Using available: {self.device}")
        
        epoch_losses = []
        validation_losses = {}
        weight_loss = self.config['weight_loss']
        
        for epoch in range(self.config['num_epochs']):
            epoch_loss = 0
            for idx, (data, weights, samples, sc_weight) in tqdm(enumerate(dataloader)):
                
                self.model.train()
                data = data.to(self.device)
                sc_weight = sc_weight.to(self.device)
                self.optimizer.zero_grad()
                
                #if idx % 50 == 0:
                #    print(f"Done {round(idx/len(dataloader),2)}%")
                
                # Feeding a batch into the network to obtain the output image, mu, and logVar
                out, mu, logVar = self.model(data)
                loss, mse, kld = self.loss_function(out, data, mu, logVar, variational=self.model.variational)
                
                #Multiply the loss by the weights
                loss = torch.dot(self.config['added_weight_factor']*sc_weight, loss) if weight_loss else torch.sum(loss)
                
                # Backpropagation based on the loss
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            epoch_loss = epoch_loss / len(dataloader)
            epoch_losses.append(epoch_loss)
            print(f"Epoch: {epoch}, Loss: {epoch_loss}")
                
                
            #And then output a validation loss per sample (per group)
            if val_loader and self.config['val_frequency'] > 0:
                if epoch % self.config['val_frequency'] == 0:
    
                    val_losses = {}
                    val_counts = {}

                    self.model.eval()
                    for idy, (data, weights, samples, sc_weights) in enumerate(val_loader):

                        data = data.to(self.device)
                        samples=samples[0]
                        out, mu, logVar = self.model(data)
                        loss, mse, kld = self.loss_function(out, data, mu, logVar, variational=self.model.variational)

                        #Multiply the loss by the weights
                        loss = torch.dot(self.config['added_weight_factor']*sc_weights, loss) if weight_loss else torch.sum(loss)

                        group = self.reversed_groupings.get(samples,'All')
                        running_sum = val_losses.get(group,0) + loss.item()
                        running_counts = val_losses.get(group,0) + 1
                        val_losses[group] = running_sum
                        val_counts[group] = running_counts

                    for key in val_counts.keys():
                        key_loss = val_losses[key]/val_counts[key]
                        val_losses[key] = key_loss
                        loss_store = validation_losses.get(key, [])
                        loss_store.append(key_loss)
                        validation_losses[key] = loss_store
                        
                    #validation_losses.append(val_losses)
                    print('Epoch {}: Validation Loss {}'.format(epoch, val_losses))
        
        #TODO:
        # - add early stopping
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        return epoch_losses, validation_losses
    
    def save_training(self):
        
        #Save the model and optimizer
        torch.save(self.model.state_dict(), os.path.join(self.output_dir, 'model_state_dict.pt'))
        torch.save(self.optimizer.state_dict(), os.path.join(self.output_dir, 'optimizer_state_dict.pt'))
                
    def run(self):
        
        train_loader, val_loader = t.get_dataset()
        epoch_losses, validation_losses = t.train_vae(train_loader, val_loader)
        t.save_training()
        print("Finished training.")
        
        p = Plotter()
        p.plot_scatter([i for i in range(len(epoch_losses))], epoch_losses, xlab='Epoch', ylab='Epoch loss',
                   save_name=os.path.join(self.output_dir,'Epoch_losses.png'))
    
        validation_losses['Train'] = epoch_losses
        p.plot_scatter_overlay(validation_losses, xlab='Epoch', ylab='Epoch loss',
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
    with open(args.config,'r') as f:
        conf = yaml.safe_load(f)
    
    
    useful_columns = [col for col in conf['training_variables'] if col not in ['sample','weight', 'scaled_weight']]
    enc_dim = [len(useful_columns),8]
    dec_dim = [8,len(useful_columns)]
    z_dim = 4
    model_type = conf['model_type']
    if model_type == 'AE':
        model = AE(enc_dim, dec_dim, z_dim)
    elif model_type == 'VAE':
        model = VAE(enc_dim, dec_dim, z_dim)
    
    
    #Train
    t = Trainer(model, config=args.config)
    t.run()
    
    

    

    
     