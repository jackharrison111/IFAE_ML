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
from tqdm import tqdm
from time import perf_counter



class Trainer():
    
    def __init__(self, model, config, sample_groupings={}):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        
        self.reversed_groupings = sample_groupings
        self.config = config
        
        date = dt.datetime.strftime(dt.datetime.now(),"%H%M-%d-%m-%Y")
        self.output_dir = 'outputs/VAE_'+date
        if self.config['test_dump']:
            self.output_dir = 'outputs/test_dump'
        
        
        self.make_optimizer()
        
        
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
        
        
    def train(self, dataloader, val_loader=None):
        
        
        print(f"Using available: {self.device}")
        
        epoch_losses = []
        validation_losses = []
        weight_loss = self.config['weight_loss']
        
        for epoch in range(self.config['num_epochs']):
            epoch_loss = 0
            for idx, (data, weights, samples, sc_weight) in enumerate(dataloader):
                
                self.model.train()
                data = data.to(self.device)
                sc_weight = sc_weight.to(self.device)
                self.optimizer.zero_grad()
                
                if idx % 50 == 0:
                    print(f"Done {round(idx/len(dataloader),2)}%")
                
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
            if val_loader and epoch % self.config['val_frequency'] == 0:
                
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
                    #if weight_loss:
                    #    loss = torch.dot(added_weight_factor*sc_weights, loss)
                    #else:
                    #    loss = torch.sum(loss)

                    group = self.reversed_groupings.get(samples,'All')
                    running_sum = val_losses.get(group,0) + loss.item()
                    running_counts = val_losses.get(group,0) + 1
                    val_losses[group] = running_sum
                    val_counts[group] = running_counts
                    
                for key in val_counts.keys():
                    val_losses[key] = val_losses[key]/val_counts[key]
                validation_losses.append(val_losses)
                print('Epoch {}: Validation Loss {}'.format(epoch, val_losses))
        
        #TODO:
        # - add early stopping
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        return epoch_losses, validation_losses
    
    def save_training(self, output_dir):
        
        #Save the model and optimizer
        torch.save(self.model.state_dict(), os.path.join(output_dir, 'model_state_dict.pt'))
        torch.save(self.optimizer.state_dict(), os.path.join(output_dir, 'optimizer_state_dict.pt'))
                
                
                
if __name__ == '__main__':
    
    #Read config input
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config",default="configs/training_config.yaml", help="Choose the master config to use")
    args = parser.parse_args()
    
    
    from preprocessing.dataset_io import DatasetHandler
    
    dh = DatasetHandler(args.config)
    
    data, val = dh.split_per_sample(val=True)
    #data, test = dh.split_per_sample()
    #print(len(data), len(val_data), len(test_data))
    
    from preprocessing.dataset import data_set
    train_data = data_set(data)
    val_data = data_set(val)
    
    #test_data = data_set(test)
    #print(len(test_data))
    
    #Make a dataloader
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_data, batch_size=dh.config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=1)
    
    
    from model.autoencoder import VAE, AE

    useful_columns = [col for col in data.columns if col not in ['sample','weight', 'scaled_weight']]
    enc_dim = [len(useful_columns),4]
    dec_dim = [4,len(useful_columns)]
    z_dim = 2
    model_type = dh.config['model_type']
    if model_type == 'AE':
        model = AE(enc_dim, dec_dim, z_dim)
    elif model_type == 'VAE':
        model = VAE(enc_dim, dec_dim, z_dim)
    
    t = Trainer(model, config=dh.config)
    t.train(train_loader, val_loader)