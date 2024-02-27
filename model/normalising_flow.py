import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model.mlp import MLP
from model.core_normflows import NormalizingFlow

import normflows as nf

#Wrapper class for using a normalising flow
class NormFlow(nn.Module):
    
    def __init__(self, use_spline=True, input_dims=10, num_layers=8, hidden_layers=2, hidden_units=64, func_type='Tanh'):
        
        super().__init__()
        
        # Set up model        
        self.num_dims = input_dims
        self.num_layers = num_layers
        self.hidden_layers = hidden_layers
        self.hidden_units = hidden_units
        self.func_type = func_type
        self.increased_size = False
        
        if use_spline:
            
            print("Using autoregressive quadratic spline normalising flow.")
            # Define flows
            K = 8
            hidden_units = 64
            hidden_layers = 2
            flows = []
            for i in range(self.num_layers):
                flows += [nf.flows.AutoregressiveRationalQuadraticSpline(self.num_dims, 
                                                                         self.hidden_layers, 
                                                                         self.hidden_units)]
                flows += [nf.flows.LULinearPermute(self.num_dims)]
                
            # Set prior and q0
            base = nf.distributions.DiagGaussian(self.num_dims, trainable=False)

            # Construct flow model
            self.model = NormalizingFlow(q0=base, flows=flows)
            
    
        else:

            # Set up model
            print("Using RealNVP normalising flow.")
            # Define 2D Gaussian base distribution
            if self.num_dims % 2 != 0:
                self.num_dims += 1
                self.increased_size = True
                print("Increasing size of input dimension by 1, due to odd input numbers.")
                
            base = nf.distributions.base.DiagGaussian(self.num_dims)

            # Define list of flows
            flows = []
            hidden_block = [self.num_dims//2] + [self.hidden_units for i in range(self.hidden_layers)] + [self.num_dims]
            

            for i in range(self.num_layers):
                # Neural network with two hidden layers having 64 units each
                # Last layer is initialized by zeros making training more stable
                param_map = MLP(hidden_block, init_zeros=True, func_type=self.func_type, leaky=0.05)#, dropout=0.05)
                # Add flow layer
                flows.append(nf.flows.AffineCouplingBlock(param_map))
                # Swap dimensions
                flows.append(nf.flows.Permute(self.num_dims, mode='shuffle'))

            # Construct flow model
            self.model = NormalizingFlow(base, flows)

        
    #The loss function here is used as the way to train
    def loss_function(self, **kwargs):
        
        x = kwargs['data']
        loss = self.model.forward_kld(x, average=False)
        return {'loss' : loss, 'data': kwargs['data']}
        
        
    def forward(self, x):
        
        if self.increased_size:
            x = torch.cat([x, torch.zeros((len(x),1))], dim=1)
        
        return {'data':x}
        
    def get_anomaly_score(self, x, **kwargs):
        return self.anomaly_score(x['data'], **kwargs)
        
        
    def anomaly_score(self, x, **kwargs):
        log_prob = self.model.log_prob(x)
        
        if 'min_loss' in kwargs.keys():
            self.min_prob = kwargs['min_loss']
            self.max_prob = kwargs['max_loss']
            log_prob, _, _ = self.scale_log_prob(log_prob, min_prob=kwargs['min_loss'],
                                          max_prob=kwargs['max_loss'])
        
        return log_prob
    

    def scale_log_prob(self, log_probs, min_prob=None, max_prob=None, use_01=True):
    
        if use_01:
            log_probs = -log_probs
            
            #new_min_prob = max_prob
            #new_max_prob = min_prob
            #self.min_prob = new_min_prob
            #self.max_prob = new_max_prob
            
            #If we reverse them then we need to switch the max and the min?
        
        if min_prob is None:
            min_prob = log_probs.min()
            max_prob = log_probs.max()
            self.min_prob = min_prob
            self.max_prob = max_prob

        log_probs = (log_probs - self.min_prob) / (self.max_prob - self.min_prob)
        print(f"Scaled loglikelihood using {self.min_prob} , {self.max_prob} to mean: {log_probs.mean()}")
        return log_probs, min_prob, max_prob
        
    #Adding sampled function
    def sample(self, num_samples):
    
        z, log_q = self.model.sample(num_samples=num_samples)
        return z, log_q

    
    
    
if __name__ == '__main__':
    
    ...