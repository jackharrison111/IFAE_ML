#############################################
#
# Wrapper class for normalising flows
# 
#
# Jack Harrison 15/07/2022
#############################################


import torch
import torch.nn as nn
import torch.nn.functional as F
import normflows as nf

#Class to make a regular autoencoder
class NormFlow(nn.Module):
    
    def __init__(self, input_dims, layers, enc_dim):
        super().__init__()
        self.variational=False
        
        # Set up model
        self.num_dims = input_dims
        self.num_layers = layers
        self.enc_dim = enc_dim
        
        # Define 2D Gaussian base distribution
        base = nf.distributions.base.DiagGaussian(self.num_dims)

        # Define list of flows
        flows = []
        for i in range(self.num_layers):
            # Neural network with two hidden layers having 64 units each
            # Last layer is initialized by zeros making training more stable
            param_map = nf.nets.MLP([self.num_dims//2, self.enc_dim, self.enc_dim, self.num_dims], 
                                    init_zeros=True, leaky=0.05)
            # Add flow layer
            flows.append(nf.flows.AffineCouplingBlock(param_map))
            # Swap dimensions
            flows.append(nf.flows.Permute(num_dims, mode='swap'))

        # Construct flow model
        self.norm_flow = nf.NormalizingFlow(base, flows)
        
    def loss_function(self, recon_x, x, mu, log_var, **kwargs):
        loss = self.norm_flow.forward_kld(recon_x)
        return loss
        
    def forward(self, data):
        return data