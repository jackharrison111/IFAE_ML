#############################################
#
# Classes for creating a VAE + AE
# 
#
# Jack Harrison 15/07/2022
#############################################

import torch
import torch.nn as nn
import torch.nn.functional as F


#Class to store the encoder 
class Encoder(nn.Module):
    
    def __init__(self, dimensions, latent_dim, variational=False, func='Tanh'):
        
        super().__init__()
        
        func='LeakyReLU'
        func='ELU'
        func='Tanhshrink'
        func='Sigmoid'
        
        layers = []
        self.variational = variational
        
        for i, dim in enumerate(dimensions):
            
            if dim == dimensions[-1]:
                if self.variational:   #Don't add the single output to z dim
                    break
                layers.append(
                    nn.Sequential(
                        nn.Linear(dim, latent_dim),
                        #getattr(nn, func)()
                        #nn.ReLU()
                    )
                )
                break
                
            layers.append(
                nn.Sequential(
                    nn.Linear(dim, dimensions[i+1]),
                    getattr(nn, func)()
                    #nn.ReLU()
                )
            )
            
        self.encoder = nn.Sequential(*layers)
        
        if self.variational:
            self.f_mu = nn.Linear(dimensions[-1], latent_dim)
            self.f_var = nn.Linear(dimensions[-1], latent_dim)
            
        
    def forward(self, data):
        
        result = self.encoder(data)
        
        if self.variational:
            mu = self.f_mu(result)
            var = self.f_var(result)
            return mu, var
        
        return result, None
            
        
#Class to store the decoder
class Decoder(nn.Module):
    
    def __init__(self, dimensions, latent_dim, func='Tanh'):
        super().__init__()
        
        func = 'LeakyReLU'
        func='ELU'
        func='Tanhshrink'
        func='Sigmoid'
        layers = []
        layers.append(
            nn.Sequential(
                    nn.Linear(latent_dim, dimensions[0]),
                    getattr(nn, func)()
                    #nn.ReLU()
                )
            )
        for i, dim in enumerate(dimensions[:-1]):
            if dim == dimensions[-2]:
                layers.append(   #Don't add ReLU output
                   nn.Sequential(
                        nn.Linear(dim, dimensions[i+1])
                    )
                )
                break
            layers.append(
                nn.Sequential(
                    nn.Linear(dim, dimensions[i+1]),
                    getattr(nn, func)()
                    #nn.ReLU()
                )
            )
            
        
        self.decoder = nn.Sequential(*layers)
        
    def forward(self, data):
        #Add chosen output function
        #func_choice = 'Tanh'
        #func = getattr(nn, func_choice)()
        prediction = self.decoder(data)
        return prediction
        
        
#Class to make a regular autoencoder
class AE(nn.Module):
    
    def __init__(self, enc_dim, dec_dim, latent_dim):
        super().__init__()
        self.variational=False
        self.encoder = Encoder(enc_dim, latent_dim)
        self.decoder = Decoder(dec_dim, latent_dim)
        
    def loss_function(self, recon_x, x, mu, log_var, **kwargs):
        #BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        MSE = F.mse_loss(x, recon_x, reduction='none')   #Works it out element-wise
        MSE = torch.mean(MSE, dim=1)   #Average across features, leaving per-example MSE
        return MSE, None, None
        
    def forward(self, data):
        z,_ = self.encoder(data)
        prediction = self.decoder(z)
        return prediction, z, _
        

        
#Class to make a variational autoencoder
class VAE(nn.Module):
    
    def __init__(self,enc_dimensions, dec_dimensions, latent_dim):
        
        super().__init__()
        self.variational=True
        self.encoder = Encoder(enc_dimensions, latent_dim, variational=self.variational)
        self.decoder = Decoder(dec_dimensions, latent_dim)
        self.latent_dim = latent_dim
        
    def loss_function(self, recon_x, x, mu, log_var, beta=None, **kwargs):
        
        #BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        MSE = F.mse_loss(x, recon_x, reduction='none')   #Works it out element-wise
        MSE = torch.mean(MSE, dim=1)   #Average across features, leaving per-example MSE
        KLD = 1 + log_var - mu.pow(2) - log_var.exp()   #Again worked out element-wise
        KLD = -0.5 * torch.sum(KLD, dim=1)   #Sum across features, leaving per-example KLD
        
        if beta is not None:
            return (1/beta)*MSE + beta*KLD, (1/beta)*MSE, beta*KLD
        else:
            return MSE + KLD, MSE, KLD
    
        
    def forward(self, data):
        
        self.input_shape = data.shape
        self.z_mu, self.z_logvar = self.encoder(data)
        
        #Perform reparametrization trick
        std = torch.exp(self.z_logvar/2)    
        epsilon = torch.randn_like(std)               #Samples a N(0,1) in the shape of std
        z_sample = epsilon.mul(std).add_(self.z_mu)   #Does elementwise multiplication and addition
        
        prediction = self.decoder(z_sample).view(self.input_shape)
        return prediction, self.z_mu, self.z_logvar
        
        
    def sample(self, num_samples=1):
        z_sample = torch.randn(num_samples, self.latent_dim)
        prediction = self.decoder(z_sample)
        return prediction
        
        
        
        
if __name__ == '__main__':
    
    encoder = [20,10,5]
    decoder = [5,10,20]
    latent_dim = 1
    
    vae = VAE(encoder,decoder, latent_dim)