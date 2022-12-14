'''
Class for a feed-forward neural network,
used to compare between supervised and 
unsupervised.


Jack Harrison 23/08/2022
'''


import torch
import torch.nn as nn
import torch.nn.functional as F


class NN(nn.Module):
    
    def __init__(self, dimensions):
        
        super().__init__()
        func = 'Tanh'
        layers = []
        
        for i, dim in enumerate(dimensions):
            if dim == dimensions[-2]:
                layers.append(   #Don't add func to output
                   nn.Sequential(
                        nn.Linear(dim, dimensions[i+1])
                    )
                )
                break
            layers.append(
                    nn.Sequential(
                        nn.Linear(dim, dimensions[i+1]),
                        getattr(nn, func)()
                    )
                )
            
        self.network = nn.Sequential(*layers)
        
    def loss_function(self, input_x, target_x, w=None):
        
        loss = F.binary_cross_entropy(input_x, target_x, weight=w)*500
        output_dict = {'loss': loss}
        return output_dict
        
        
    def forward(self, data):
        
        result = torch.sigmoid(self.network(data))
        return result

    