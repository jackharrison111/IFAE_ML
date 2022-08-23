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
            layers.append(
                    nn.Sequential(
                        nn.Linear(dim, dimensions[i+1]),
                        getattr(nn, func)()
                        #nn.ReLU()
                    )
                )
            
        self.network = nn.Sequential(layers)
        
    def forward(self, data):
        
        result = nn.F.sigmoid(self.network(data))
        return result

    