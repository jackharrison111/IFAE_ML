# importing the required libraries
import torch
from torch.utils.data import Dataset


# Dataset class, needs to be built from pandas dataframe
class data_set(Dataset):
    
    def __init__(self, df):
        
        self.weights = None
        self.samples = None
        self.scaled_weights  = None
        
        if 'weight' in df.columns:
            self.weights = torch.tensor(df['weight'].values, dtype=torch.float32)
            df.drop('weight', axis=1, inplace=True)
            
        if 'scaled_weight' in df.columns:
            self.scaled_weights = torch.tensor(df['scaled_weight'].values, dtype=torch.float32)
            df.drop('scaled_weight', axis=1, inplace=True)
            
        if 'sample' in df.columns:
            self.samples = df['sample']
            df.drop('sample', axis=1, inplace=True)
        
        self.data = torch.tensor(df.values, dtype=torch.float32)
  
    def __len__(self):
        return len(self.data)
  
    def __getitem__(self, index):
        
        #weights =  self.weights[index] if self.weights!=None else None
        #samples =  [] if self.samples == None else self.samples[index]
        #TODO: ADD CHECKING FOR IF NO WEIGHTS OR SAMPLES ARE PASSED
        sc_weight = self.scaled_weights[index] if self.scaled_weights != None else None
        return self.data[index], self.weights[index], self.samples.iloc[index], sc_weight
