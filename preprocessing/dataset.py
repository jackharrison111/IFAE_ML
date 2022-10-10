# importing the required libraries
import torch
from torch.utils.data import Dataset


# Dataset class, needs to be built from pandas dataframe
class data_set(Dataset):
    
    def __init__(self, df):
        
        extra_cols = ['weight', 'scaled_weight', 'sample', 'eventNumber', 'label']
        self.format_dataset(df, extra_cols)    
        self.data = torch.tensor(df.values, dtype=torch.float32)
        

    def format_dataset(self, df, cols):
        
        for col in cols:
            setattr(self, col, None)
            if col in df.columns:
                if col in ['weight', 'scaled_weight', 'label']:
                    setattr(self, col, torch.tensor(df[col].values, dtype=torch.float32))
                else:
                    setattr(self, col, df[col])
                df.drop(col, axis=1, inplace=True)

    def __len__(self):
        return len(self.data)
  

    def __getitem__(self, index):
        
        #TODO:
        #Return this as a dictionary ? See if possible
        #Torch tensors
        data = self.data[index] if self.data is not None else None
        weight = self.weight[index] if self.weight is not None else None
        sc_weight = self.scaled_weight[index] if self.scaled_weight is not None else None
        label = self.label[index] if self.label is not None else 0
        
        
        #Pandas series
        samples = self.sample.iloc[index] if self.sample is not None else None
        eventNumber = self.eventNumber.iloc[index].astype(int) if self.eventNumber is not None else 0
        
        output_dict = {
            'data' : data,
            'weight' : weight,
            'sample' : samples,
            'scaled_weight' : sc_weight,
            'label' : label,
            'eventNumber' : eventNumber
        }
        #print(output_dict)
        
        return output_dict
        '''
        #weights =  self.weights[index] if self.weights!=None else None
        #samples =  [] if self.samples == None else self.samples[index]
        #TODO: ADD CHECKING FOR IF NO WEIGHTS OR SAMPLES ARE PASSED
        sc_weight = self.scaled_weights[index] if self.scaled_weights != None else None
        return self.data[index], self.weights[index], self.samples.iloc[index], sc_weight
        '''
