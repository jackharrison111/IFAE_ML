# importing the required libraries
import torch
from torch.utils.data import Dataset


# Dataset class, needs to be built from pandas dataframe
class data_set(Dataset):
    
    def __init__(self, df):
        
        extra_cols = ['weight', 'scaled_weight', 'sample', 'eventNumber', 'label']
        if df is None:
            self.data = None
        elif len(df) != 0:
            self.format_dataset(df, extra_cols)    
            self.data = torch.tensor(df.values, dtype=torch.float32)
        else:
            self.data = None

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
        
        #TODO: Check if this is slow
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
        #Could try taking a list of names and then doing get item and append to output list 
        
        return output_dict
    
