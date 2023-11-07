# importing the required libraries
import torch
from torch.utils.data import Dataset


class data_set(Dataset):
    
    def __init__(self, df):
        metadata_cols = ['sample', 'eventNumber','label', 'index']
        weight_cols = ['weight', 'scaled_weight']

        if 'label' not in df.columns:
            metadata_cols.remove('label')
            
        self.data = torch.tensor(df.drop(columns=metadata_cols+weight_cols,errors='ignore').values, dtype=torch.float32)
        self.metadata = df[metadata_cols].to_dict(orient='records')
        self.weight = torch.tensor(df['weight'].values, dtype=torch.float32)
        self.scaled_weight = torch.tensor(df['scaled_weight'].values, dtype=torch.float32)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        item = {
            'data': self.data[index],
            'weight': self.weight[index],
            'scaled_weight': self.scaled_weight[index],
            **self.metadata[index]
        }
        
        return item
    
