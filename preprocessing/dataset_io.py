'''
Class to hold the I/O of the dataset work,
along with some initial preprocessing.

Functions include:
- Cleaning
- Scaling
- Train/Test Split


Jack Harrison 23/08/2022
'''

import pandas as pd
import numpy as np
import pickle
import yaml
import os

#Scaling includes
from sklearn.preprocessing import StandardScaler

#Splitting includes
from sklearn.model_selection import train_test_split



class DatasetHandler():
    
    def __init__(self, dataset_config=None):
        
        with open(dataset_config, 'r') as f:
            self.config = yaml.safe_load(f)
            
        data_path = os.path.join(self.config['samples_path'],self.config['feather_file'])
        
        self.get_dataset(data_path, chosen_samples=self.config.get('chosen_samples',None))
        self.calculate_mcweight()
        self.reduce_to_training_variables(self.config['training_variables'])
        self.clean_dataset()
        self.scale_variables()
        
        
    def get_dataset(self, infile, chosen_samples=None):
        
        self.data = pd.read_feather(infile)
        if chosen_samples:
            self.data = self.data.loc[self.data['sample'].isin(chosen_samples)]
        
        
    #Function to calculate the MC weights from input datagframe
    def calculate_mcweight(self):

        total_lum = 138965.16
        self.data.loc[self.data['RunYear'].isin([2015,2016]), 'lumi_scale'] = 36207.66*(1/total_lum)
        self.data.loc[self.data['RunYear'].isin([2017]), 'lumi_scale'] = 44307.4*(1/total_lum)
        self.data.loc[self.data['RunYear'].isin([2018]), 'lumi_scale'] = 58450.1*(1/total_lum)

        self.data['weight'] = self.data['lumi_scale']*self.data['custTrigSF_TightElMediumMuID_FCLooseIso_DLT']*self.data['weight_pileup']*self.data['jvtSF_customOR']*self.data['bTagSF_weight_DL1r_77']*self.data['weight_mc']*self.data['xs']*self.data['lep_SF_CombinedLoose_0']*self.data['lep_SF_CombinedLoose_1']*self.data['lep_SF_CombinedLoose_2']*self.data['lep_SF_CombinedLoose_3']/self.data['totalEventsWeighted']

    
    def reduce_to_training_variables(self, training_variables):
        self.data = self.data[training_variables]
        
        
    def clean_dataset(self, r_negative=True, r_zero=True, r_duplicates=True):
        
        num_zero_weights = len(self.data.loc[self.data['weight']==0])
        num_negative_weights = len(self.data.loc[self.data['weight']<0])
        print(f"Found {num_zero_weights} events with zero weights. \nFound {num_negative_weights} events with negative weights.")
        
        if r_zero:
            self.data = self.data.loc[self.data['weight']!=0]
        if r_negative:
            self.data = self.data.loc[self.data['weight']>=0]
        if r_duplicates:
            # Check for duplicate rows
            before = len(self.data)
            self.data = self.data.drop_duplicates()
            print(f"Removing duplicates: {before} -> {len(self.data)}")
            
        #TODO: 
        #include dropping nans
         
    
    def scale_variables(self):
        
        self.scalers = {}
        for col in self.data.columns:
    
            scaler = StandardScaler()
            if col == 'weight':
                
                #Adjust the weights to be centered on 1
                self.data.loc[:,'scaled_weight'] = self.data.loc[:,col]/self.data[col].sum()
                continue
        
            if col == 'sample':
                continue
        
            self.data.loc[:, col] = scaler.fit_transform(np.array(self.data[col]).reshape(len(self.data[col]),1))
            print(f"Scaled {col} to mean: {self.data[col].mean()}")
            self.scalers[col] = scaler
            
    
    def save_scalers(self, output_dir):
        
        #Save the scalers used for each column
        scaler_folder = os.path.join(output_dir, 'scalers')
        if not os.path.exists(scaler_folder):
            os.makedirs(scaler_folder)
    
        for col, sc in self.scalers.items():
            pickle.dump(sc, open(os.path.join(scaler_folder,col+'_scaler.pkl'),'wb'))
            
            
    def split_per_sample(self, val=False):
        
        self.test_data = pd.DataFrame()
        print("Length of dataset: ", len(self.data))
        
        if val:
            fraction = self.config['validation_fraction']
        else:
            fraction = self.config['test_fraction']
            
        for sample in list(self.data['sample'].unique()):
            sample_data = self.data.loc[self.data['sample'] == sample]
            if len(sample_data) == 0:
                continue
            if len(sample_data) * fraction < 1:
                continue
                
            train, test = train_test_split(sample_data, test_size=fraction)
            
            self.test_data = pd.concat([self.test_data, test])
            self.data.drop(index=test.index.values, axis=0, inplace=True)
            
            #TODO:
            # - add this into verbose mode
            #print(f"Split {sample} by removing {len(test)} events.")
        
        print(f"Removed: {len(self.test_data)}, Remaining length: {len(self.data)}")
        return self.data, self.test_data