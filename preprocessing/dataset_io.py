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

#Utils
from utils._utils import load_yaml_config, make_output_folder

#Scaling includes
from sklearn.preprocessing import StandardScaler

#Splitting includes
from sklearn.model_selection import train_test_split



class DatasetHandler():
    
    def __init__(self, dataset_config=None, scalers=None, job_name=None, out_dir=None):
        
        
        self.config = load_yaml_config(dataset_config)

        if out_dir:
            self.output_dir = out_dir
        else:
            root = os.path.join('results', job_name) if job_name is not None else os.path.join('results',self.config['out_folder'])
            self.output_dir = make_output_folder(self.config, root_loc=root)
        
        self.data_path = os.path.join(self.config['samples_path'], self.config['feather_file'])
        chosen_samples = self.config.get('chosen_samples', None)
        self.signal = False
        
        if scalers:
            self.signal=True
            self.data_path = os.path.join(self.config['samples_path'], self.config['signal_file'])
            chosen_samples = self.config.get('signal_samples',None)
        
        self.get_dataset(self.data_path, chosen_samples=chosen_samples)
        
        if len(self.data)!= 0:
            
            self.calculate_mcweight()
            
            #self.add_additional_weight()

            self.reduce_to_training_variables(self.config['training_variables'])
            

            self.clean_dataset(
                r_negative=self.config['remove_negative_weights'],
                r_zero=self.config['remove_zero_weights'])
            

            self.scale_variables(scalers=scalers)

        else:
            self.data = []
        
        
    def get_dataset(self, infile, chosen_samples=None):
        
        self.data = pd.read_feather(infile)

        if chosen_samples:
            before = len(self.data)
            self.data = self.data.loc[self.data['sample'].isin(chosen_samples)]

        if self.signal:
            if self.config['num_test_samples'] != -1:
                self.data = self.data[:self.config['num_test_samples']]
        else:
            if self.config['train_size'] != -1:
                self.data = self.data[:self.config['train_size']]
        
        
    #Function to calculate the MC weights from input dataframe
    def calculate_mcweight(self, total_lum=138965.16):

        if len(self.data) == 0:
            return None
        
        if self.config['mc_weight_choice'] == 'VLL_production':
            
            self.data.loc[self.data['RunYear'].isin([2015,2016]), 'lumi_scale'] = 36207.66*(1/total_lum)
            self.data.loc[self.data['RunYear'].isin([2017]), 'lumi_scale'] = 44307.4*(1/total_lum)
            self.data.loc[self.data['RunYear'].isin([2018]), 'lumi_scale'] = 58450.1*(1/total_lum)

            for i in range(4):
                self.data.loc[self.data[f"lep_ID_{i}"]==0,f"lep_SF_{i}"] = 1
                self.data.loc[~self.data[f"lep_ID_{i}"].isin([13,-13]), f"lep_SF_{i}"] = self.data[f"lep_SF_CombinedLoose_{i}"]
                self.data.loc[self.data[f"lep_ID_{i}"].isin([13,-13]), f"lep_SF_{i}"] = self.data[f"lep_SF_Mu_TTVA_AT_{i}"]*self.data[f"lep_SF_Mu_ID_Loose_AT_{i}"]
            
            
            self.data['weight'] = self.data['lumi_scale']*self.data['custTrigSF_LooseID_FCLooseIso_DLT']*self.data['weight_pileup']*self.data['jvtSF_customOR']*self.data['bTagSF_weight_DL1r_Continuous']*self.data['weight_mc']*self.data['xs']*self.data['lep_SF_0']*self.data['lep_SF_1']*self.data['lep_SF_2']*self.data['lep_SF_3']/self.data['totalEventsWeighted']

            
            
        else:
            
            self.data.loc[self.data['RunYear'].isin([2015,2016]), 'lumi_scale'] = 36207.66*(1/total_lum)
            self.data.loc[self.data['RunYear'].isin([2017]), 'lumi_scale'] = 44307.4*(1/total_lum)
            self.data.loc[self.data['RunYear'].isin([2018]), 'lumi_scale'] = 58450.1*(1/total_lum)

            self.data['weight'] = self.data['lumi_scale']*self.data['custTrigSF_TightElMediumMuID_FCLooseIso_DLT']*self.data['weight_pileup']*self.data['jvtSF_customOR']*self.data['bTagSF_weight_DL1r_77']*self.data['weight_mc']*self.data['xs']*self.data['lep_SF_CombinedLoose_0']*self.data['lep_SF_CombinedLoose_1']*self.data['lep_SF_CombinedLoose_2']*self.data['lep_SF_CombinedLoose_3']/self.data['totalEventsWeighted']

            
            
        #Need to multiply by total_lum
        self.data['weight'] = self.data['weight']*total_lum

    
    def add_additional_weight(self):
        
        ###
        # VV Njet 
        ###
        VV_njet_DSIDS = ['364250', '363489', '345705', '345706', '345715', '345718', '345723',
                        '364253','364254', '364255',
                         '364283', '364284','364285','364286','364287',
                         '363355','363356','363357','363358','363359','363360'
                        ]
        njet_corrections = {}
        factors = [0.965492, 0.835492, 0.769111, 0.725872, 0.694378, 0.669898]
        for i, col in enumerate(self.data.columns):
            print(i,col)
        for ind, i in enumerate(range(1,7)):
            print(f"Updating VV Njet {i} with factor: {factors[ind]}")

            self.data.loc[(self.data['mcChannelNumber'].isin(VV_njet_DSIDS))&self.data['nJets_OR']==i, 'weight'] = factors[ind] * self.data.loc[(self.data['mcChannelNumber'].isin(VV_njet_DSIDS))&self.data['nJets_OR']==i, 'weight']

        #>= 7:
        self.data.loc[(self.data['mcChannelNumber'].isin(VV_njet_DSIDS))&self.data['nJets_OR']>=7, 'weight'] = 0.650040 * self.data.loc[(self.data['mcChannelNumber'].isin(VV_njet_DSIDS))&self.data['nJets_OR']>= 7, 'weight']
        
        
        ###
        # ggVV
        ###
        ggVV_DSIDs = ['345705', '345706', '345715', '345718', '345723']
        
        self.data.loc[(self.data['mcChannelNumber'].isin(ggVV_DSIDs)), 'weight'] = 1.7 * self.data.loc[(self.data['mcChannelNumber'].isin(ggVV_DSIDs)), 'weight']
        
        
        ...
    
    
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
            if before != len(self.data):
                print(f"Removed duplicates: {before} -> {len(self.data)}")
            
         
    
    def scale_variables(self, scalers=None):
        
        self.scalers = {}
        if scalers:
            cols = list(scalers.keys())
            cols += ['weight']
            print("Using prefitted scalers.")
        else:
            cols = self.data.columns
            
        for col in cols:
            
            if col == 'weight':
                #Adjust the weights to be centered on 1 #normalise
                

                if self.config.get('clip_weight', False):
                    
                    self.data.loc[:,'scaled_weight'] =  (self.data.loc[:,col]/self.data[col].sum()).clip(lower=-0.5, upper=0.5)
                    continue

                if self.config.get('norm_weight_to_one', False):
                    self.data.loc[:,'scaled_weight'] = 1+((self.data.loc[:,col] - self.data.loc[:,col].mean())/(max(self.data.loc[:,col])-min(self.data.loc[:,col])))
                    continue

                elif self.config.get('shift_weight_to_one', False):
                    self.data.loc[:,'scaled_weight'] = 1 + ((self.data.loc[:,col] - self.data.loc[:,col].mean()))
                    continue
                elif self.config.get('standard_scale', False):
                    self.data.loc[:,'scaled_weight'] = 1 + ((self.data.loc[:,col] - self.data.loc[:,col].mean())/self.data.loc[:,col].std())
                    continue

                else: 
                    self.data.loc[:,'scaled_weight'] = self.data.loc[:,col]/self.data[col].sum()
                    continue

            if col in ['sample','eventNumber', 'index']:
                continue
                
            scaler = StandardScaler() if scalers is None else scalers[col]
            if scalers is None:
                self.data.loc[:, col] = scaler.fit_transform(np.array(self.data[col]).reshape(len(self.data[col]),1))
            else:
                self.data.loc[:, col] = scaler.transform(np.array(self.data[col]).reshape(len(self.data[col]),1))
            print(f"Scaled {col} to mean: {self.data[col].mean()}")
            self.scalers[col] = scaler
        
        self.save_scalers(self.output_dir)
            
    
    def save_scalers(self, output_dir):
        
        #Save the scalers used for each column
        scaler_folder = os.path.join(output_dir, 'scalers')
        if not os.path.exists(scaler_folder):
            os.makedirs(scaler_folder)
    
        for col, sc in self.scalers.items():
            pickle.dump(sc, open(os.path.join(scaler_folder,col+'_scaler.pkl'),'wb'))
            
            
    def split_dataset(self, use_val=False, use_eventnumber=False):
        
        val = []
        if use_eventnumber:
            print("Splitting dataset by event number.")
            if self.config['even_or_odd'] == 'Even':
                print("Using even.")
                train = self.data.loc[self.data['eventNumber'] % 2 == 0]
                test = self.data.loc[self.data['eventNumber'] % 2 == 1]
            else:
                print("Using odd.")
                train = self.data.loc[self.data['eventNumber'] % 2 == 1]
                test = self.data.loc[self.data['eventNumber'] % 2 == 0]
            
            #Sample 20% for the validation
            if use_val:
                test, val = train_test_split(test, test_size=self.config['validation_fraction'])
                
            print(f"Train size: {len(train)}, Val size: {len(val)}, Test size: {len(test)}")
            return train, val, test
        
        #--------------------------------------------------
        else:
            #TODO: Remove or alter this
            
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

                if len(sample_data ) * fraction < 1:
                    continue

                if len(sample_data) == 1:
                    self.test_data = pd.concat([self.test_data, sample_data])
                    self.data.drop(index=sample_data.index.values, axis=0, inplace=True)
                    print(f"Found sample: {sample} with 1 example.")
                    continue

                train, test = train_test_split(sample_data, test_size=fraction)

                self.test_data = pd.concat([self.test_data, test])
                self.data.drop(index=test.index.values, axis=0, inplace=True)

                #TODO:
                # - add this into verbose mode
                print(f"Split {sample} by removing {len(test)} events.")

            print(f"Total removed: {len(self.test_data)}, Remaining length: {len(self.data)}")
            return self.data, self.test_data