
#Get model

#Load weights

#Get all the ntuples to run over

#evaluate model over them


from model.model_getter import get_model
from preprocessing.create_variables import VariableMaker
from feather.make_feather import FeatherMaker

import yaml
import uproot
import torch
import os
import pickle
import numpy as np
import math
import pandas as pd
import json

def find_root_files(root, directory, master_list=[]):
        files = os.listdir(os.path.join(root,directory))
        for f in files:
            if '.root' in f:
                master_list.append(os.path.join(os.path.join(root,directory),f))
                continue
            else:
                master_list = find_root_files(os.path.join(root,directory),f, master_list)
        return master_list

if __name__ == '__main__':
    
    from model.model_getter import get_model
    
    feather_conf = "feather/feather_config.yaml"
    with open(feather_conf, 'r') as f:
        feather_config = yaml.safe_load(f)
    cut_expr = feather_config[feather_config['cut_choice']]
    
    train_conf="configs/training_config.yaml"
    #VLL conf
    conf = "configs/VLL_signal_config.yaml"
    conf = "configs/data_config.yaml"
    #conf = train_conf
    with open(conf,'r') as f:
        config = yaml.safe_load(f)
    
    #EVEN means TRAINED on even, and so we want even model to evaluate on ODD data
    even_load_dir = "outputs/EVEN_FINAL_VAE_1318-23-09-2022"
    odd_load_dir = "outputs/ODD_FINAL_VAE_1414-23-09-2022"
    
    #SWITCH THE WEIGHTS AROUND
    even_model = get_model(conf=train_conf)
    even_model.load_state_dict(torch.load(os.path.join(odd_load_dir,'model_state_dict.pt')))
    
    odd_model = get_model(conf=train_conf)
    odd_model.load_state_dict(torch.load(os.path.join(even_load_dir,'model_state_dict.pt')))
    
    
    all_root_files = find_root_files(config['base_dir'], '', [])
    
    train_vars = [v for v in config['training_variables'] if v not in ['eventNumber', 'weight', 'sample']]
    
    fm = FeatherMaker()
    #Get the required variables from tree
    variables = fm.get_features()
    
    #Get the used DSIDS:
    with open('configs/Bkg_samples.json') as f:
        sample_map = json.load(f)
    
    all_dsids = []
    #for sample in config['chosen_samples']:
    #    all_dsids += sample_map[f'XXX_{sample}_samples']
        
    
    for i, file in enumerate(all_root_files):
        
        print(f"Running file {file}. {i} / {len(all_root_files)}")
        
        save_path = file.split('newvars/data/')[1]
        outfile = os.path.join(config['out_dir'],save_path)
        if not os.path.exists(os.path.split(outfile)[0]):
            os.makedirs(os.path.split(outfile)[0])
        
        #CHECK IF DSID IS IN THE TRAINING SAMPLES
        dsid = os.path.split(save_path)[1]
       
    
        #Instead of opening the whole file, find a way to chunk it and update the ntuples? 
        
        with uproot.open(file + ':nominal') as evts:
                 
            if len(evts) == 0:
                with uproot.recreate(outfile) as f:
                    ...
                continue
                
            all_data = evts.arrays(cut=cut_expr, library='pd')
            #all_data = evts.arrays(library='pd')
            
            #all_evts = all_data.copy()
            
            #Make the variables required
            vm = VariableMaker()
            funcs = [vm.find_bestZll_pair, vm.calc_4lep_mZll, vm.calc_4lep_pTll, vm.calc_m3l]
            for i, f in enumerate(funcs):
                all_data = f(all_data)
                print(f"Done function {i}.")
                
            all_data.drop(columns=['best_Zllpair', 'other_Zllpair'], inplace=True)
            
            '''
            if os.path.splitext(dsid)[0] not in all_dsids:
                all_data['VAE_score_1Z0b2SFOS'] = -99
                with uproot.recreate(outfile) as f:
                    f['nominal'] = all_data
                    print(f"Saved default file to {outfile}.")
                    continue
            '''
            
            odd_evts = all_data.loc[all_data['eventNumber'] % 2 == 1]
            even_evts = all_data.loc[all_data['eventNumber'] % 2 == 0]
            print(f"Even events: {len(even_evts)}, Odd events: {len(odd_evts)}")
            
            even_data = []
            odd_data = []
            
            #Scale data for each column using presaved scalers
            if len(odd_evts) != 0:
                odd_evts = odd_evts[train_vars]
                for col in odd_evts.columns:
                    scaler = pickle.load(open(os.path.join(odd_load_dir,f'scalers/{col}_scaler.pkl'),'rb'))
                    odd_evts[col] = scaler.transform(np.array(odd_evts[col]).reshape(-1,1))
                    odd_data = torch.Tensor(odd_evts.values)
            
            if len(even_evts) != 0:
                even_evts = even_evts[train_vars]
                print(len(even_evts))
                for col in even_evts.columns:
                    scaler = pickle.load(open(os.path.join(even_load_dir,f'scalers/{col}_scaler.pkl'),'rb'))
                    even_evts[col] = scaler.transform(np.array(even_evts[col]).reshape(-1,1))
                    even_data = torch.Tensor(even_evts.values)
                    
            print(f"Even data: {len(even_data)}, Odd data: {len(odd_data)}")
            with torch.no_grad():
                
                #Even:
                if len(even_data) != 0:
                    out, mu, logVar = even_model(even_data)
                    even_loss, mse, kld = even_model.loss_function(out, even_data, mu, logVar)
                    even_logloss = torch.log(even_loss)
                    all_data.loc[(all_data['eventNumber'] % 2==0), 'VAE_score_1Z0b2SFOS'] = even_logloss.numpy()
                
                #Odd:
                if len(odd_data) != 0:
                    out, mu, logVar = odd_model(odd_data)
                    loss, mse, kld = odd_model.loss_function(out, odd_data, mu, logVar)
                    logloss = torch.log(loss)
                    all_data.loc[(all_data['eventNumber'] % 2==1),'VAE_score_1Z0b2SFOS'] =  logloss.numpy()
                
            
        with uproot.recreate(outfile) as f:
            f['nominal'] = all_data
            print(f"Saved file to {outfile}.")
            
    
