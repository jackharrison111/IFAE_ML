
#File to evaluate a model over NTUPLES


from model.model_getter import get_model
from preprocessing.create_variables import VariableMaker
from feather.make_feather import FeatherMaker

from model.model_getter import get_model
from utils._utils import load_yaml_config

import yaml
import uproot
import torch
import os
import pickle
import numpy as np
import math
import pandas as pd
import json
import argparse


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
    
    
    parser = argparse.ArgumentParser("Running VAE evaluation")
    
    parser.add_argument("-i","--inputConfig",action="store", help="Set the input config to use, default is 'configs/training_config.yaml'", default="configs/training_config.yaml", required=False)
    parser.add_argument("-f","--featherConfig",action="store", help="Set the feather config to use, default is 'feather/feather_config.yaml'", default="feather/feather_config.yaml", required=False)
    parser.add_argument("-e","--evenModelPath",action="store", help="Set the trained even model to use", default="outputs/EVEN_FINAL_VAE_1318-23-09-2022", required=False)
    parser.add_argument("-o","--oddModelPath",action="store", help="Set the trained odd model to use", default="outputs/ODD_FINAL_VAE_1414-23-09-2022", required=False)
    parser.add_argument("-r","--Region",action="store", help="Set the region to config use, default is to use the config", 
                        default=None, required=False)
    parser.add_argument("-first","--First",action="store", help="Set the first file to run over", 
                        default=-1, required=False, type=int)
    parser.add_argument("-last","--Last",action="store", help="Set the last file to run over", 
                        default=-1, required=False, type=int)
    args = parser.parse_args()
    
    #EVEN means TRAINED on even, and so we want even model to evaluate on ODD data
    even_load_dir = args.evenModelPath
    odd_load_dir = args.oddModelPath
    featherconfig = args.featherConfig
    region = args.Region
    first = args.First
    last = args.Last
    
    #even_load_dir = 'outputs/good_runs/LongRun_VAE_1Z_0b_2SFOS_10GeV/Run_1507-22-12-2022_B/'
    #even_load_dir = 'outputs/good_runs/LongRun_VAE_1Z_0b_2SFOS_25GeV/Run_1402-17-01-2023'
    #odd_load_dir = even_load_dir
    
    
    #region = '1Z_1b_2SFOS'
    #even_load_dir = 'results/1Z_1b_2SFOS_FirstRun/Run_1252-24-02-2023'
    #odd_load_dir = 'results/1Z_1b_2SFOS_OddRun/Run_1609-24-02-2023'
    
    featherconfig = os.path.join('configs/feather_configs/10GeV',f"{region}.yaml")
    
    feather_conf = load_yaml_config(featherconfig)
    cut_expr = feather_conf[feather_conf['cut_choice']]
    
    
    
    train_conf = args.inputConfig
    train_conf = os.path.join(even_load_dir, 'config.yaml')
    
    #Read from args
    train_conf = load_yaml_config(train_conf)
    
    
    
    #SWITCH THE WEIGHTS AROUND
    even_model = get_model(conf=train_conf)
    even_model.load_state_dict(torch.load(os.path.join(odd_load_dir,'model_state_dict.pt')))
    
    odd_model = get_model(conf=train_conf)
    odd_model.load_state_dict(torch.load(os.path.join(even_load_dir,'model_state_dict.pt')))
    
    
    all_root_files = find_root_files(train_conf['ntuple_path'], '', [])
    train_vars = [v for v in train_conf['training_variables'] if v not in ['eventNumber', 'weight', 'sample']]
    
    
    #Get the required variables from tree
    fm = FeatherMaker(master_config=feather_conf)
    variables = fm.get_features()
    
    #Get the used DSIDS:
    with open(os.path.join(f"configs/sample_jsons/{fm.master_config['region']}/{fm.master_config['json_output']}")) as f:
        sample_map = json.load(f)
    
    all_dsids = []
    #for sample in config['chosen_samples']:
    #    all_dsids += sample_map[f'XXX_{sample}_samples']
        
    
    for i, file in enumerate(all_root_files):
        
        if i < first and first!=-1:
            continue
        if i > last and last!=-1:
            break
        
        
        print(f"Running file {file}. {i} / {len(all_root_files)}")
        
        save_path = file.split(os.path.basename(train_conf['ntuple_path']))[1]
        
        if save_path[0] == '/':
            save_path = save_path[1:]
        
        outfile = os.path.join(train_conf['ntuple_outdir'],save_path)
        
        print(f"Saving file to: {outfile}")
        if not os.path.exists(os.path.split(outfile)[0]):
            os.makedirs(os.path.split(outfile)[0])
        
        
        
        #CHECK IF DSID IS IN THE TRAINING SAMPLES
        dsid = os.path.split(save_path)[1]
       
        
        out_data = []
        with uproot.open(file+':nominal') as f:
            vars = list(f.keys())
            
        out_var_name = f"{train_conf['Region_name']}_{train_conf['model_type']}"
            
        #Instead of opening the whole file, find a way to chunk it and update the ntuples? 
        for all_data in uproot.iterate(file+':nominal', cut=cut_expr, library='np', allow_missing=True,step_size='100 MB'):
            
            all_data = pd.DataFrame(all_data, columns=vars)
            
            if len(all_data) == 0:
                continue
            
            #Make the variables required
            vm = VariableMaker()
            
            #Use a config for this
            func_strings = fm.master_config[fm.master_config['variable_functions_choice']]
            funcs = [getattr(vm, s) for s in func_strings]
            
            for i, f in enumerate(funcs):
                all_data = f(all_data)
                print(f"Done function {i}.")
                
            del vm
            
            if len(all_data) == 0:
                continue
                
            #CHECK
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
                for col in even_evts.columns:
                    scaler = pickle.load(open(os.path.join(even_load_dir,f'scalers/{col}_scaler.pkl'),'rb'))
                    even_evts[col] = scaler.transform(np.array(even_evts[col]).reshape(-1,1))
                    even_data = torch.Tensor(even_evts.values)
                    
            del even_evts, odd_evts, scaler
            print(f"Even data: {len(even_data)}, Odd data: {len(odd_data)}")
            
            with torch.no_grad():
                
                #Even:
                if len(even_data) != 0:
                    out, mu, logVar = even_model(even_data)
                    even_loss, mse, kld = even_model.loss_function(out, even_data, mu, logVar)
                    even_logloss = torch.log(even_loss)
                    all_data.loc[(all_data['eventNumber'] % 2==0), out_var_name] = even_logloss.numpy()
                
                #Odd:
                if len(odd_data) != 0:
                    out, mu, logVar = odd_model(odd_data)
                    loss, mse, kld = odd_model.loss_function(out, odd_data, mu, logVar)
                    logloss = torch.log(loss)
                    all_data.loc[(all_data['eventNumber'] % 2==1), out_var_name] =  logloss.numpy()
                
                if len(out_data) == 0:
                    out_data = all_data
                else:
                    out_data = pd.concat([out_data,all_data])
                    
                print(f"Output data: {len(out_data)}")
                del all_data, even_data, odd_data
            
        if len(out_data) == 0:
            with uproot.recreate(outfile) as f:
                #NEED TO MAKE THIS AN EMPTY TTREE
                f.mkdir('nominal')
                print(f"Saved file to {outfile}.")
        else:
            with uproot.recreate(outfile) as f:
                f['nominal'] = out_data
                print(f"Saved file to {outfile}.")
            
    
