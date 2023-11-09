import yaml
from train import Trainer
from test import Tester
import pickle
from time import perf_counter
from preprocessing.dataset_io import DatasetHandler
import os 
from preprocessing.dataset import data_set
from torch.utils.data import DataLoader
import argparse
import torch
import numpy as np

# Example of how to run just an evaluation...
# Make sure to have uncommented 
#python run/run_all.py -n 1 -r 0Z_0b_0SFOS -j NF_FixedIndex_Even -t 1 -e Even



if __name__ == '__main__':
    
    start = perf_counter()
    
    parser = argparse.ArgumentParser("Running standalone analysis package")
    parser.add_argument("-i","--inputConfig",action="store", help="Set the input config to use, default is 'configs/training_config.yaml'", 
                        default="configs/training_config.yaml", required=False)
    parser.add_argument("-r","--Region",action="store", help="Set the region to config use, default is to use the config", 
                        default=None, required=False, type=str)
    parser.add_argument("-n","--NormFlows",action="store", help="Set whether to use normalising flows config, use in conjuction with region", default=False, required=False, type=bool)
    parser.add_argument("-j","--JobName",action="store", help="Set the job name, for outputting to a given folder", 
                        default=None, required=False, type=str)
    parser.add_argument("-t","--NoTrain",action="store", help="Choose whether to train the model or not", 
                        default=False, required=False, type=str)
    parser.add_argument("-e","--EvenOrOdd",action="store", help="Choose whether to train the model on Even or Odd event numbers", 
                        default=False, required=False, type=str)
    
    parser.add_argument("-s","--nEpochs",action="store", help="Set number of epochs to run over", default=False, required=False, type=bool)
    
    args = parser.parse_args()
    conf = args.inputConfig
    
    if args.Region:
        if args.NormFlows:
            conf = f"configs/training_configs/Regions/{args.Region}/nf_config.yaml"
        else:
            conf = f"configs/training_configs/Regions/{args.Region}/training_config.yaml"
    
    
    # Make model
    from model.model_getter import get_model
    model = get_model(conf)
    print(model)
    
    
    #Get the dataset
    dh = DatasetHandler(conf, job_name=args.JobName)
    
    #Use flag here to set even or odd
    if args.EvenOrOdd == 'Even':
        dh.config['even_or_odd'] = 'Even'
    elif args.EvenOrOdd == 'Odd':
        dh.config['even_or_odd'] = 'Odd'
        
    if args.nEpochs:
        dh.config['num_epochs'] = args.nEpochs
    
    train, val, test = dh.split_dataset(use_val=dh.config['validation_set'], 
                use_eventnumber=dh.config.get('use_eventnumber',None))
    
    
    sum_train_w = train['weight'].sum()
    sum_val_w = val['weight'].sum()
    sum_test_w = test['weight'].sum()
    
    out_str = [f"Total sum of weights: {sum_train_w + sum_val_w + sum_test_w}\n"]
    out_str.append(f"Total sum of train: {sum_train_w}\n")
    out_str.append(f"Total sum of val: {sum_val_w}\n")
    out_str.append(f"Total sum of test: {sum_test_w}\n")
    
    with open(os.path.join(dh.output_dir, 'TrainWeightSummary.txt'),'w') as f:
        f.writelines(out_str)
    
    
    train_data = data_set(train)
    test_data = data_set(test)
    train_loader = DataLoader(train_data, batch_size=dh.config['batch_size'], shuffle=True)    
    test_loader = DataLoader(test_data, batch_size=2048)
    
    if len(val) != 0:
        val_data = data_set(val)
        val_loader = DataLoader(val_data, batch_size=2048)
    else:
        val_loader=None
    
    
    #Train the model
    t = Trainer(model, config=dh.config, output_dir=dh.output_dir)
    
    if not args.NoTrain:
        model = t.train(train_loader, val_loader=val_loader)
        
    else:
        
        #Try loading model from somewhere...
        with open('evaluate/region_settings/nf_NewYields.yaml', 'r') as f:
            run_dirs = yaml.safe_load(f)
            
        if t.config['even_or_odd'] =='Even':
            #Load the model using the 
            path_choice = 'even_path'
            base_dir = 'even_base_dir'
        else:
            #Load the model using the 
            path_choice = 'odd_path'
            base_dir = 'odd_base_dir'
            
        dict_path = os.path.join(run_dirs[base_dir], run_dirs['regions'][args.Region][path_choice])
        model.load_state_dict(torch.load(os.path.join(dict_path,'model_state_dict.pt')))
    
    
    #Test the model
    tester = Tester(config=t.config)
    tester.out_dir = t.output_dir
    output = tester.evaluate(model, test_loader)
    
    #Evaluate over val
    val_output = tester.evaluate(model, val_loader)
    
    #------------------------------------------------------------------------
    # Format the output
    #------------------------------------------------------------------------

    import shutil
    shutil.copy(conf, os.path.join(t.output_dir, 'config.yaml'))
        
    model_info =f"""
    Model architecture:
    {model}
    
    Optimizer:
    {t.optimizer}
    """

    with open(os.path.join(t.output_dir, 'model_description.txt'),'w') as f:
        f.writelines(model_info)
        

    #TODO: SAVE TO COMMON AREA INSTEAD
    dataset_savedir = '/data/at3/common/multilepton/VLL_production/trainings'
    dataset_jobdir = os.path.join(dataset_savedir, args.JobName)

    #with open(os.path.join(t.output_dir,'New_saved_outputs_2.pkl'), 'wb') as f:
    #    pickle.dump(output, f)
    #with open(os.path.join(t.output_dir,'New_saved_val_outputs_2.pkl'), 'wb') as f:
    #    pickle.dump(val_output, f)

    if not os.path.exists(dataset_jobdir):
        os.makedirs(dataset_jobdir)
        
    with open(os.path.join(dataset_jobdir,'saved_outputs.pkl'), 'wb') as f:
        pickle.dump(output, f)
    with open(os.path.join(dataset_jobdir,'saved_val_outputs.pkl'), 'wb') as f:
        pickle.dump(val_output, f)
        
        
    if t.config['model_type'] == 'NF':
        
        min_test = min(output['ad_score'])
        max_test = max(output['ad_score'])
        
        min_val = min(val_output['ad_score'])
        max_val = max(val_output['ad_score'])
        
        prob_min = min(min_test,min_val)
        prob_max = max(max_test,max_val)
        
        #Also add the percentiles of these...
        all_scores = np.append(output['ad_score'], val_output['ad_score'])
        percentile1 = np.percentile(all_scores, 1)
        percentile99 = np.percentile(all_scores,99)
        
        percentile05 = np.percentile(all_scores, 0.5)
        percentile995 = np.percentile(all_scores,99.5)
        
        percentile01 = np.percentile(all_scores, 0.1)
        percentile999 = np.percentile(all_scores, 99.9)
        
        
        #Before using 99.9 => 0.1 try using 99.99 => 0.01
        percentile001 = np.percentile(all_scores, 0.01)
        proper_max_all = max(all_scores)
        
        
        #Get the minimum of the val and test output's 
        scale_strs = []
        scale_strs.append("Scaled NF scores using: \n")
        scale_strs.append(f"Test min_prob: {min_test}\n")
        scale_strs.append(f"Test max_prob: {max_test}\n")
        scale_strs.append(f"Val min_prob: {min_val}\n")
        scale_strs.append(f"Val max_prob: {max_val}\n")
        scale_strs.append(f"Use for scaling: {prob_min} , {prob_max}\n")
        scale_strs.append(f"Percentiles1_99: {percentile1} , {percentile99}\n")
        scale_strs.append(f"Percentiles05_995: {percentile05} , {percentile995}\n")
        scale_strs.append(f"Percentiles01_999: {percentile01} , {percentile999}\n")
        scale_strs.append(f"Best scaling: {-proper_max_all} , {-percentile001}\n")
        with open(os.path.join(t.output_dir,'scalers','NF_likelihood_scaling.txt'),'w') as f:
            f.writelines(scale_strs)
    
    #try:
    print(f"Loading signal models...")
    sig_dh = DatasetHandler(conf, scalers=dh.scalers, out_dir=t.output_dir)
    sig_data = data_set(sig_dh.data)
    sig_loader = DataLoader(sig_data, batch_size=2048, shuffle=True)
    sig_output = tester.evaluate(t.model, sig_loader)

    #TODO: SAVE TO COMMON
    with open(os.path.join(dataset_jobdir,'saved_signal_outputs.pkl'), 'wb') as f:
        pickle.dump(sig_output, f)

   
    out_plots = t.config.get('output_plots' , None)
    
    #out_plots = False
    if not out_plots:
        print("No output plots specified for evaluation run. Ending script...")
    else:
        tester.analyse_results(output, sig_output=sig_output, **out_plots)
    
    
    end = perf_counter()
    print(f"Time taken for everything: {end-start}s.")
    