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



if __name__ == '__main__':
    
    start = perf_counter()
    
    parser = argparse.ArgumentParser("Running standalone analysis package")
    parser.add_argument("-i","--inputConfig",action="store", help="Set the input config to use, default is 'configs/training_config.yaml'", 
                        default="configs/training_config.yaml", required=False)
    parser.add_argument("-r","--Region",action="store", help="Set the region to config use, default is to use the config", 
                        default=None, required=False, type=str)
    
    args = parser.parse_args()
    conf = args.inputConfig
    
    if args.Region:
        conf = f"configs/training_configs/Regions/{args.Region}/training_config.yaml"
    
    
    # Make model
    from model.model_getter import get_model
    model = get_model(conf)
    
   
    print(model)
    
    #Get the dataset
    #Make input variable plots ? 
    #Get signal and bkg? 
    dh = DatasetHandler(conf)
    train, val, test = dh.split_dataset(use_val=dh.config['validation_set'], 
                use_eventnumber=dh.config.get('use_eventnumber',None))
    
    
    
    #TODO: Add plotting of input variables here?
    
    
    
    train_data = data_set(train)
    test_data = data_set(test)
    train_loader = DataLoader(train_data, batch_size=dh.config['batch_size'], shuffle=True)    
    test_loader = DataLoader(test_data, batch_size=1)
    
    if len(val) != 0:
        val_data = data_set(val)
        val_loader = DataLoader(val_data, batch_size=1)
    else:
        val_loader=None
    
    
    
    
    #Train the model
    t = Trainer(model, config=conf, output_dir=dh.output_dir)
    epoch_loss,epoch_logloss, val_loss = t.train_model(train_loader, val_loader=val_loader)
    
    
    
    #Test the model
    tester = Tester(config=conf)
    tester.out_dir = t.output_dir
    output = tester.evaluate_vae(t.model, test_loader)
    
    
    

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
        
    with open(os.path.join(t.output_dir,'saved_outputs.pkl'), 'wb') as f:
        pickle.dump(output, f)
        
    
    #try:
    print(f"Loading signal models...")
    sig_dh = DatasetHandler(conf, scalers=dh.scalers)
    sig_data = data_set(sig_dh.data)
    sig_loader = DataLoader(sig_data, batch_size=1, shuffle=True)
    sig_output = tester.evaluate_vae(t.model, sig_loader)

    with open(os.path.join(t.output_dir,'saved_signal_outputs.pkl'), 'wb') as f:
        pickle.dump(sig_output, f)

    #except:
    #    print("Couldn't run over signal data... likely 0 signal events in region.")
    #    sig_output=None
        
    
    outplots  = {
        'loss_hist' : True,
        'logloss_hist' : True,
        'logloss_sample_hist' : True,
        'logloss_BkgvsSig_hist' : True,
        'cdf_hist': True,
        'chi2_plots' : True,
        'nsig_vs_nbkg' : True,
        'sig_vs_nbkg' : False,
        'trexfitter_plot':True
    }
    
    tester.analyse_results(output, sig_output=sig_output, **outplots)
    
    
    # Add back into ntuples
    # Needs to be able to do even and odd at the same time? 
    
    from feather.add_to_ntuple import NtupleWriter
    nw = NtupleWriter()
    
    if t.config.get('add_to_ntuple',None):
        
        print("[INFO]   Adding outputs to ntuples")
        if t.config['even_or_odd'] == 'Even':
            nw.process_output(sig_output, t.config.get('base_dir', None), t.config.get('out_dir', None))
            #nw.update_ntuples(sig_output, t.config.get('out_dir'))
            #nw.process_output(sig_output, )
        elif t.config['even_or_odd'] == 'Odd':
            nw.update_ntuples(output, t.config.get('out_dir'))
            nw.process_output(sig_output, t.config.get('vll_base_dir'), t.config.get('vll_out_dir'))
    
    
    end = perf_counter()
    print(f"Time taken for everything: {end-start}s.")
    