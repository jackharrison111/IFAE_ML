import yaml
from train import Trainer
from test import Tester
import pickle
from time import perf_counter
from preprocessing.dataset_io import DatasetHandler
import os 
from preprocessing.dataset import data_set
from torch.utils.data import DataLoader


if __name__ == '__main__':
    
    start = perf_counter()
    
    # Make model
    from model.autoencoder import VAE, AE
    
    conf = "configs/training_config.yaml"
    with open(conf,'r') as f:
        conf = yaml.safe_load(f)
    
    
    useful_columns = [col for col in conf['training_variables'] if col not in ['sample','weight', 'scaled_weight', 'eventNumber']]
    enc_dim = [len(useful_columns),8]
    dec_dim = [8,len(useful_columns)]
    z_dim = 4
    model_type = conf['model_type']
    if model_type == 'AE':
        model = AE(enc_dim, dec_dim, z_dim)
    elif model_type == 'VAE':
        model = VAE(enc_dim, dec_dim, z_dim)
    
    conf = "configs/training_config.yaml"
    
    #Split data based on eventnumber
    
    
    # Run training on half dataset
    #Train
    t = Trainer(model, config=conf)
    train, test = t.get_dataset()
    epoch_loss, val_loss = t.train_vae(train)
    
    print(f"Length of train: {len(train)}")
    
    # Run testing on other half
    tester = Tester(config=conf)
    tester.out_dir = t.output_dir
    output = tester.evaluate_vae(t.model, test)
    
    with open(os.path.join(t.output_dir,'saved_outputs.pkl'), 'wb') as f:
        pickle.dump(output, f)
        
    scaler_folder = os.path.join(t.output_dir, 'scalers')
    if not os.path.exists(scaler_folder):
        os.makedirs(scaler_folder)
    
    import pickle
    for col, sc in t.dh.scalers.items():
        pickle.dump(sc, open(os.path.join(scaler_folder,col+'_scaler.pkl'),'wb'))
    
    vll_conf= 'configs/VLL_VAE_config.yaml'
    data_conf = 'configs/data_config.yaml'
    
    sig_dh  =  DatasetHandler(data_conf, scalers=t.dh.scalers)
    sig_data = data_set(sig_dh.data)
    sig_loader = DataLoader(sig_data, batch_size=1, shuffle=True)
    sig_output = tester.evaluate_vae(t.model, sig_loader)
    
    
    with open(os.path.join(t.output_dir,'saved_signal_outputs.pkl'), 'wb') as f:
        pickle.dump(sig_output, f)
    
    outplots  = {
        'loss_hist' : True,
        'logloss_hist' : True,
        'logloss_sample_hist' : True,
        'logloss_BkgvsSig_hist' : True,
        'cdf_hist': True,
        'chi2_plots' : True, 
    }
    
    tester.analyse_results(output, sig_output=sig_output, **outplots)
    
    print("Actually here")
    
    # Add back into ntuples
    from feather.add_to_ntuple import NtupleWriter
    nw = NtupleWriter()
    
    if t.config.get('add_to_ntuple',None):
        
        if t.config['even_or_odd'] == 'Even':
            nw.process_output(sig_output, t.config.get('base_dir', None), t.config.get('out_dir', None))
            #nw.update_ntuples(sig_output, t.config.get('out_dir'))
            #nw.process_output(sig_output, )
        else:
            nw.update_ntuples(output, t.config.get('out_dir'))
            nw.process_output(sig_output, t.config.get('vll_base_dir'), t.config.get('vll_out_dir'))
    
    
    end = perf_counter()
    print(f"Time taken for everything: {end-start}s.")
    
    
