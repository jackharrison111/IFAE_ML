import yaml
from run.train import Trainer
from run.test import Tester
import pickle
from time import perf_counter
from preprocessing.dataset_io import DatasetHandler
import os 
from preprocessing.dataset import data_set
from torch.utils.data import DataLoader


if __name__ == '__main__':
    
    #Get the model
    #Train on half the dataset
    #Save the scalers used and the model
    
    
    start = perf_counter()
    
    # Make model
    
    from model.model_getter import get_model
    
    conf="configs/training_config.yaml"
    
    model = get_model(conf=conf)
    
    
    
    # Run training on half dataset
    #Train
    t = Trainer(model, config=conf)
    train, test = t.get_dataset()
    print(f"Length of train: {len(train)}")
    
    epoch_loss, val_loss = t.train_vae(train)
    
    scaler_folder = os.path.join(t.output_dir, 'scalers')
    if not os.path.exists(scaler_folder):
        os.makedirs(scaler_folder)
    import pickle
    for col, sc in t.dh.scalers.items():
        pickle.dump(sc, open(os.path.join(scaler_folder,col+'_scaler.pkl'),'wb'))
    
    
    # Run testing on other half
    tester = Tester(config=conf)
    tester.out_dir = t.output_dir
    output = tester.evaluate_vae(t.model, test)
    
    with open(os.path.join(t.output_dir,'saved_outputs.pkl'), 'wb') as f:
        pickle.dump(output, f)

    
    vll_conf= 'configs/VLL_VAE_config.yaml'
    data_conf = 'configs/data_config.yaml'
    
    sig_dh  =  DatasetHandler(vll_conf, scalers=t.dh.scalers)
    sig_data = data_set(sig_dh.data)
    sig_loader = DataLoader(sig_data, batch_size=1, shuffle=True)
    sig_output = tester.evaluate_vae(t.model, sig_loader)
    
    with open(os.path.join(t.output_dir,'sig_outputs.pkl'), 'wb') as f:
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

    
