from model.autoencoder import VAE, AE
import yaml
from utils._utils import load_yaml_config

def get_model(conf="configs/training_config.yaml"):
    
    
    conf = load_yaml_config(conf)
    
    useful_columns = [col for col in conf['training_variables'] if col not in ['sample','weight', 'scaled_weight', 'eventNumber']]
    inp_dim = len(useful_columns)
    
    
    enc_dim = [inp_dim]+conf['enc_layers']
    dec_dim = conf['dec_layer']+[inp_dim]
    z_dim = conf['z_dim']
    
    model_type = conf['model_type']
    if model_type == 'AE':
        model = AE(enc_dim, dec_dim, z_dim)
    elif model_type == 'VAE':
        model = VAE(enc_dim, dec_dim, z_dim)
        
    return model