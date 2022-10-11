from model.autoencoder import VAE, AE
import yaml

def get_model(conf="configs/training_config.yaml"):
    
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
        
    return model