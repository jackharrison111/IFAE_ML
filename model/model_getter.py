from model.autoencoder import VAE, AE
from model.normalising_flow import NormFlow
import yaml
from utils._utils import load_yaml_config

def get_model(conf="configs/training_config.yaml"):
    
    
    conf = load_yaml_config(conf)
    
    useful_columns = [col for col in conf['training_variables'] if col not in ['sample','weight', 'scaled_weight', 'eventNumber', 'index']]
    inp_dim = len(useful_columns)
    
    
    enc_dim = [inp_dim]+conf.get('enc_layers',[])
    dec_dim = conf.get('dec_layer',[])+[inp_dim]
    z_dim = conf.get('z_dim', 0)
    
    model_type = conf['model_type']
    
    if model_type == 'AE':
        model = AE(enc_dim, dec_dim, z_dim)
        
    elif model_type == 'VAE':
        model = VAE(enc_dim, dec_dim, z_dim)
        
    elif model_type == 'NF':    
        use_spline = conf.get('use_spline', False)
        num_layers = conf.get('num_layers', 8)
        hidden_layers = conf.get('hidden_depth', 2)
        hidden_dim = conf.get('hidden_dim', 64)
        func_type = conf.get('func_type', 'Tanh')
        
        model = NormFlow(use_spline=use_spline, input_dims=inp_dim, num_layers=num_layers, hidden_layers=hidden_layers,
                        hidden_units=hidden_dim, func_type=func_type)
    return model