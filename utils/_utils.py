import yaml
import datetime as dt
import os


#Function to load a config from a string
def load_yaml_config(infile):
    
    if type(infile) == str:
        with open(infile, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = infile
    return config


def get_reversed_groupings(grouping_dict):
    
    reversed_groupings = {}
    for key, values in grouping_dict.items():
        for val in values:
            reversed_groupings[val] = key
    return reversed_groupings
    

def make_output_folder(config, root_loc='results'):
    
    
    date = dt.datetime.strftime(dt.datetime.now(),"%H%M-%d-%m-%Y")
    output_dir = os.path.join(root_loc, config['out_folder'], f'Run_{date}')
        
    if config['test_dump']:
        print("[INFO]   Outputting to test_dump...")
        output_dir = os.path.join(root_loc, 'test_dump')
            
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    return output_dir


def find_root_files(root, directory, master_list=[]):
        files = os.listdir(os.path.join(root,directory))
        for f in files:
            if '.root' in f:
                master_list.append(os.path.join(os.path.join(root,directory),f))
                continue
            else:
                master_list = find_root_files(os.path.join(root,directory),f, master_list)
        return master_list