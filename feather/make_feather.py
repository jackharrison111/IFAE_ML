import os
import uproot
import yaml
import json
import pandas as pd
import numpy as np
from time import perf_counter
from utils._utils import load_yaml_config

'''
Remember to export the python path variable:
export PYTHONPATH=/nfs/pic.es/user/j/jharriso/IFAE_ML
'''

#Class needs to:
# - Read master config
# - Find the files to run over
# - Get the variables required
# - Loop over all files in each sample
# - Calculate the required extra variables - pass this as a function?
# - Reset it and save the feather file

class FeatherMaker():
    
    def __init__(self, master_config='feather/feather_config.yaml'):
        
        self.master_config = load_yaml_config(master_config)
        self.region = self.master_config['region']
    
            
    def get_features(self):

        infile = os.path.join('configs/variable_choices', self.region, self.master_config['variables_file'])
        with open(infile, 'r') as file:
            vars = file.readlines()
            vars = [v.rstrip() for v in vars if v[0] != '#']
        vars = list(set(vars))
        return vars


    '''
    Function to recursively find root files contained within a directory.
    Files must end in .root
    '''
    def find_root_files(self, root, directory, master_list=[]):
        files = os.listdir(os.path.join(root,directory))
        for f in files:
            if '.root' in f:
                master_list.append(os.path.join(os.path.join(root,directory),f))
                continue
            else:
                master_list = self.find_root_files(os.path.join(root,directory),f, master_list)

        return master_list
    
    
    def extract_DSIDs(self, sample_file, output_json):
    
        s_file = os.path.join('configs/sample_choices', self.region, sample_file)
        with open(s_file, 'r') as file:
            sample_txt = file.readlines()

        samples_dict = {}
        for line in sample_txt:
            all_samples = []
            sample_list = line.split(':')[-1]
            sample_name = line.split(':')[0]
            if sample_name[0]=='#':
                #sample_name = sample_name[1:]
                continue
            samples = sample_list.split(',')
            DSIDs = [path.split('/')[-1][:6] for path in samples]
            for id in DSIDs:
                if id == '':
                    continue
                all_samples.append(id)
            all_samples = list(set(all_samples))
            samples_dict[sample_name] = all_samples

        o_json = os.path.join('configs/sample_jsons', self.region, output_json)
        with open(o_json,'w') as f:
            json.dump(samples_dict, f)
        self.output_json = o_json
            
            
    def make_sample_file_map(self, all_files):
        
        #Read the mapping of sample : DSIDs
       
        with open(self.output_json, 'r') as f:
            sample_map = json.load(f)

        #Transform from sample : DSIDs to sample : file_path
        self.sample_file_paths = {}
        for sample, files in sample_map.items():
            file_paths = []
            for file in files:
                for fp in all_files:
                    if file in fp:
                        file_paths.append(fp)

            self.sample_file_paths[sample] = file_paths
        return self.sample_file_paths
    
    
    def make_output_feather(self, sample_file_paths, variables, variable_funcs):
        
        output_data = pd.DataFrame()
        cut_expr = self.master_config[self.master_config['cut_choice']]
        
        print(cut_expr)
        
        #Loop over all the samples and read using uproot
        for sample, files in sample_file_paths.items():
            
            print(f"Running {sample}...")
            if len(files) == 0:
                print(sample, 0)
                continue
                
            nominals = [f+':nominal' for f in files]
            
            #Use uproot to chain all the files together - has the potential for failing:
            #https://uproot.readthedocs.io/en/latest/basic.html#reading-many-files-into-big-arrays
            #TODO: Try and improve this, as it's slow!
            #array = uproot.concatenate(nominals, variables, cut=cut_expr, library='pd', allow_missing=True)
            
            if 'data' in sample:
                array = None
                for f in nominals:
                    for data in uproot.iterate(f, variables, cut=cut_expr, library='np', allow_missing=True):
                        if array is None:
                            array = pd.DataFrame(data, columns=variables)
                        else:
                            data = pd.DataFrame(data, columns=variables) 
                            array = pd.concat([array,data])
                    print(f"Got {len(array)} data events.")
            else:
                array = uproot.concatenate(nominals, variables, cut=cut_expr, library='np', allow_missing=True)
                array = pd.DataFrame(array, columns=variables)
            
            if type(array) == list or len(array) == 0:
                print(sample, array)
                continue
                
            #Add a sample name to the feather file
            name = sample.split('_')[1]
            array['sample'] = name
            print(sample, name, len(array))
            
            #Make additional variables
            start = perf_counter()
            for i, func in enumerate(variable_funcs):
                #print(f"Trying function: {func}")
                array = func(array)
                lap = perf_counter()
                print(f"Finished function {i}. Time taken: {round(lap-start,2)}s")
                
            output_data = pd.concat([output_data, array])
            
            
        output_data.reset_index(inplace=True)
        save_name = os.path.join(self.master_config['feather_path'],
                            f"{self.master_config['region_save_name']}.ftr")
        output_data.to_feather(save_name)
        print(f"Saved feather file to: {save_name}")
                
            
            

if __name__ == '__main__':
    
    
    s = perf_counter()
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config",default="feather/feather_config.yaml", help="Choose the master config to use")
    args = parser.parse_args()
    
    
    print(f"Starting up!\nMaking feather file using config: {args.config}")
    
    fm = FeatherMaker(master_config=args.config)
    print(f"Making feather from: {fm.master_config['nominal_path']}\nSaving feather to: {fm.master_config['feather_path']}")
    
    #Get the required variables from tree
    variables = fm.get_features()
    
    
    #Extract the file DSIDs to use
    #TODO: add an option to skip this in the case that you already have a mapping of sample : DSID 
    fm.extract_DSIDs(fm.master_config['samples_file'], fm.master_config['json_output'])
    
    #Get all the files needed
    all_files = fm.find_root_files(fm.master_config['nominal_path'], directory='')
    sample_file_paths = fm.make_sample_file_map(all_files)
    
    
    #Get user-defined functions to run over dataframe (output is saved to feather)
    from preprocessing.create_variables import VariableMaker
    vm = VariableMaker()
    
    #TODO: Make this into a config input
    func_strings = fm.master_config[fm.master_config['variable_functions_choice']]
    funcs = [getattr(vm, s) for s in func_strings]
    
    
    #Make the feather file
    fm.make_output_feather(sample_file_paths, variables, funcs)
    f = perf_counter()
    print(f"Finished running! \n Time taken: {round(f-s,2)}s.")
   
    