import uproot
import pandas as pd
import numpy as np
import os

#!/usr/bin/python -tt
from tqdm import tqdm

class NtupleWriter():
    
    def __init__(self):
        print("Made class: NtupleWriter")
        
    def add_score_to_ntuple(self, df_with_score, file_path, outfile, merge_col='eventNumber', new_col='vae_score', default_val=-99):
        with uproot.open(file_path + ':nominal') as evts:
            all_evts = evts.arrays(library='pd')
            if not 'eventNumber' in all_evts.columns:
                print(f"Event number not found in root file {file_path}")
                return None

            merged_df = all_evts.merge(df_with_score, on=merge_col, how='left')
            if new_col not in merged_df:
                print(f"Adding dummy {new_col}")
                merged_df[new_col] = default_val
            merged_df[new_col] = merged_df[new_col].replace(np.nan, default_val)

        if not os.path.exists(os.path.split(outfile)[0]):
            os.makedirs(os.path.split(outfile)[0])
            
        with uproot.recreate(outfile) as f:
            f['nominal'] = merged_df
        
        print(f"Added {new_col} to {file_path} and saved under {outfile}")
    
    
    def find_root_files(self, root, directory, master_list=[]):
        files = os.listdir(os.path.join(root,directory))
        for f in files:
            if '.root' in f:
                master_list.append(os.path.join(os.path.join(root,directory),f))
                continue
            else:
                master_list = self.find_root_files(os.path.join(root,directory),f, master_list)
        return master_list
    
    
    def update_ntuples(self, vae_output, ntuple_directory):
        
        #`Find all files 
        all_files = self.find_root_files(ntuple_directory, '', [])
        
        #Make df out of the vae_output
        df = {
            'eventNumber': vae_output['eventNumber'],
            'vae_score': vae_output['log_losses']
        }
        df = pd.DataFrame(df)
        
        for i, f in enumerate(all_files):
            
            self.update_ntuple_file(df, f)
            if i % 100 == 0:
                print(f"Finished file {i} out of {len(f)}")
        
    
    def update_ntuple_file(self, vae_df, file):
        
        with uproot.open(file+':nominal') as evts:
            
            evtNumScore = evts.arrays(library='pd')
            df = evtNumScore.merge(vae_df, on='eventNumber', how='left')
            df['vae_score'] = df['vae_score_y'].combine_first(df['vae_score_x'])
            df.drop(columns=['vae_score_x','vae_score_y'],inplace=True)
        
        with uproot.recreate(file) as evts:
            evts['nominal'] = df
    
    
    def process_output(self, vae_output, base_directory, output_directory):

        #`Find all files 
        all_files = self.find_root_files(base_directory, '', [])

        #Make df out of the vae_output
        df = {
            'eventNumber': vae_output['eventNumber'],
            'vae_score': vae_output['log_losses']
        }
        df = pd.DataFrame(df)

        #loop over the files and add  score to ntuple, making the right output name
        for f in tqdm(all_files):
            path = f.split(base_directory)[1][1:]
            print(path)
            self.add_score_to_ntuple(df, f, os.path.join(output_directory, path))
        ...




if __name__ == '__main__':

	base_dir = '/data/at3/scratch3/jharrison/nominal'
	out_dir = '/data/at3/scratch3/jharrison/mod_ntuples'

	nw = NtupleWriter()
	nw.process_output(None, base_dir, out_dir)
