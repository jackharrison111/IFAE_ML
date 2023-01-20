
import uproot
import argparse
from make_feather import FeatherMaker
from time import perf_counter
import numpy as np
import pandas as pd 

#Define cut string
cut_1Z_0b_2SFOS = '(total_charge==0) & (nTaus_OR==0) & (nJets_OR_DL1r_77==0)  & (lep_Pt_0*1e-3>25) & (lep_Pt_1*1e-3>25) & (lep_Pt_2*1e-3>25) & (lep_Pt_3*1e-3>25) & (lep_isolationFCLoose_0) & (lep_isolationFCLoose_1) & (lep_isolationFCLoose_2) & (lep_isolationFCLoose_3) & ((lep_ID_0==-lep_ID_1)&(abs(Mll01-91.2e3)<10e3)&((lep_ID_2!=lep_ID_3) | (abs(Mll23-91.2e3)>10e3)) | (lep_ID_0==-lep_ID_2)&(abs(Mll02-91.2e3)<10e3)&((lep_ID_1!=lep_ID_3) | (abs(Mll13-91.2e3)>10e3)) |(lep_ID_0==-lep_ID_3)&(abs(Mll03-91.2e3)<10e3)&((lep_ID_1!=lep_ID_2) | (abs(Mll12-91.2e3)>10e3)) |(lep_ID_1==-lep_ID_2)&(abs(Mll12-91.2e3)<10e3)&((lep_ID_0!=lep_ID_3) | (abs(Mll03-91.2e3)>10e3)) |(lep_ID_1==-lep_ID_3)&(abs(Mll13-91.2e3)<10e3)&((lep_ID_0!=lep_ID_2) | (abs(Mll02-91.2e3)>10e3)) |(lep_ID_2==-lep_ID_3)&(abs(Mll23-91.2e3)<10e3)&((lep_ID_0!=lep_ID_1) | (abs(Mll01-91.2e3)>10e3))) & ( (((lep_ID_0==-lep_ID_1)&(lep_ID_2==-lep_ID_3))) | (((lep_ID_0==-lep_ID_2)&(lep_ID_1==-lep_ID_3))) | (((lep_ID_0==-lep_ID_3)&(lep_ID_1==-lep_ID_2))) )'



parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config",default="feather/feather_config.yaml", help="Choose the master config to use")
args = parser.parse_args()
print(f"Starting up!\nMaking feather file using config: {args.config}")
fm = FeatherMaker(master_config=args.config)


all_files = fm.find_root_files(fm.master_config['nominal_path'], directory='')
sample_file_paths = fm.make_sample_file_map(all_files)

#print(sample_file_paths)

#Get the required variables from tree
variables = fm.get_features()

cut_expr = cut_1Z_0b_2SFOS
#Loop over all the samples and read using uproot
times = {}

for sample, files in sample_file_paths.items():
    print(f"Running sample: {sample}")
    nominals = [f+':nominal' for f in files]
    if len(nominals) == 0:
        continue
    s = perf_counter()
    
    if 'data' in sample:
        all_data = None
        for f in nominals:
            for data in uproot.iterate(f, variables, cut=cut_expr, library='np', allow_missing=True):
                if all_data is None:
                    all_data = pd.DataFrame(data, columns=variables)
                else:
                    data = pd.DataFrame(data, columns=variables) 
                    all_data = pd.concat([all_data,data])
                print(all_data.head())
            l = perf_counter()
            print(f"Taken {l-s}s to open {f}.")
    else:
        #all_data = uproot.concatenate(nominals, variables, cut=cut_expr, library='pd', allow_missing=True)
        all_data = uproot.concatenate(nominals, variables, cut=cut_expr, library='np', allow_missing=True)
        all_data = pd.DataFrame(all_data, columns=variables)
        
    f = perf_counter()
    times[sample] = f-s
    print(all_data.head())
    break
for sample, time in times.items():
    print(f"Time to load {sample}: {round(time,3)}s.")
