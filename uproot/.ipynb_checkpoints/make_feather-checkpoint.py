import os
import uproot

'''
Function to read the lines from a text file
File should be in format:
Var1\n
Var2\n
'''
def get_features(infile):
    
    with open(infile, 'r') as file:
        vars = file.readlines()
        vars = [v.rstrip() for v in vars if v[0] != '#']
    return vars


'''
Function to recursively find root files contained within a directory.
Files must end in .root
'''
def find_root_files(root, directory, master_list=[]):
    files = os.listdir(os.path.join(root,directory))
    for f in files:
        if '.root' in f:
            master_list.append(os.path.join(os.path.join(root,directory),f))
            continue
        else:
            master_list = find_root_files(os.path.join(root,directory),f, master_list)
            
    return master_list


if __name__ == '__main__':
    
    #Set the config inputs
    variables_file = 'configs/VLL_variables.txt'
    samples_file = 'configs/VLL_samples.txt'
    json_output = 'configs/VLL_samples.json'
    feather_path = "/data/at3/scratch3/jharrison/nominal_feather"
    nominal_path = "/data/at3/scratch3/jharrison/VLL/"
    region_name = 'CR_1Z_0b_2SFOS_VLLs'
    
    #Add this to reading from a file?
    cut_choice = '(total_charge==0) & (nTaus_OR==0) & (nJets_OR_DL1r_77==0)  & (lep_Pt_0*1e-3>25) & (lep_Pt_1*1e-3>25) & (lep_Pt_2*1e-3>25) & (lep_Pt_3*1e-3>25) & (lep_isolationFCLoose_0) & (lep_isolationFCLoose_1) & (lep_isolationFCLoose_2) & (lep_isolationFCLoose_3) & ((lep_ID_0==-lep_ID_1)&(abs(Mll01-91.2e3)<10e3)&((lep_ID_2!=lep_ID_3) | (abs(Mll23-91.2e3)>10e3)) | (lep_ID_0==-lep_ID_2)&(abs(Mll02-91.2e3)<10e3)&((lep_ID_1!=lep_ID_3) | (abs(Mll13-91.2e3)>10e3)) |(lep_ID_0==-lep_ID_3)&(abs(Mll03-91.2e3)<10e3)&((lep_ID_1!=lep_ID_2) | (abs(Mll12-91.2e3)>10e3)) |(lep_ID_1==-lep_ID_2)&(abs(Mll12-91.2e3)<10e3)&((lep_ID_0!=lep_ID_3) | (abs(Mll03-91.2e3)>10e3)) |(lep_ID_1==-lep_ID_3)&(abs(Mll13-91.2e3)<10e3)&((lep_ID_0!=lep_ID_2) | (abs(Mll02-91.2e3)>10e3)) |(lep_ID_2==-lep_ID_3)&(abs(Mll23-91.2e3)<10e3)&((lep_ID_0!=lep_ID_1) | (abs(Mll01-91.2e3)>10e3))) & ( (((lep_ID_0==-lep_ID_1)&(lep_ID_2==-lep_ID_3))) | (((lep_ID_0==-lep_ID_2)&(lep_ID_1==-lep_ID_3))) | (((lep_ID_0==-lep_ID_3)&(lep_ID_1==-lep_ID_2))) )'
    
    #Get the required variables from tree
    variables = get_features(variables_file)
    
    #Make output dataframe
    import pandas as pd
    output_data = pd.DataFrame()
    
    #Extract the file DSIDs to use
    from make_samples_json import extract_DSIDs
    extract_DSIDs(samples_file, json_output)
    
    #Get all the files needed
    all_files = find_root_files(nominal_path, directory='')
    
    #Read the mapping of sample : DSIDs
    import json
    with open(json_output, 'r') as f:
        sample_map = json.load(f)
        
    #Transform from sample : DSIDs to sample : file_path
    sample_file_paths = {}
    for sample, files in sample_map.items():
        file_paths = []
        for file in files:
            for fp in all_files:
                if file in fp:
                    file_paths.append(fp)
        
        sample_file_paths[sample] = file_paths
   
    #Loop over all the samples and read using uproot
    for sample, files in sample_file_paths.items():
        if len(files) == 0:
            print(sample, 0)
            continue
        nominals = [f+':nominal' for f in files]
        
        #Use uproot to chain all the files together - has the potential for failing:
        #https://uproot.readthedocs.io/en/latest/basic.html#reading-many-files-into-big-arrays
        array = uproot.concatenate(nominals, variables, cut=cut_choice, library='pd', allow_missing=False)
        if type(array) == list:
            print(sample, array)
            continue
            
        #Add a sample name to the feather file
        name = sample.split('_')[1]
        array['sample'] = name
        print(sample, name, len(array))
        output_data = pd.concat([output_data, array])
   
    output_data.reset_index(inplace=True)
    output_data.to_feather(os.path.join(feather_path, f"Regions/{region_name}.ftr"))