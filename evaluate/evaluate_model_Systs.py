
import uproot
import numpy as np
import pandas as pd
import gc
import argparse
from model.model_getter import get_model
from utils._utils import load_yaml_config, find_root_files
import torch
import os 
#from preprocessing.create_variables import VariableMaker
import pickle

#Temporarily ignore scikit learn warnings
import warnings
warnings.filterwarnings('ignore')


def needs_reevaluating(nom_filename, eval_filename):

    #If no file already
    if not os.path.exists(eval_filename):
        return True

    print("Nom file: ", nom_filename)
    print("Eval file: ", eval_filename)
    
    #If file
    try:
        nom_f = uproot.open(nom_filename)
    except:
        print("Couldn't open the nominal file, returning True to re-evaluate!")
        return True

    try:
        eval_f = uproot.open(eval_filename)
    except:
        print("Couldn't open the eval file, returning True to re-evaluate!")
        return True

    nom_keys = nom_f.keys()
    eval_keys = eval_f.keys()
    eval_keys_split = [key.split(';')[0] if ';' in key else key for key in eval_keys]
    
    #if len(nom_keys) != len(eval_keys):
    #    print("Got different numbers of trees in nom/eval, re-evaluating!")
    #    return True

    diff_flag = False
    for key in nom_keys:
        if ";" in key:
            key = key.split(';')[0]
        if key not in eval_f.keys() and key not in eval_keys_split:
            if nom_f[key].num_entries == 0:
                print(f"Found empty tree {key} in nominal, skipping...")
            else:
                print(f"Found problem with non-empty file in nominal and no tree {key} in syst")
                diff_flag=True
            continue
                
        print("Found ", nom_f[key].num_entries, " entries in nom ", key," compared to: ", eval_f[key].num_entries)
        if nom_f[key].num_entries != eval_f[key].num_entries:
            diff_flag = True
            #print("Found different entries in tree:", key)

    if not diff_flag:
        return False
    
    return True

def scale_log_prob(log_probs, min_prob=None, max_prob=None, use_01=True):
    
        if use_01:
            log_probs = -log_probs
        
        if min_prob is None:
            min_prob = log_probs.min()
            max_prob = log_probs.max()
            min_prob = min_prob
            max_prob = max_prob

        log_probs = (log_probs - min_prob) / (max_prob - min_prob)
        print(f"Scaled loglikelihood using {min_prob} , {max_prob} to mean: {log_probs.mean()}")
        return log_probs, min_prob, max_prob


def get_anomaly_score(model, df, **kwargs):
    
    with torch.no_grad():
        df = torch.Tensor(df.values)
        outputs = model(df)
    
        #Get the loss
        losses = model.loss_function(**outputs)
                
        #For AE the AD score is the reco loss
        #For NF the AD score is the log prob (needs scaling over the whole of the input dataset)
        #Make an anomaly score
        
        anomaly_score = model.get_anomaly_score(losses, **kwargs)
        
        #Add scaling of anomaly score? DO WE WANT THIS? OR PUT IT AS A POSTPROCESSING STEP? 
        
    return anomaly_score.reshape(-1,1)


def predict_model(model, arr_frame, scaler_path, training_vars, **kwargs):
    
    scores = np.empty([len(arr_frame), 1], dtype=np.float16)
    
    arr_frame = arr_frame[training_vars]

    #Scale and return default for default inputs
    batch_size = 100000
    BATCH_INDICES = np.arange(start=0, stop=len(arr_frame), step=batch_size)  # row indices of batches
    BATCH_INDICES = np.append(BATCH_INDICES, len(arr_frame))  # add final batch_end row

    
    for index in np.arange(len(BATCH_INDICES) - 1):
        batch_start = BATCH_INDICES[index]  # first row of the batch
        batch_end = BATCH_INDICES[index + 1]  # last row of the batch
        
        
        data = arr_frame[batch_start:batch_end]
        
        
        #If any are default values then 
        default_indices = np.where(np.any(data == -999, axis=1))[0]
        
        non_default_indices = np.where(np.all(data!=-999, axis=1))[0]

        if len(default_indices) > 0:
            scores[batch_start:batch_end][default_indices] = -999
        
        if len(non_default_indices) > 0:
            print(f"Found {len(non_default_indices)} interesting events.")
            non_default_data = scale_data(data.iloc[non_default_indices].copy(), scaler_path)
            scores[batch_start:batch_end][non_default_indices] = get_anomaly_score(model,
                                                            non_default_data, **kwargs)
            
            _ = gc.collect()
        
    return scores


def file_past_date(outfile):

    if not os.path.exists(outfile):
        return True
    
    from datetime import datetime
    september_1st = datetime(2024, 9, 26)
    
    if os.path.isfile(outfile):
        # Get the modification time or creation time depending on your OS
        file_stat = os.stat(outfile)
        creation_time = datetime.fromtimestamp(file_stat.st_ctime)  # Creation time on Windows
        modification_time = datetime.fromtimestamp(file_stat.st_mtime)  # Modification time
        print(creation_time, " , ", modification_time)
        
        # Compare with September 1st, 2024
        if creation_time >= september_1st and modification_time >= september_1st:
            print(f"{outfile} was created/modified on or after September 1st, 2024")
            return True
        else:
            return False
    else:
        print("Found a non-file when checking date...")
        return True


def scale_data(data, scaler_paths) :
    for col in data.columns:
        scaler = pickle.load(open(os.path.join(scaler_paths,f'scalers/{col}_scaler.pkl'),'rb'))
        transformed = scaler.transform(data[col].to_numpy().reshape(-1,1))
        data.loc[:,col] = transformed
    return data


if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser("Running model evaluation")
    
    parser.add_argument("-i","--inputConfig",action="store", help="Set the input config to use, default is 'configs/training_config.yaml'", default="configs/training_config.yaml", required=False)
    
    parser.add_argument("-f","--featherConfig",action="store", help="Set the feather config to use, default is 'feather/feather_config.yaml'", default="feather/feather_config.yaml", required=False)
    
    parser.add_argument("-e","--evenModelPath",action="store", help="Set the trained even model to use", default="results/AllSigs/1Z_0b_2SFOS_AllSigs/Run_1159-21-04-2023", required=False)
    
    parser.add_argument("-o","--oddModelPath",action="store", help="Set the trained odd model to use", default="outputs/ODD_FINAL_VAE_1414-23-09-2022", required=False)
    

    parser.add_argument("-r","--Region",action="store", help="Set the region to config use, default is to use the config", default=None, required=False)
    
    parser.add_argument("-first","--First",action="store", help="Set the first file to run over", 
                        default=-1, required=False, type=int)
    
    parser.add_argument("-last","--Last",action="store", help="Set the last file to run over", 
                        default=-1, required=False, type=int)
    
    parser.add_argument("-ni","--NtuplePathIn",action="store", help="Pass a custom ntuple path for all to read", 
                        default=False, required=False)
    
    parser.add_argument("-no","--NtuplePathOut",action="store", help="Pass a custom ntuple path for outputting", 
                        default=False, required=False)


    args = parser.parse_args()
    
    
    even_load_dir = args.evenModelPath
    odd_load_dir = args.oddModelPath
    feather_conf = args.featherConfig
    region = args.Region
    first = args.First
    last = args.Last

    check_outdate = True  #Flag for checking if after September 1st
    check_reeval = True  #Flag for checking if output trees are the same length or reeval
    use_old_vm = True  #Flag to use 'working' evaluations
    use_file_skipper = True
    
    #######################################################################################
    
    #Load even and odd model
    even_config = load_yaml_config(os.path.join(odd_load_dir, 'config.yaml'))
    even_model = get_model(conf=os.path.join(odd_load_dir, 'config.yaml'))
    even_model.load_state_dict(torch.load(os.path.join(odd_load_dir,'model_state_dict.pt')))
    
    odd_config = load_yaml_config(os.path.join(even_load_dir, 'config.yaml'))
    odd_model = get_model(conf=os.path.join(even_load_dir, 'config.yaml'))
    odd_model.load_state_dict(torch.load(os.path.join(even_load_dir,'model_state_dict.pt')))
    
    #Get the required variables from tree
    from feather.make_feather import FeatherMaker
    fm = FeatherMaker(master_config=feather_conf)

    train_conf = odd_config #Use the one from the even directory


    #Set a new ntuple path
    #train_conf['ntuple_path'] = '/data/at3/common/multilepton/VLL_production/nominal'

    if args.NtuplePathIn:
        #train_conf['ntuple_path'] = '/data/at3/scratch2/multilepton/Shalini_ntuples/VLL_newsamples_NN'
        train_conf['ntuple_path'] = args.NtuplePathIn

    if args.NtuplePathOut:
        #train_conf['ntuple_outpath'] = '/data/at3/common/multilepton/VLL_production/evaluations/Shalini_ntuples'
        train_conf['ntuple_outpath'] = args.NtuplePathOut

    
    SCORE_NAME =  f"Score_{region}_{train_conf['model_type']}"
    
    
    if odd_config['model_type'] == 'NF':

        
        #Scale the anomaly score distributions properly

        #Read the odd and even scores from the training...
        outputs_filename = 'saved_outputs.pkl'
        val_outputs_filename = 'saved_val_outputs.pkl'
        
        if 'Q2' in region:
            outputs_filename = 'New_saved_outputs_2.pkl'
            val_outputs_filename = 'New_saved_val_outputs_2.pkl'
        
        #scale_file_odd = os.path.join(odd_load_dir,'scalers/NF_likelihood_scaling.txt')
        #scale_file_even = os.path.join(even_load_dir, 'scalers/NF_likelihood_scaling.txt')
        
        with open(os.path.join(even_load_dir, outputs_filename), 'rb') as f:
            even_data = pickle.load(f)

        with open(os.path.join(even_load_dir, val_outputs_filename), 'rb') as f:
            even_val = pickle.load(f)

        with open(os.path.join(odd_load_dir, outputs_filename), 'rb') as f:
            odd_data = pickle.load(f)

        with open(os.path.join(odd_load_dir, val_outputs_filename), 'rb') as f:
            odd_val = pickle.load(f)

        all_even_scores = np.append(even_data['ad_score'], even_val['ad_score'])
        all_odd_scores = np.append(odd_data['ad_score'], odd_val['ad_score'])
        
        # Need to get the 0 and 99.9% percentile of ALL the scores
        all_scores = np.append(all_even_scores, all_odd_scores)
          
        scale_max = max(all_scores)
        scale_min = np.percentile(all_scores, 0.01)
        
        min_all = -scale_max
        max_all = -scale_min
    
    
    #Loop over predefined number of files
    print(train_conf['ntuple_path'])
    all_root_files = find_root_files(train_conf['ntuple_path'], '', [])


    if use_old_vm:
        from preprocessing.create_variables_old import VariableMaker
    else:
        from preprocessing.create_variables import VariableMaker
    number_to_eval = 0
    indices_to_eval = []
    for i, file in enumerate(all_root_files):
        
        if i < first and first!=-1:
            continue
        if i > last and last!=-1:
            break
        
        print(f"Running file {file}. {i} / {len(all_root_files)}")
        
        #Check if the file already exists and don't recreate otherwise
        if 'tmp' in file:
            print("Skipping file as found tmp in path...")
            continue

        if use_file_skipper:
            name = '/nfs/pic.es/user/j/jharriso/IFAE_ML/evaluate/needEval.txt'
            with open(name,'r') as f:
                files_to_skip = f.readlines()
            files_to_skip = [(s.split(' ')[0],s.split(' ')[1])for s in files_to_skip]
            file_to_run = False
            for f in files_to_skip:
                if f[0].strip().replace('\n', '') in file and f[1].strip().replace('\n', '') in file:
                    file_to_run=True

            if not file_to_run:
                print("Skipping file as not in: ", name)
                continue
                
        save_path = file.split(os.path.basename(train_conf['ntuple_path']))[1]
        if save_path[0] == '/':
            save_path = save_path[1:]
            
        outdir = os.path.join(train_conf['ntuple_outpath'], region)
        whole_out_string = os.path.join(outdir, save_path)

        #Check whether the file has been produced recently or not
        if check_outdate:
            if os.path.exists(whole_out_string):
                print("Skipping file as outfile already exists")
                continue
            if not file_past_date(whole_out_string):
                print("Found file that was created before September 1st... Skipping!")
                continue
        if check_reeval:
            if not needs_reevaluating(file, whole_out_string):
                print("Found file that doesn't need evaluating... skipping!")
                continue
        
        #if os.path.exists(whole_out_string):
        #    print(f"Skipping file {file}, since output file already exists: {whole_out_string}")
        #    continue

        number_to_eval+=1 
        indices_to_eval.append(i)
        continue
    
        #Make the variables needed into a df:
        vm = VariableMaker()
        variables = vm.varcols
        train_vars = [v for v in train_conf['training_variables'] if v not in ['eventNumber', 'weight', 'sample', 'index']]
        
        non_produced = list(set(train_vars).difference(set(vm.created_cols)))
        
        func_strings = fm.master_config[fm.master_config['variable_functions_choice']]
        funcs = [getattr(vm, s) for s in func_strings]
        
        
        #Get all trees
        try:
            with uproot.open(file) as f:
                trees = f.keys()
            trees = [t[:-2] for t in trees]
        except:
            print("Couldn't open file: ", file, " - Skipping...")
            continue
        
        #Loop over each tree
        for j,tree in enumerate(trees):
        
            print(f"Running tree {tree}. {j} / {len(trees)}")
        
            try:
                f = uproot.open(file + f":{tree}",library="pd")

                if len(f.keys()) == 0:
                    print(f"WARNING:: Found no events in tree: {tree}.")
                    
                    #TODO: Should write empty tree here
                    continue
            except:
                print(f"Found bad file/tree, and cannot open:  {file}/{tree}.")
                continue
        
            #Get the variables that we need
            #Only load the variables needed for the variable maker, plus the others needed in the train_vars
            #that aren't produced

            all_data = f.arrays(list(set(variables+non_produced)),library="pd")
            events = f.arrays(["eventNumber"],library="np")

            print("\n--- Memory usage: ---")
            all_data.info(verbose = False, memory_usage = 'deep')
            print("---------------------\n")

            for i, f in enumerate(funcs):
                all_data = f(all_data)
                print(f"Done function {i}.") #Add timing

            

            print("Total size of events: " , len(all_data))

            if odd_config['model_type'] == 'NF':
                #Check whether any of the input variables are default, and then return a default value for the model
                even_scores = predict_model(even_model, all_data.loc[all_data['eventNumber'] % 2 == 0],
                                           scaler_path = even_load_dir, training_vars=train_vars, 
                                            min_loss=min_all, max_loss=max_all)
                odd_scores = predict_model(odd_model, all_data.loc[all_data['eventNumber'] % 2 == 1],
                                          scaler_path = odd_load_dir, training_vars=train_vars,
                                          min_loss=min_all, max_loss=max_all)
            else:
                #Check whether any of the input variables are default, and then return a default value for the model
                even_scores = predict_model(even_model, all_data.loc[all_data['eventNumber'] % 2 == 0],
                                           scaler_path = even_load_dir, training_vars=train_vars)
                odd_scores = predict_model(odd_model, all_data.loc[all_data['eventNumber'] % 2 == 1],
                                          scaler_path = odd_load_dir, training_vars=train_vars)


            all_scores = np.empty([events["eventNumber"].size,1])
            all_scores[events["eventNumber"]%2==0] = even_scores
            all_scores[events["eventNumber"]%2==1] = odd_scores



            events[SCORE_NAME] = all_scores.reshape(-1)

            ######################################
            # Add unscaled output variables here #
            ######################################

            for col in train_vars:
                events[f"{col}_{region}_{train_conf['model_type']}"] = all_data[col]


            outdir = os.path.join(train_conf['ntuple_outpath'], region)


            whole_out_string = os.path.join(outdir, save_path)
            print("Saving to: ", whole_out_string)

            print("Got base path: ", os.path.split(whole_out_string)[0])
            if not os.path.exists(os.path.split(whole_out_string)[0]):
                os.makedirs(os.path.split(whole_out_string)[0])
                
            #WRITE THEM ALL INTO SEPARATE FILES AND THEN JUST HADD THEM AFTER
            #Replace .root with tree.root
            out_string_with_tree = whole_out_string.replace('.root', f"_TREE_{tree}.root")

            print("Saving tree to: ", out_string_with_tree)
            with uproot.recreate(out_string_with_tree) as rootfile:
                rootfile[tree]=events
                gc.collect()
                
            '''    
            #Check if the first tree to be written instead of checking if it already exists
            #if os.path.exists(whole_out_string):
            if j != 0:
                
                #IF THIS FAILS, MAKE A NEW FILE AND HADD IT
                
                with uproot.update(whole_out_string) as rootfile:
                    rootfile[tree] = events
                    gc.collect()
            else:
                with uproot.recreate(whole_out_string) as rootfile:
                    rootfile[tree]=events
                    gc.collect()
            '''
            print(f"Done tree {tree}...")

            if tree == trees[-1]:
                #Hadd all the trees
                all_tree_files = whole_out_string.replace('.root', '_TREE_*.root')

                #hadd -f targetfile source1 source2 ...
                print("Hadd-ing all trees to: ",whole_out_string)
                #print(f"hadd -f {whole_out_string} {all_tree_files}")
                os.system(f"hadd -f {whole_out_string} {all_tree_files}")
                #os.system(f"rm {all_tree_files}")

    
    print("Finished running over all requested files.")
    print(f"Had {number_to_eval} to rerun...")
    print(f"With indices: ", indices_to_eval)
    
    #################################################################################
