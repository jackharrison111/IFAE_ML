import numpy as np
import os
import pickle
import pandas as pd
from utils._utils import load_yaml_config


def get_dataset(region_choice, r_config, old_name=True, use_val=False):
    
    even_dir = os.path.join(r_config['even_base_dir'],r_config['regions'][region_choice]['even_path'])
    odd_dir = os.path.join(r_config['odd_base_dir'],r_config['regions'][region_choice]['odd_path'])
    
    if not use_val:
        even_dir = os.path.join(r_config['base_path'],r_config['even_eval_base'], r_config['regions'][region_choice]['even_eval'])
        odd_dir = os.path.join(r_config['base_path'],r_config['odd_eval_base'], r_config['regions'][region_choice]['odd_eval'])
    
    if old_name:
        outputs_filename = 'saved_outputs.pkl'
        val_outputs_filename = 'saved_val_outputs.pkl'
        sig_outputs_filename = 'saved_signal_outputs.pkl'
    else:
        outputs_filename = 'New_saved_outputs_2.pkl'
        val_outputs_filename = 'New_saved_val_outputs_2.pkl'
        sig_outputs_filename = 'New_saved_signal_outputs_2.pkl'
    

    print("Opening: ", even_dir, outputs_filename)
    with open(os.path.join(even_dir, outputs_filename), 'rb') as f:
        even_data = pickle.load(f)

    print("Sum weights start: ", sum(even_data['weights']))
    with open(os.path.join(odd_dir, outputs_filename), 'rb') as f:
        odd_data = pickle.load(f)
    
    
    if use_val:
        with open(os.path.join(even_dir, val_outputs_filename), 'rb') as f:
            even_val = pickle.load(f)

        with open(os.path.join(odd_dir, val_outputs_filename), 'rb') as f:
            odd_val = pickle.load(f)


    with open(os.path.join(even_dir, sig_outputs_filename), 'rb') as f:
        even_sig = pickle.load(f)

    with open(os.path.join(odd_dir, sig_outputs_filename), 'rb') as f:
        odd_sig = pickle.load(f)
        
    
    all_scores = np.append(even_data['ad_score'], odd_data['ad_score'])
    print("Even / odd weights:", sum(even_data['weights']), " ", sum(odd_data['weights']))
    all_w = np.append(even_data['weights'],odd_data['weights'])

    print("Here2 sumw: ", sum(all_w))
    
    if use_val:
        all_scores = np.append(all_scores, even_val['ad_score'])
        all_scores = np.append(all_scores, odd_val['ad_score'])

        print("even_val w: ", sum(even_val['weights']))
        all_weights = np.append(all_w, even_val['weights'])

        print("Here3 sumw: ", sum(all_weights))
        print("odd val w: ", sum(odd_val['weights']))
        all_weights = np.append(all_weights, odd_val['weights'])

    else:
        all_weights = all_w

    print(even_data.keys())
    
    
    all_sigs = np.append(even_sig['ad_score'],odd_sig['ad_score'])
    all_sig_w = np.append(even_sig['weights'], odd_sig['weights'])
    all_sig_samples = np.append(even_sig['samples'], odd_sig['samples'])
    #all_sig_inds = np.append(even_sig['index'], odd_sig['index'])
    
    #all_inds = np.append(even_data['index'], odd_data['index'])
    #all_inds = np.append(all_inds, even_val['index'])
    #all_inds = np.append(all_inds, odd_val['index'])
    
    output = {}
    print(sum(all_weights) ," = all w now...")
    output['all_scores'] = all_scores
    output['all_weights'] = all_weights
    output['all_sigs'] = all_sigs
    output['all_sig_w'] = all_sig_w
    output['all_sig_samples'] = all_sig_samples
    #output['all_inds'] = all_inds
    #output['all_sig_inds'] = all_sig_inds
    
    return output



def get_data_scaling(even_dir, odd_dir):
    
    #Get the min / max scaling values
    scale_file = 'NF_likelihood_scaling.txt'
    scale_file_odd = os.path.join(odd_dir,'scalers', scale_file)
    scale_file_even = os.path.join(even_dir,'scalers', scale_file)

    with open(scale_file_odd, 'r') as f:
        lines_o = f.readlines()

    nums = lines_o[-1].split(':')[-1].split(',')  #Get the last line of the file, everything after : and split by ,
    min_scaling, max_scaling = float(nums[0]),float(nums[1])
    return min_scaling, max_scaling



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




'''
Function to get the scores from a raw feather file.

Args:
    - feather_file: raw data file to load
    - r_config: the run config file (specifying the directories of the trainings of models)
    - region: the region name to run
    - feather_conf: the feather config file used to make the input variables (for the variable maker)

Note:
    - This uses a hard-coded weight function for 4lep events. Change this if needed!
    - This uses hard-coded sample selections. Change this if needed!

Copied from cells in 4lep.ipynb
'''
def get_data_from_feather(feather_file, r_config, region, 
                         apply_conv_veto=False,
                         scale_ad_score=True):

    print("Reading feather file...")
    data = pd.read_feather(feather_file)

    even_load_dir = os.path.join(r_config['even_base_dir'],r_config['regions'][region]['even_path'])
    odd_load_dir = os.path.join(r_config['odd_base_dir'],r_config['regions'][region]['odd_path'])
    train_conf = load_yaml_config(os.path.join(even_load_dir, 'config.yaml'))

    feather_conf = os.path.join("/nfs/pic.es/user/j/jharriso/IFAE_ML/configs/feather_configs/10GeV", f"{region}.yaml")
    if 'Q2' in region:
        feather_conf = os.path.join("/nfs/pic.es/user/j/jharriso/IFAE_ML/configs/feather_configs/10GeV/Q2", f"{region}.yaml")
    #################################################################################################################
    print("Making variables...")
    from feather.make_feather import FeatherMaker
    from preprocessing.create_variables import VariableMaker

    fm = FeatherMaker(master_config=feather_conf)
    vm = VariableMaker()
    variables = vm.varcols
    train_vars = [v for v in train_conf['training_variables'] if v not in ['eventNumber', 'weight', 'sample', 'index']]
    
    
    func_strings = fm.master_config[fm.master_config['variable_functions_choice']]  #Use the FeatherMaker
    funcs = [getattr(vm, s) for s in func_strings] #Use the VariableMaker
            
    for i, f in enumerate(funcs):
        all_data = f(data)
        print(f"Done function {func_strings[i]}.") #Add timing

    #################################################################################################################

    def calculate_mcweight(data, total_lum=140068.94):

        data.loc[data['RunYear'].isin([2015,2016]), 'lumi_scale'] = 36646.74*(1/total_lum)
        data.loc[data['RunYear'].isin([2017]), 'lumi_scale'] = 44630.6*(1/total_lum)
        data.loc[data['RunYear'].isin([2018]), 'lumi_scale'] = 58791.6*(1/total_lum)
    
        data['weight'] = data['lumi_scale']*data['custTrigSF_TightElMediumMuID_FCLooseIso_DLT']*data['weight_pileup']*data['jvtSF_customOR']*data['bTagSF_weight_DL1r_Continuous']*data['weight_mc']*data['xs']*data['lep_SF_CombinedLoose_0']*data['lep_SF_CombinedLoose_1']*data['lep_SF_CombinedLoose_2']*data['lep_SF_CombinedLoose_3']/data['totalEventsWeighted']
    
    
        #Need to multiply by total_lum
        data['weight'] = data['weight']*total_lum
    
        return data
    print("Calculating MC weight...")
    all_data = calculate_mcweight(all_data)

    #################################################################################################################
    print("Selecting samples...")
    #USE WITH ATANAY'S NEW FEATHERS
    chosen_samples = [
        'ZZ', 'ggZZ', 'ZZlow',
        'WZ', 'ggWZ', 'WZlow',
        'H4l',
        'VVV',
        'VHalt',
        'Zjets',
        'ZjetsInt',
        'Wjets',
        'VgammaNew',
        'ttbarnonallhad',
        'ttZMadNew','ttlllowMass',
        'ttWMG','ttWMGEW',
        'ttH',
        'singleToptchan', 'singleTopschan',
        'threeTop',
        'fourTop',
        'rareTop',
        'ttWW',
        'WtZ',
        'tW',
        'tZ',
        'tHjb',
        'ttZZ','ttWH', 'ttHH'
    ]
    '''
    chosen_samples = ['ttbarnonallhad', 'ttW2210', 'ttW2210EW', 'ttH',
           'ttlllowMass', 'rareTop', 'ZZ', 'WZ', 'ggZZ', 'Zjets',
           'tW', 'threeTop', 'fourTop', 'ttWW', 'tZ', 'WtZ',
           'VVV', 'VH', 'tHjb', 'tWH', 'ttZZ', 'ttWH', 'ttHH', 'ttZMadNew']
    '''
    all_data = all_data.loc[all_data['sample'].isin(chosen_samples)]


    ################################################################################################################

    if apply_conv_veto:
        print("Applying conversion veto to the data...")
        print("Length of data before: ", len(all_data))
        all_data = all_data.loc[(all_data['lep_ID_0']==0)|((all_data['lep_ambiguityType_0']==0)&(all_data['lep_DFCommonAddAmbiguity_0']<1))]
        all_data = all_data.loc[(all_data['lep_ID_1']==0)|((all_data['lep_ambiguityType_1']==0)&(all_data['lep_DFCommonAddAmbiguity_1']<1))]
        all_data = all_data.loc[(all_data['lep_ID_2']==0)|((all_data['lep_ambiguityType_2']==0)&(all_data['lep_DFCommonAddAmbiguity_2']<1))]
        all_data = all_data.loc[(all_data['lep_ID_3']==0)|((all_data['lep_ambiguityType_3']==0)&(all_data['lep_DFCommonAddAmbiguity_3']<1))]
        all_data = all_data.loc[(all_data['lep_ID_4']==0)|((all_data['lep_ambiguityType_4']==0)&(all_data['lep_DFCommonAddAmbiguity_4']<1))]
        if 'lep_DFCommonAddAmbiguity_5' in all_data.columns:
            all_data = all_data.loc[(all_data['lep_ID_5']==0)|((all_data['lep_ambiguityType_5']==0)&(all_data['lep_DFCommonAddAmbiguity_5']<1))]
        print("Length of data after: ", len(all_data))
        '''
        XXX_ConvVeto: (((lep_ID_0==0)||((lep_ID_0!=0)&&(lep_ambiguityType_0==0 && lep_DFCommonAddAmbiguity_0<1)))&& ((lep_ID_1==0)||((lep_ID_1!=0)&&(lep_ambiguityType_1==0 && lep_DFCommonAddAmbiguity_1<1)))&& ((lep_ID_2==0)||((lep_ID_2!=0)&&(lep_ambiguityType_2==0 && lep_DFCommonAddAmbiguity_2<1)))&& ((lep_ID_3==0)||((lep_ID_3!=0)&&(lep_ambiguityType_3==0 && lep_DFCommonAddAmbiguity_3<1)))&& ((lep_ID_4==0)||((lep_ID_4!=0)&&(lep_ambiguityType_4==0 && lep_DFCommonAddAmbiguity_4<1)))&& ((lep_ID_5==0)||((lep_ID_5!=0)&&(lep_ambiguityType_5==0 && lep_DFCommonAddAmbiguity_5<1))))
        '''
    

    #################################################################################################################

    # Split into even / odd for 
    even = all_data.loc[all_data['eventNumber']%2==0]
    odd = all_data.loc[all_data['eventNumber']%2 ==1]
    
    even_w = even['weight']
    odd_w = odd['weight']

    even_inds = even['index']
    odd_inds = odd['index']

    even = even[train_vars]
    odd = odd[train_vars]

    #################################################################################################################
    print("StandardScaling input variables...")
    # Get the scalers
    import pickle 
    
    def scale_data(data, scaler_paths) :
        for col in data.columns:
            scaler = pickle.load(open(os.path.join(scaler_paths,f'scalers/{col}_scaler.pkl'),'rb'))
            transformed = scaler.transform(data[col].to_numpy().reshape(-1,1))
            data.loc[:,col] = transformed
        return data
    
    #Scale the even and odd separately
    
    even = scale_data(even, even_load_dir)
    odd = scale_data(odd, odd_load_dir)


    ################################################################################################################
    #Get the model
    print("Loading models...")
    from model.model_getter import get_model
    import torch
    
    even_config = load_yaml_config(os.path.join(odd_load_dir, 'config.yaml'))
    even_model = get_model(conf=os.path.join(odd_load_dir, 'config.yaml'))
    even_model.load_state_dict(torch.load(os.path.join(odd_load_dir,'model_state_dict.pt')))
    
    odd_config = load_yaml_config(os.path.join(even_load_dir, 'config.yaml'))
    odd_model = get_model(conf=os.path.join(even_load_dir, 'config.yaml'))
    odd_model.load_state_dict(torch.load(os.path.join(even_load_dir,'model_state_dict.pt')))

    ################################################################################################################
    #Get the scores

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


    ################################################################################################################
    #Get the rescaling the same way as done in ROOT ntuples
    print("Reading min/max scaling...")

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

    

    ################################################################################################################
    #Evaluate model
    print("Evaluating even/odd models...")
    if not scale_ad_score:
        print("Ignoring scaling...")
        even_scores = get_anomaly_score(even_model, even)
        odd_scores = get_anomaly_score(odd_model, odd)
    else:
        even_scores = get_anomaly_score(even_model, even, min_loss=min_all, max_loss=max_all)
        odd_scores = get_anomaly_score(odd_model, odd, min_loss=min_all, max_loss=max_all)
    
    even_scores = even_scores.detach().numpy()
    odd_scores = odd_scores.detach().numpy()


    ################################################################################################################

    all_scores =  np.append(even_scores, odd_scores)
    all_w = np.append(even_w, odd_w)
    all_inds = np.append(even_inds, odd_inds)

    dataset = {}
    dataset['all_scores'] = all_scores
    dataset['all_weights'] = all_w
    dataset['all_inds'] = all_inds
    return dataset
