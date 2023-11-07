import pickle
import os
import yaml
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd


def get_dataset(region_choice, r_config):
    
    even_dir = os.path.join(r_config['even_base_dir'],r_config['regions'][region_choice]['even_path'])
    odd_dir = os.path.join(r_config['odd_base_dir'],r_config['regions'][region_choice]['odd_path'])
    
    outputs_filename = 'New_saved_outputs_2.pkl'
    val_outputs_filename = 'New_saved_val_outputs_2.pkl'
    sig_outputs_filename = 'New_saved_signal_outputs_2.pkl'
    
    
    #outputs_filename = 'saved_outputs.pkl'
    #val_outputs_filename = 'saved_val_outputs.pkl'
    #sig_outputs_filename = 'saved_signal_outputs.pkl'
    
    
    
    #new_bkg_outputs Use this for the old run
    print("Opening: ", even_dir, outputs_filename)
    with open(os.path.join(even_dir, outputs_filename), 'rb') as f:
            even_data = pickle.load(f)

    with open(os.path.join(even_dir, val_outputs_filename), 'rb') as f:
        even_val = pickle.load(f)

    with open(os.path.join(odd_dir, outputs_filename), 'rb') as f:
        odd_data = pickle.load(f)

    with open(os.path.join(odd_dir, val_outputs_filename), 'rb') as f:
        odd_val = pickle.load(f)

    with open(os.path.join(even_dir, sig_outputs_filename), 'rb') as f:
        even_sig = pickle.load(f)

    with open(os.path.join(odd_dir, sig_outputs_filename), 'rb') as f:
        odd_sig = pickle.load(f)
        
    all_scores = np.append(even_data['ad_score'], odd_data['ad_score'])
    all_w = np.append(even_data['weights'],odd_data['weights'])

    all_scores = np.append(all_scores, even_val['ad_score'])
    all_scores = np.append(all_scores, odd_val['ad_score'])

    all_weights = np.append(all_w, even_val['weights'])
    all_weights = np.append(all_weights, odd_val['weights'])
    
    all_sigs = np.append(even_sig['ad_score'],odd_sig['ad_score'])
    all_sig_w = np.append(even_sig['weights'], odd_sig['weights'])
    all_sig_samples = np.append(even_sig['samples'], odd_sig['samples'])
    all_sig_inds = np.append(even_sig['index'], odd_sig['index'])
    
    all_inds = np.append(even_data['index'], odd_data['index'])
    all_inds = np.append(all_inds, even_val['index'])
    all_inds = np.append(all_inds, odd_val['index'])
    
    output = {}
    
    output['all_scores'] = all_scores
    output['all_weights'] = all_weights
    output['all_sigs'] = all_sigs
    output['all_sig_w'] = all_sig_w
    output['all_sig_samples'] = all_sig_samples
    output['all_inds'] = all_inds
    output['all_sig_inds'] = all_sig_inds
    
    return output
    
    
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


    
# Get dataset as normal

# Create cumulative background distributions 

# Calculate MC stat error distributions 

#We want to plot the nbkg vs the mc stat error 

#We want to plot the mc stat error vs ad score

#We want to plot the excluded cross section

#We want to scan from right to left and find the first point
#where the MC stat error is the threshold, AND the number of bkg events > the threshold
#pick that bin
#then scan again and use the increase factor * the threshold 


# Perform binning algorithm
# args:
# - mc stat upper bound
# - bkg lower bound
# - increase factor per bin
# - fit the excluded cross section per bin
# - whether to use toys or not
    

if __name__ == '__main__': 
    
    run_region = [
        '0Z_0b_0SFOS',
        '0Z_0b_1SFOS',
        '0Z_0b_2SFOS',
        '1Z_0b_1SFOS',
        '1Z_0b_2SFOS',
        '2Z_0b',
        '0Z_1b_0SFOS',
        '0Z_1b_1SFOS',
        '0Z_1b_2SFOS',
        '1Z_1b_1SFOS',
        '1Z_1b_2SFOS',
        '2Z_1b'
    ]
    
    '''
    run_region = [
        'Q2_0b_e',
        'Q2_0b_eu',
        'Q2_0b_u',
        'Q2_1b_e',
        'Q2_1b_eu',
        'Q2_1b_u',
    ]
    '''

    
    #run_region = ['Q2_0b_e']
    
    #run_file = 'evaluate/region_settings/nf_NewYields.yaml'
    run_file = 'evaluate/region_settings/nf_Q2.yaml'
    output_folder = 'binning/outputs/Q2'
    
    make_plots = True
    
    with open(run_file, 'r') as f:
        r_config = yaml.safe_load(f)
    
        
    flavour_split_regions = {
       '0Z_0b_1SFOS' : ['0Z_0b_1SFOS_EgtM', '0Z_0b_1SFOS_MgtE'],
       '0Z_0b_2SFOS' : ['0Z_0b_2SFOS_EgtM', '0Z_0b_2SFOS_EeqM', '0Z_0b_2SFOS_MgtE'],
       '1Z_0b_2SFOS' : ['1Z_0b_2SFOS_EgtM', '1Z_0b_2SFOS_MgetE'],
       '0Z_1b_1SFOS' : ['0Z_1b_1SFOS_EgtM', '0Z_1b_1SFOS_MgtE'],
       '0Z_1b_2SFOS' : ['0Z_1b_2SFOS_EgtM', '0Z_1b_2SFOS_EeqM', '0Z_1b_2SFOS_MgtE'],
       '1Z_1b_2SFOS' : ['1Z_1b_2SFOS_EgtM', '1Z_1b_2SFOS_MgetE'],
    }
    
    flavour_split_regions = {}
    
    for region in run_region:
        if region not in flavour_split_regions.keys():
            flavour_split_regions[region] = [region]
    
    
    for region, flavour_regions in flavour_split_regions.items():
        
        print(f"Running region: {region}")
        out_dir = os.path.join(output_folder, region)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
    
        dataset = get_dataset(region, r_config)
          
          
        even_dir = os.path.join(r_config['even_base_dir'],r_config['regions'][region]['even_path'])
        odd_dir = os.path.join(r_config['odd_base_dir'],r_config['regions'][region]['odd_path'])
        
        min_scale, max_scale = get_data_scaling(even_dir, odd_dir)
        
        region_scores = dataset['all_scores']
        region_sigs = dataset['all_sigs']
        region_weights = dataset['all_weights']
        region_inds = dataset['all_inds']

        print("Length of indices: ", len(region_inds))
        print("Length of weights: ", len(region_weights))

        region_sig_w = dataset['all_sig_w']
        region_sig_inds = dataset['all_sig_inds']
        region_sig_samples = dataset['all_sig_samples']
        
        region_scores, _,_ = scale_log_prob(region_scores, min_prob=min_scale, max_prob=max_scale)
        region_sigs,_,_ = scale_log_prob(region_sigs, min_prob=min_scale, max_prob=max_scale)
        
        
        #Need to include overflows (wherever the score is bigger than max, set it to max)
        region_scores[region_scores > 1] = 0.9999
        region_sigs[region_sigs > 1] = 0.9999
        
        bin_min = 0
        bin_max = 1
        num_bins = 100
        
        ad_bins = np.linspace(bin_min, bin_max, num_bins)
        fine_ad_bins = np.linspace(bin_min, bin_max, 5000)
        fs = 16
        
        #Now split into flavour regions
        for sub_region in flavour_regions:

            if sub_region != region:

                out_dir = os.path.join(output_folder, region, sub_region)
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
            
                base_dir = '/data/at3/scratch2/multilepton/VLL_production/feather/4lepQ0'
                
                infile = os.path.join(base_dir,region+'_10GeV.ftr')
                
                #Join on the original feather file
                original_data = pd.read_feather(infile)
                new_data = pd.DataFrame()
                new_data['Scores'] = region_scores
                new_data['Weights'] = region_weights
                new_data['index'] = region_inds
                print("Scores length, pre merge: ",len(new_data))
                print("OG data length", len(original_data))
                merged_data = pd.merge(new_data, original_data, left_on='index', right_on='index', how='inner')
                print("Score length, post merge: ", len(merged_data))

                #####################################################################
                #Get the same for the signals
                sigfile = os.path.join(base_dir,region+'_Sigs_10GeV.ftr')
                og_sig_data = pd.read_feather(sigfile)
                new_sig_data = pd.DataFrame()
                new_sig_data['Scores'] = region_sigs
                new_sig_data['Weights'] = region_sig_w
                new_sig_data['index'] = region_sig_inds
                new_sig_data['Samples'] = region_sig_samples

                print("Sig length, pre merge: ",len(new_sig_data))

                merged_sig_data = pd.merge(new_sig_data, og_sig_data, on='index', how='inner')
                print("Sig length, post merge: ", len(merged_sig_data))


                if '0Z_0b_1SFOS' in sub_region or '0Z_1b_1SFOS' in sub_region:

                    if 'EgtM' in sub_region:
                        
                        #Select on quadlep_type == 4 
                        merged_data = merged_data.loc[merged_data['quadlep_type']==4]
                        merged_sig_data = merged_sig_data.loc[merged_sig_data['quadlep_type']==4]

                    elif 'MgtE' in sub_region:

                        merged_data = merged_data.loc[merged_data['quadlep_type']==2]
                        merged_sig_data = merged_sig_data.loc[merged_sig_data['quadlep_type']==2]

                elif '0Z_0b_2SFOS' in sub_region or '0Z_1b_2SFOS' in sub_region:

                    if 'EgtM' in sub_region:
                        merged_data = merged_data.loc[merged_data['quadlep_type']==5]
                        merged_sig_data = merged_sig_data.loc[merged_sig_data['quadlep_type']==5]

                    elif 'EeqM' in sub_region:
                        merged_data = merged_data.loc[merged_data['quadlep_type']==3]
                        merged_sig_data = merged_sig_data.loc[merged_sig_data['quadlep_type']==3]

                    elif 'MgtE' in sub_region:
                        merged_data = merged_data.loc[merged_data['quadlep_type']==1]
                        merged_sig_data = merged_sig_data.loc[merged_sig_data['quadlep_type']==1]

                elif '1Z_0b_2SFOS' in sub_region or  '1Z_1b_2SFOS' in sub_region:

                    #Find the Z type
                    pairings = {'01':'23', '02':'13', '03':'12', '12':'03', '13':'02','23':'01'}
                    for leps, pair in pairings.items():

                        merged_data.loc[(abs(merged_data[f"Mll{pair}"])-91.2e3<10e3)&
                                        (merged_data[f"lep_ID_{leps[0]}"]==-merged_data[f"lep_ID_{leps[1]}"])&
                                        (abs(merged_data[f"lep_ID_{leps[0]}"])==11),'Ztype'] = 'E'

                        merged_data.loc[(abs(merged_data[f"Mll{pair}"])-91.2e3<10e3)&
                                        (merged_data[f"lep_ID_{leps[0]}"]==-merged_data[f"lep_ID_{leps[1]}"])&
                                        (abs(merged_data[f"lep_ID_{leps[0]}"])==13),'Ztype'] = 'M'


                        merged_sig_data.loc[(abs(merged_sig_data[f"Mll{pair}"])-91.2e3<10e3)&
                                        (merged_sig_data[f"lep_ID_{leps[0]}"]==-merged_sig_data[f"lep_ID_{leps[1]}"])&
                                        (abs(merged_sig_data[f"lep_ID_{leps[0]}"])==11),'Ztype'] = 'E'

                        merged_sig_data.loc[(abs(merged_sig_data[f"Mll{pair}"])-91.2e3<10e3)&
                                        (merged_sig_data[f"lep_ID_{leps[0]}"]==-merged_sig_data[f"lep_ID_{leps[1]}"])&
                                        (abs(merged_sig_data[f"lep_ID_{leps[0]}"])==13),'Ztype'] = 'M'


                    if 'EgtM' in sub_region:
                        #quadlep_type = 5
                        merged_data = merged_data.loc[(merged_data['quadlep_type'] == 5)|((merged_data['Ztype'] == 'M')&(merged_data['quadlep_type'] == 3))]
                        merged_sig_data = merged_sig_data.loc[(merged_sig_data['quadlep_type'] == 5)|((merged_sig_data['Ztype'] == 'M')&(merged_sig_data['quadlep_type'] == 3))]



                    elif 'MgtE' in sub_region:
                        merged_data = merged_data.loc[(merged_data['quadlep_type'] == 1)|((merged_data['Ztype'] == 'E')&(merged_data['quadlep_type'] == 3))]
                        merged_sig_data = merged_sig_data.loc[(merged_sig_data['quadlep_type'] == 1)|((merged_sig_data['Ztype'] == 'E')&(merged_sig_data['quadlep_type'] == 3))]



                all_scores = merged_data['Scores']
                all_weights = merged_data['Weights']


                all_sigs = merged_sig_data['Scores']
                all_sig_w = merged_sig_data['Weights']
                all_sig_samples = merged_sig_data['Samples']


                print("Score length, post selection: ", len(merged_data)) 
            
            else:
                
                all_scores = region_scores
                all_weights = region_weights
                
                all_sigs = region_sigs
                all_sig_w = region_sig_w
                all_sig_samples = region_sig_samples
                
                
        
            if make_plots:
                #Make the standard anomaly score plot
                plt.figure()
                counts, edges, _ = plt.hist(all_scores, bins=ad_bins, weights=all_weights, alpha=0.8)
                plt.ylabel('Counts', fontsize=fs)
                plt.xlabel('Anomaly score', fontsize=fs)
                #plt.title(region)
                #plt.legend(title=r)
                plt.ylim(bottom=0)
                plt.savefig(os.path.join(out_dir, 'AD_score_dist.png'), dpi=300)
                plt.savefig(os.path.join(out_dir, 'AD_score_dist.pdf'), dpi=300)
                plt.close()

                #Plot it in log y 
                plt.figure()
                counts, edges, _ = plt.hist(all_scores, bins=ad_bins, weights=all_weights, alpha=0.8)
                plt.ylabel('Counts',fontsize=fs)
                plt.xlabel('Anomaly score', fontsize=fs)
                #plt.title(region)
                plt.yscale('log')
                #plt.legend(title=r)
                plt.savefig(os.path.join(out_dir, 'Log_AD_score_dist.png'),dpi=300)
                plt.savefig(os.path.join(out_dir, 'Log_AD_score_dist.pdf'),dpi=300)
                plt.close()
            
            
            #Make cumulative distributions 
            yields_sqd, bin_edges = np.histogram(all_scores, bins=fine_ad_bins, weights=all_weights**2)
            yields, bin_edges_all = np.histogram(all_scores, bins=fine_ad_bins, weights=all_weights)

            #Make a cum sum from right to left:
            even_cumsum = np.flip(np.cumsum(np.flip(yields)))  #Use this as the new AD score dist.
            even_cumsum_sqd = np.flip(np.cumsum(np.flip(yields_sqd)))
            
            if make_plots:
                plt.figure()
                plt.hist(bin_edges_all[:-1], bin_edges_all, weights=even_cumsum, alpha=0.8)
                plt.xlabel('Anomaly score')
                plt.ylabel(r'$Cum. sum.$')
                plt.title(region)
                plt.savefig(os.path.join(out_dir, 'cum_sum_plot.png'))
                plt.close()

                plt.figure()
                plt.hist(bin_edges_all[:-1], bin_edges_all, weights=even_cumsum, alpha=0.8)
                plt.xlabel('Anomaly score')
                plt.ylabel(r'$Cum. sum.$')
                plt.title(region)
                plt.yscale('log')
                plt.savefig(os.path.join(out_dir, 'cum_sum_LOGplot.png'))
                plt.close()

                plt.figure()
                plt.hist(bin_edges_all[:-1], bin_edges_all, weights=even_cumsum_sqd, alpha=0.8, label='Even')
                #plt.hist(even_scores, bins=ad_bins,alpha=0.8, label='Even', weights=even_data['weights'])
                plt.xlabel('Anomaly score')
                plt.ylabel(r'$Cum. sum.^2$')
                plt.title(region)
                plt.savefig(os.path.join(out_dir, 'cum_sum_Sqd_plot.png'))
                plt.close()

                plt.figure()
                plt.hist(bin_edges_all[:-1], bin_edges_all, weights=even_cumsum_sqd, alpha=0.8, label='Even')
                #plt.hist(even_scores, bins=ad_bins,alpha=0.8, label='Even', weights=even_data['weights'])
                plt.xlabel('Anomaly score')
                plt.ylabel(r'$Cum. sum.^2$')
                plt.title(region)
                plt.yscale('log')
                plt.savefig(os.path.join(out_dir, 'cum_sum_Sqd_LOG_plot.png'))
                plt.close()
            
            
            #Make plots of AD score vs MC stat error
            stat_err = np.sqrt(even_cumsum_sqd)
            frac_err = np.sqrt(even_cumsum_sqd) / abs(even_cumsum)

            frac_err[np.isnan(frac_err)] = 0
            frac_err[np.isinf(frac_err)] = 0
            
            if make_plots:
                plt.figure()
                plt.plot(bin_edges_all[:-1], frac_err)
                plt.xlabel('Anomaly score')
                plt.ylabel(r'$\sigma_{w}$ / $\sum{w}$')
                plt.title(region)
                plt.savefig(os.path.join(out_dir, 'frac_error_ADscore.png'))
                plt.close()


                #Make overlay of error and AD score
                fig,ax = plt.subplots()
                ax.hist(all_scores, bins=fine_ad_bins, alpha=0.8, weights=all_weights)
                ax.set_xlabel("Anomaly score",fontsize=fs)
                ax.set_ylabel("Counts",fontsize=fs)

                # twin object for two different y-axis on the sample plot
                ax2=ax.twinx()
                plt.plot(bin_edges_all[:-1], frac_err, color="red",alpha=0.8)
                ax2.set_ylabel(r'$\sigma_{w}$ / $\sum{w}$',fontsize=fs)

                plt.title(region)
                plt.savefig(os.path.join(out_dir, 'frac_error_Overlay_ADscore.png'))
                plt.close()

            
                #Make overlay of error and AD score
                fig,ax = plt.subplots()
                ax.hist(all_scores, bins=ad_bins, alpha=0.8, weights=all_weights)
                ax.set_xlabel("Anomaly score", fontsize=fs)
                ax.set_ylabel("Counts", fontsize=fs)
                plt.yscale('log')

                # twin object for two different y-axis on the sample plot
                ax2=ax.twinx()
                plt.plot(bin_edges_all[:-1], frac_err, color="red",alpha=0.8)
                ax2.set_ylabel(r'$\sigma_{w}$ / $\sum{w}$', fontsize=15.5)
                #plt.title(region)
                plt.savefig(os.path.join(out_dir, 'frac_error_OverlayLOG_ADscore.png'),dpi=300)
                plt.savefig(os.path.join(out_dir, 'frac_error_OverlayLOG_ADscore.pdf'),dpi=300)
                plt.close()
            
            
                #Make plots of nbkg vs AD score
                plt.figure()
                plt.plot(even_cumsum, frac_err)
                plt.xlabel(r'$N_{bkg}$')
                plt.ylabel(r'$\sigma_{w}$ / $\sum{w}$')
                plt.title(region)
                plt.savefig(os.path.join(out_dir, 'frac_error_NBKG.png'))
                plt.close()
            
                #Make plots of nbkg vs AD score
                #Plot only up to 5% mc stat error
                ind = np.argwhere(frac_err > 0.05)[0]
                if len(ind) == 0:
                    ind = -1
                ad_val = even_cumsum[ind]

                plt.figure()
                plt.plot(even_cumsum, frac_err)
                plt.xlabel(r'$N_{bkg}$')
                plt.xlim(0,ad_val)
                plt.ylabel(r'$\sigma_{w}$ / $\sum{w}$')
                plt.title(region)
                plt.savefig(os.path.join(out_dir, 'frac_error_NBKG_Trimmed.png'))
                plt.close()
            
            
            #Run binning algorithm
            
            #Can I do this without using binned histograms?
            
            #Sort the events into an ascending list of AD scores
            #Sort the weights equally
            #Scan along the list until the sum is 0.1bkg or 20% stat error
            
            bkg_scale_factors = [4]

            for bkg_scale_factor in bkg_scale_factors:
                
                og_bkg_events = 0.1
                min_bkg_events = 0.1
                min_stat_error = 0.2
                
                #Find the first place where the bins reach the min bkg events
                print(f"Running sf: {bkg_scale_factor}")
                
                alg_dir = os.path.join(out_dir, str(bkg_scale_factor))
                if not os.path.exists(alg_dir):
                    os.makedirs(alg_dir)


                    
                #Run from left to right and find the first point where we hit it
                #Then run right to left again and find the first point where we hit it * sf
                
                first_bkg_point = np.argwhere(even_cumsum < min_bkg_events)
                
                if len(first_bkg_point) == 0:  # ie. never less than min number of events
                    first_bkg_point = len(even_cumsum) #Only use the stat error 
                
                else:
                    first_bkg_point = first_bkg_point[0][-1]-1 #Otherwise take the furthest right point
                    
                    
                    
                print(first_bkg_point)
                print("HERE")
                print("index : ", first_bkg_point)
                print("yield : ", even_cumsum[first_bkg_point])
                print("AD point: ",fine_ad_bins[first_bkg_point])
                
                alg_bins = [1]
                bin_yields = []
                
  

                delta_frac_err = frac_err - min_stat_error
                signs = np.sign(delta_frac_err)
                diffs = np.diff(signs)
                inds = np.where(diffs != 0) #Where there's a change in sign
                if len(inds) == 0: #Ie. either always above or always below 
                    if sum(signs) < 0: #Always below
                        first_stat_error = len(even_cumsum)  #Only use bkg
                    else:  #Always above 
                        first_stat_error = 0   #Never use bkg
                else:
                    first_stat_error = inds[0][0]
                
                
                '''Old way 
                #For the error, take the left-most point where the error is < min error
                err_arr = np.argwhere(frac_err >= min_stat_error)
                if len(err_arr) == 0: #ie. error is always below min
                    first_stat_error = len(even_cumsum)  #Set to max so that min bkg is always lower         
                elif err_arr[0] == 0: #ie. 
                
                else:
                    first_stat_error = err_arr[0]-1
                    
                for i in range(len(err_arr)):
                    print(err_arr[i], frac_err[err_arr[i]])
                '''

                print("first stat ind: ", first_stat_error)
                print("first stat err: ", frac_err[first_stat_error])
                print("Yield at first stat error", even_cumsum[first_stat_error])
                print("AD score at first stat error",fine_ad_bins[first_stat_error] )
                    
                first_bin_pos = min(first_bkg_point, first_stat_error)
                first_ad_score = fine_ad_bins[first_bin_pos]
                alg_bins.append(first_ad_score)

                min_bkg_events = even_cumsum[first_bin_pos]
                bin_yields.append(float(even_cumsum[first_bin_pos]))
                
                prev_bin_pos = first_bin_pos

                
                
                i=0
                print("Before while loop: ", min_bkg_events, " , ", even_cumsum[0], " , " ,prev_bin_pos)
                while(min_bkg_events < even_cumsum[0] and prev_bin_pos != 0):
                    
                    print("==== next loop ===")
                    
                    print(min_bkg_events, " , ", bkg_scale_factor, " , ")
                    #Get an updated number of background events
                    min_bkg_events = bkg_scale_factor * min_bkg_events
                    #print(f"new min bkg: ", min_bkg_events)
                    
                    
                    #Find the points where the bin 
                    first_bkg_point = np.argwhere(even_cumsum-even_cumsum[prev_bin_pos] < min_bkg_events)
                    
                    #Don't need to keep recalculating the stat error place
                   
                    
                    if len(first_bkg_point) == len(even_cumsum):  #If the yield is never less than the bkg events
                        print("Setting to point 0")
                        first_bkg_point = 0#len(even_cumsum)
                    else:
                        first_bkg_point = first_bkg_point[0] - 1
                    

                    

                    first_bin_pos = min(first_bkg_point, first_stat_error)
                    first_ad_score = fine_ad_bins[first_bin_pos]
                    
                    
                    print("First bkg: ", first_bkg_point, " , first stat error: ", first_stat_error)
                    print("New index : ", first_bin_pos)
                    print("New ad score val : ", first_ad_score)
                    print("Next bin yield : ", even_cumsum[first_bin_pos] - even_cumsum[prev_bin_pos])
                    
                    if(first_ad_score != alg_bins[-1]):
                        alg_bins.append(float(first_ad_score))
                    else:
                        print(f"Found same limit with bkg events: {min_bkg_events}")
                    
                    print("Previous bin pos: " ,prev_bin_pos)
                    
                    min_bkg_events = even_cumsum[first_bin_pos] - even_cumsum[prev_bin_pos]
                    bin_yields.append(float(even_cumsum[first_bin_pos]) - float(even_cumsum[prev_bin_pos]))
                    prev_bin_pos = first_bin_pos
                    
                    
                    print("New min_bkg_events: " , min_bkg_events)
                    print("Current bins: ", alg_bins)
                    i+=1
                    #if i >14:
                    #    break
                    
                if 0 not in alg_bins:
                    alg_bins.append(0)
                    
                alg_bins.reverse()
                bin_yields.reverse()
                print("Bin yields: ", bin_yields)
                make_plots=True

                if make_plots:
                    #Make the standard anomaly score plot
                    plt.figure()
                    counts, edges, _ = plt.hist(all_scores, bins=alg_bins, weights=all_weights, alpha=0.8)
                    plt.ylabel('Counts')
                    plt.xlabel('Anomaly score')
                    plt.title(f"{region}: min bkg={og_bkg_events}, scale={bkg_scale_factor}")
                    #plt.legend(title=r)
                    plt.ylim(bottom=0)
                    plt.savefig(os.path.join(alg_dir, f"AD_score_BINALG_{bkg_scale_factor}.png"))
                    plt.close()

                    plt.figure()
                    counts, edges, _ = plt.hist(all_scores, bins=alg_bins, weights=all_weights, alpha=0.8)
                    plt.ylabel('Counts')
                    plt.xlabel('Anomaly score')
                    plt.title(f"{region}: min bkg={og_bkg_events}, scale={bkg_scale_factor}")
                    plt.yscale('log')
                    plt.ylim(bottom=0)
                    plt.savefig(os.path.join(alg_dir, f"Log_AD_score_BINALG_{bkg_scale_factor}.png"))
                    plt.close()
                    
                str_alg_bins = ['%s' % float('%.5g' % a) for a in alg_bins]
                comb_str = ','.join(str_alg_bins)

                bin_str = f"Binning: {comb_str}\n"
                print(bin_str)
                
                str_yield_bins = ['%s' % float('%.5g' % a) for a in bin_yields]
                y_str = ','.join(str_yield_bins)
                yield_str = f"Bin yields: {y_str}\n"
                print(yield_str)
                with open(os.path.join(alg_dir, f"{region}_OutputBins.txt"), 'w') as f:
                    f.writelines([bin_str, yield_str])

                #Inject low mass, middle mass, high mass signals
                chosen_sigs = ['Edoublet300', 'Edoublet700', 'Edoublet1200',
                            'VLLe600S350', 'VLLe1200S750', 'Esinglet150']
                sig_names = ['E(D,300)', 'E(D,700)', 'E(D,1200)', 'E(600,S350)','E(1200,S750)','E(S,150)']
                norm_total_bkg = True

                for i, sig in enumerate(chosen_sigs):
                    #Get the signal events
                    inds = np.where(np.isin(all_sig_samples, [sig]))[0]

                    print("samples: ", len(all_sig_samples))
                    print("all_sigs: ", len(all_sigs))
                    print("All_sig_w: ", len(all_sig_w))
                    
                    sig_scores, sig_weights = np.array(all_sigs)[inds], np.array(all_sig_w)[inds]

                    density=True
                    if norm_total_bkg:
                        sig_weights = sig_weights / sum(sig_weights)
                        sig_weights = sig_weights * sum(all_weights)
                        density = False

                    #Make it into a stackplot
                    plt.figure()
                    plt.hist(all_scores, bins=alg_bins, weights=all_weights, alpha=0.5, label='Background')
                    plt.hist(sig_scores, bins=alg_bins, weights=sig_weights, alpha=0.5, label=sig_names[i])
                    plt.ylabel('Counts')
                    plt.xlabel('Anomaly score')
                    #plt.title(f"{region}: min bkg={og_bkg_events}, scale={bkg_scale_factor}")
                    plt.legend()
                    plt.ylim(bottom=0)
                    plt.savefig(os.path.join(alg_dir, f"AD_score_BINALG_{bkg_scale_factor}_{sig}.png"))
                    plt.close()

                    plt.figure()
                    plt.hist(all_scores, bins=alg_bins, weights=all_weights, alpha=0.5, label='Background')
                    plt.hist(sig_scores, bins=alg_bins, weights=sig_weights, alpha=0.5, label=sig_names[i])
                    plt.ylabel('Counts')
                    plt.xlabel('Anomaly score')
                    #plt.title(f"{region}: min bkg={og_bkg_events}, scale={bkg_scale_factor}")
                    plt.legend()
                    plt.yscale('log')
                    plt.savefig(os.path.join(alg_dir, f"AD_score_BINALG_LOG_{bkg_scale_factor}_{sig}.png"))
                    plt.close()

                    plt.figure()
                    groups = [all_scores,sig_scores]
                    group_names =['Background',sig_names[i]]
                    group_weights = [all_weights, sig_weights]

                    plt.hist(groups, bins=alg_bins,alpha=0.8, density=density, stacked=True, label=group_names, weights=group_weights)
                    plt.ylabel('Counts', fontsize=fs)
                    plt.xlabel('Anomaly score',fontsize=fs)
                    #plt.title(f"{region}: min bkg={og_bkg_events}, scale={bkg_scale_factor}")
                    plt.legend()
                    plt.yscale('log')
                    plt.savefig(os.path.join(alg_dir, f"Stacked_AD_score_BINALG_LOG_{bkg_scale_factor}_{sig}.png"))
                    plt.savefig(os.path.join(alg_dir, f"Stacked_AD_score_BINALG_LOG_{bkg_scale_factor}_{sig}.pdf"))
                    plt.close()
                    
                    plt.hist(groups, bins=ad_bins,alpha=0.8, density=density, stacked=True, label=group_names, weights=group_weights)
                    plt.ylabel('Counts')
                    plt.xlabel('Anomaly score')
                    #plt.title(f"{region}: min bkg={og_bkg_events}, scale={bkg_scale_factor}")
                    plt.legend()
                    plt.yscale('log')
                    plt.savefig(os.path.join(alg_dir, f"Stacked_AD_score_OGBIN_LOG_{bkg_scale_factor}_{sig}.png"))
                    plt.savefig(os.path.join(alg_dir, f"Stacked_AD_score_OGBIN_LOG_{bkg_scale_factor}_{sig}.pdf"))
                    plt.close()
                    
                    
                    

                    #Get the excluded cross section from each bin ? 
                




            
        