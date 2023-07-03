import pickle
import os
import yaml
import numpy as np 
import matplotlib.pyplot as plt 



def get_dataset(region_choice, r_config):
    
    even_dir = os.path.join(r_config['even_base_dir'],r_config['regions'][region_choice]['even_path'])
    odd_dir = os.path.join(r_config['odd_base_dir'],r_config['regions'][region_choice]['odd_path'])
    
    with open(os.path.join(even_dir, 'new_bkg_outputs.pkl'), 'rb') as f:
            even_data = pickle.load(f)

    with open(os.path.join(even_dir, 'saved_val_outputs.pkl'), 'rb') as f:
        even_val = pickle.load(f)

    with open(os.path.join(odd_dir, 'new_bkg_outputs.pkl'), 'rb') as f:
        odd_data = pickle.load(f)

    with open(os.path.join(odd_dir, 'saved_val_outputs.pkl'), 'rb') as f:
        odd_val = pickle.load(f)

    with open(os.path.join(even_dir, 'saved_signal_outputs.pkl'), 'rb') as f:
        even_sig = pickle.load(f)

    with open(os.path.join(odd_dir, 'saved_signal_outputs.pkl'), 'rb') as f:
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
    
    return all_scores, all_weights, all_sigs, all_sig_w, all_sig_samples
    
    
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
    scale_file = 'NF_new_likelihood_v2.txt'
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
def run_binning_alg():
    
    ...
    
    
    

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
    
    run_region = ['2Z_1b']
    
    run_file = 'evaluate/region_settings/nf_FinalRun.yaml'
    output_folder = 'binning/outputs'
    
    make_plots = False
    
    with open(run_file, 'r') as f:
        r_config = yaml.safe_load(f)
    
    
    for region in run_region:
        
        print(f"Running region: {region}")
        out_dir = os.path.join(output_folder, region)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
    
        all_scores, all_weights, all_sigs, all_sig_w, all_sig_samples = get_dataset(region, r_config)
            
        even_dir = os.path.join(r_config['even_base_dir'],r_config['regions'][region]['even_path'])
        odd_dir = os.path.join(r_config['odd_base_dir'],r_config['regions'][region]['odd_path'])
        
        min_scale, max_scale = get_data_scaling(even_dir, odd_dir)
        
        all_scores, _,_ = scale_log_prob(all_scores, min_prob=min_scale, max_prob=max_scale)
        all_sigs,_,_ = scale_log_prob(all_sigs, min_prob=min_scale, max_prob=max_scale)
        
        #Need to include overflows (wherever the score is bigger than max, set it to max)
        all_scores[all_scores > 1] = 0.9999
        all_sigs[all_sigs > 1] = 0.9999
        
        
        bin_min = 0
        bin_max = 1
        num_bins = 100
        
        ad_bins = np.linspace(bin_min, bin_max, num_bins)
        fine_ad_bins = np.linspace(bin_min, bin_max, 1000)
        
        if make_plots:
            #Make the standard anomaly score plot
            plt.figure()
            counts, edges, _ = plt.hist(all_scores, bins=ad_bins, weights=all_weights, alpha=0.8)
            plt.ylabel('Counts')
            plt.xlabel('Anomaly score')
            plt.title(region)
            #plt.legend(title=r)
            plt.ylim(bottom=0)
            plt.savefig(os.path.join(out_dir, 'AD_score_dist.png'))
            plt.close()

            #Plot it in log y 
            plt.figure()
            counts, edges, _ = plt.hist(all_scores, bins=ad_bins, weights=all_weights, alpha=0.8)
            plt.ylabel('Counts')
            plt.xlabel('Anomaly score')
            plt.title(region)
            plt.yscale('log')
            #plt.legend(title=r)
            plt.savefig(os.path.join(out_dir, 'Log_AD_score_dist.png'))
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
            ax.set_xlabel("Anomaly score")
            ax.set_ylabel("Counts")

            # twin object for two different y-axis on the sample plot
            ax2=ax.twinx()
            plt.plot(bin_edges_all[:-1], frac_err, color="red",alpha=0.8)
            ax2.set_ylabel(r'$\sigma_{w}$ / $\sum{w}$')

            plt.title(region)
            plt.savefig(os.path.join(out_dir, 'frac_error_Overlay_ADscore.png'))
            plt.close()

        
            #Make overlay of error and AD score
            fig,ax = plt.subplots()
            ax.hist(all_scores, bins=fine_ad_bins, alpha=0.8, weights=all_weights)
            ax.set_xlabel("Anomaly score")
            ax.set_ylabel("Counts")
            plt.yscale('log')

            # twin object for two different y-axis on the sample plot
            ax2=ax.twinx()
            plt.plot(bin_edges_all[:-1], frac_err, color="red",alpha=0.8)
            ax2.set_ylabel(r'$\sigma_{w}$ / $\sum{w}$')

            plt.title(region)
            plt.savefig(os.path.join(out_dir, 'frac_error_OverlayLOG_ADscore.png'))
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
        
        
        
        
        bkg_scale_factors = [2]

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
            
            print("HERE")
            print("index : ", first_bkg_point[0])
            print("yield : ", even_cumsum[first_bkg_point[0]])
            print("AD point: ",fine_ad_bins[first_bkg_point[0]])
            alg_bins = [1]

            
            if len(first_bkg_point) == 0:
                #Set it to the final bin
                print("Setting to point 0")
                first_bkg_point = 0
            else:
                first_bkg_point = first_bkg_point[0] - 1
                
            first_stat_error = np.argwhere(frac_err > min_stat_error)[0] - 1
            print("first stat ind: ", first_stat_error)
            print("first stat err: ", frac_err[first_stat_error])
                
            first_bin_pos = min(first_bkg_point, first_stat_error)
            first_ad_score = fine_ad_bins[first_bin_pos]
            alg_bins.append(first_ad_score[0])

            min_bkg_events = even_cumsum[first_bin_pos]

            print("min_bkg_events used", min_bkg_events)
            i=0
            while(min_bkg_events < even_cumsum[0]):

                print("==== next loop ===")
                min_bkg_events = bkg_scale_factor * min_bkg_events

                print(f"new min bkg: ", min_bkg_events)
                
                first_bkg_point = np.argwhere(even_cumsum < min_bkg_events)
                if len(first_bkg_point) == 0:
                    print("Setting to point 0")
                    first_bkg_point = 0
                else:
                    first_bkg_point = first_bkg_point[0] - 1

                

                first_stat_error = np.argwhere(frac_err > min_stat_error)[0] - 1

                first_bin_pos = min(first_bkg_point, first_stat_error)
                first_ad_score = fine_ad_bins[first_bin_pos]

                print("New index : ", first_bin_pos)
                print("New ad score val : ", first_ad_score)
                print("New bkg yield : ", even_cumsum[first_bin_pos])
                
                if(first_ad_score != alg_bins[-1]):
                    print(f"Using {min_bkg_events} events, got bin at: {first_ad_score}")
                    alg_bins.append(first_ad_score[0])
                else:
                    print(f"Found same limit with bkg events: {min_bkg_events}")
                    
                min_bkg_events = even_cumsum[first_bin_pos]
                print("New min_bkg_events: " , min_bkg_events)
                i+=1
                if i >2:
                    break

            alg_bins.reverse()
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
                
            str_alg_bins = ['%s' % float('%.3g' % a) for a in alg_bins]
            comb_str = ','.join(str_alg_bins)
            #Need to round to 3SF
            bin_str = f"Binning: {comb_str}\n"
            print(bin_str)
            with open(os.path.join(alg_dir, f"{region}_OutputBins.txt"), 'w') as f:
                f.writelines([bin_str])
            
            #Inject low mass, middle mass, high mass signals
            chosen_sigs = ['Edoublet300', 'Edoublet700', 'Edoublet1200',
                          'VLLe600S350', 'VLLe1200S750', 'Esinglet150']
            norm_total_bkg = True

            for sig in chosen_sigs:
                #Get the signal events
                inds = np.where(np.isin(all_sig_samples, [sig]))

                sig_scores, sig_weights = all_sigs[inds], all_sig_w[inds]

                density=True
                if norm_total_bkg:
                    sig_weights = sig_weights / sum(sig_weights)
                    sig_weights = sig_weights * sum(all_weights)
                    density = False

                #Make it into a stackplot
                plt.figure()
                plt.hist(all_scores, bins=alg_bins, weights=all_weights, alpha=0.5, label='Bkg')
                plt.hist(sig_scores, bins=alg_bins, weights=sig_weights, alpha=0.5, label=sig)
                plt.ylabel('Counts')
                plt.xlabel('Anomaly score')
                plt.title(f"{region}: min bkg={og_bkg_events}, scale={bkg_scale_factor}")
                plt.legend()
                plt.ylim(bottom=0)
                plt.savefig(os.path.join(alg_dir, f"AD_score_BINALG_{bkg_scale_factor}_{sig}.png"))
                plt.close()

                plt.figure()
                plt.hist(all_scores, bins=alg_bins, weights=all_weights, alpha=0.5, label='Bkg')
                plt.hist(sig_scores, bins=alg_bins, weights=sig_weights, alpha=0.5, label=sig)
                plt.ylabel('Counts')
                plt.xlabel('Anomaly score')
                plt.title(f"{region}: min bkg={og_bkg_events}, scale={bkg_scale_factor}")
                plt.legend()
                plt.yscale('log')
                plt.savefig(os.path.join(alg_dir, f"AD_score_BINALG_LOG_{bkg_scale_factor}_{sig}.png"))
                plt.close()

                plt.figure()
                groups = [all_scores,sig_scores]
                group_names =['Bkg',sig]
                group_weights = [all_weights, sig_weights]

                plt.hist(groups, bins=alg_bins,alpha=0.8, density=density, stacked=True, label=group_names, weights=group_weights)
                plt.ylabel('Counts')
                plt.xlabel('Anomaly score')
                plt.title(f"{region}: min bkg={og_bkg_events}, scale={bkg_scale_factor}")
                plt.legend()
                plt.yscale('log')
                plt.savefig(os.path.join(alg_dir, f"Stacked_AD_score_BINALG_LOG_{bkg_scale_factor}_{sig}.png"))
                plt.close()

                #Get the excluded cross section from each bin ? 
            




            
        