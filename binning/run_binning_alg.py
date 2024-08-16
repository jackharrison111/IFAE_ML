'''
run_binning_alg.py

Script to control the binning algorthim workflow.
This should load the input data as required, and
then call the binning alg function, and make the 
output plots required.


'''


import yaml
import os
from binning.utils.data_retriever import get_dataset, get_data_scaling, scale_log_prob

from binning.algs.model_independent import run_binning_alg

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

plt.style.use(hep.style.ATLAS)

if __name__ == '__main__':

    #Load the training runs 

    trainrun_file = 'evaluate/region_settings/nf_NewYields.yaml'
    #trainrun_file = 'evaluate/region_settings/nf_Q2.yaml'
    #output_folder = 'binning/outputs/modelIndepQ2'
    output_folder = 'binning/outputs/ModelDepQ2'
    make_plots = False


    #Set the regions to run over
    with open(trainrun_file, 'r') as f:
        r_config = yaml.safe_load(f)

    chosen_regions = [
        '0Z_0b_0SFOS',
        '0Z_0b_1SFOS',
        '0Z_0b_2SFOS',
        '1Z_0b_1SFOS',
        '1Z_0b_2SFOS',
        '2Z_0b'
        ]
    
    #chosen_regions  = [
    #    'Q2_0b_e'
    #]
    
    
    region_names ={
        '0Z_0b_0SFOS':'0Z 0SFOS',
        '0Z_0b_1SFOS':'0Z 1SFOS',
        '0Z_0b_2SFOS':'0Z 2SFOS',
        '1Z_0b_1SFOS':'1Z 1SFOS',
        '1Z_0b_2SFOS':'1Z 2SFOS',
        '2Z_0b':'2Z',
        'Q2_0b' : 'Q2',
    }

    #Get the data

    for region in chosen_regions:

        print(f"Running region: {region}")
        out_dir = os.path.join(output_folder, region)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        dataset = get_dataset(region, r_config, old_name=True, use_val=False)
        even_dir = os.path.join(r_config['even_base_dir'],r_config['regions'][region]['even_path'])
        odd_dir = os.path.join(r_config['odd_base_dir'],r_config['regions'][region]['odd_path'])
        min_scale, max_scale = get_data_scaling(even_dir, odd_dir)
        region_scores, _,_ = scale_log_prob(dataset['all_scores'], min_prob=min_scale, max_prob=max_scale)
        
        region_sigs,_,_ = scale_log_prob(dataset['all_sigs'], min_prob=min_scale, max_prob=max_scale)

        #Repeat for 1b regions
        region_1b = region.replace('0b', '1b')
        dataset_1b = get_dataset(region_1b, r_config, old_name=True)
        even_dir_1b = os.path.join(r_config['even_base_dir'],r_config['regions'][region_1b]['even_path'])
        odd_dir_1b = os.path.join(r_config['odd_base_dir'],r_config['regions'][region_1b]['odd_path'])
        min_scale, max_scale = get_data_scaling(even_dir_1b, odd_dir_1b)
        region_scores_1b, _,_ = scale_log_prob(dataset_1b['all_scores'], min_prob=min_scale, max_prob=max_scale)
        region_sigs_1b,_,_ = scale_log_prob(dataset_1b['all_sigs'], min_prob=min_scale, max_prob=max_scale)

        #Call binning alg
        print("Sum of weights before input to func: ", sum(dataset['all_weights']))
        output = run_binning_alg(region_scores, region_scores_1b, dataset['all_weights'], dataset_1b['all_weights'])


        all_scores = np.append(region_scores, region_scores_1b)
        all_weights = np.append(dataset['all_weights'], dataset_1b['all_weights'])

        fs=16
        ad_bins = np.linspace(0,1,100)

        #Make overlay of the two 0b and 1b histograms
        plt.figure()
        counts, edges, _ = plt.hist(region_scores, bins=ad_bins, weights= dataset['all_weights'], alpha=0.8, label='0b')
        counts, edges, _ = plt.hist(region_scores_1b, bins=ad_bins, weights= dataset_1b['all_weights'], alpha=0.8, label=r'$\geq$1b')
        plt.ylabel('Counts', fontsize=fs)
        plt.xlabel('Anomaly score', fontsize=fs)
        plt.ylim(bottom=0)
        plt.legend(title=region_names[region])
        plt.savefig(os.path.join(out_dir, 'AD_score_dist_OVERLAY.png'), dpi=300)
        plt.savefig(os.path.join(out_dir, 'AD_score_dist_OVERLAY.pdf'), dpi=300)
        plt.close()

        plt.figure()
        counts, edges, _ = plt.hist(region_scores, bins=ad_bins, weights= dataset['all_weights'], alpha=0.8, label='0b', density=True)
        counts, edges, _ = plt.hist(region_scores_1b, bins=ad_bins, weights= dataset_1b['all_weights'], alpha=0.8, label=r'$\geq$1b', density=True)
        plt.ylabel('Counts', fontsize=fs)
        plt.xlabel('Anomaly score', fontsize=fs)
        plt.ylim(bottom=0)
        plt.legend(title=region_names[region])
        plt.savefig(os.path.join(out_dir, 'AD_score_dist_NORMED.png'), dpi=300)
        plt.savefig(os.path.join(out_dir, 'AD_score_dist_NORMED.pdf'), dpi=300)
        plt.close()
        

        #Make output plots of signal and background with the new binning
        plt.figure()
        counts, edges, _ = plt.hist(all_scores, 
                                    bins=output['binning'], 
                                    weights=all_weights, alpha=0.8, label='Bkg')
        plt.ylabel('Counts', fontsize=fs)
        plt.xlabel('Anomaly score', fontsize=fs)
        #plt.title(region)
        plt.legend(title=region_names[region])
        plt.ylim(bottom=0)
        plt.savefig(os.path.join(out_dir, 'Combined_scores_NewBins.png'), dpi=300)
        plt.savefig(os.path.join(out_dir, 'Combined_scores_NewBins.pdf'), dpi=300)
        plt.close()

        plt.figure()
        counts, edges, _ = plt.hist(all_scores, 
                                    bins=output['binning'], 
                                    weights=all_weights, alpha=0.8, label='Bkg')
        plt.ylabel('Counts', fontsize=fs)
        plt.xlabel('Anomaly score', fontsize=fs)
        plt.yscale('log')
        #plt.title(region)
        plt.legend(title=region_names[region])
        plt.ylim(bottom=0)
        plt.savefig(os.path.join(out_dir, 'Combined_scores_NewBins_LOG.png'), dpi=300)
        plt.savefig(os.path.join(out_dir, 'Combined_scores_NewBins_LOG.pdf'), dpi=300)
        plt.close()


        #Plot with signal stacked

        str_alg_bins = ['%s' % float('%.5g' % a) for a in output['binning']]
        comb_str = ','.join(str_alg_bins)

        bin_str = f"Binning: {comb_str}\n"
        print(bin_str)

        str_yield_bins = ['%s' % float('%.5g' % a) for a in output['bin_yields']]
        y_str = ','.join(str_yield_bins)
        yield_str = f"Bin yields: {y_str}\n"
        print(yield_str)
        with open(os.path.join(out_dir,'binning.txt'), 'w') as f:
            f.writelines([bin_str, yield_str])





    ...