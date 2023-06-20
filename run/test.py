#############################################
#
# Class for running the testing pipeline
# of a VAE.
#
# Want to use this to evaluate signal +
# background. 
#
# May have to integrate with adding files
# back into ROOT ntuples
# 
# Jack Harrison 21/08/2022
#############################################


#---------------------------------------

#Test the model on new signal data
# - produce AD score
# - Save all relevant variables
# - Calculate separation from the Sig vs Bkg
# - need to separate the histogram production from the plotting


#---------------------------------------

import torch

import numpy as np
import os
import yaml
import math

#Plotting
from plotting.plot_results import Plotter
from tqdm import tqdm
import matplotlib.pyplot as plt

import pickle
import time
class Tester():
    
    def __init__(self, config=None):
        
        if type(config) == str:
            with open(config, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = config
            
        self.reversed_groupings = {}
        for key, values in self.config['groupings'].items():
            for val in values:
                self.reversed_groupings[val] = key
        ...
        
        
    def get_chi2distance(self, x,y):
        ch2 = np.nan_to_num(((x-y)**2)/(x+y), copy=True, nan=0.0, posinf=None, neginf=None)
        ch2 = 0.5 * np.sum(ch2)
        return ch2
    
    
    def evaluate(self, model, testloader):
        
        output = {}
        self.model = model
        
        with torch.no_grad():
            
            for idx, data_dict in enumerate(testloader):   
            
                data = data_dict['data']
                sample = data_dict['sample']
                #sc_weight = data_dict['scaled_weight']
                evtNumber = data_dict['eventNumber']
                weights = data_dict['weight']
                
                #Run results through model
                
                #Save loss and anomaly score?
                
                #Feed data into model
                outputs = model(data)
                
                #Get the loss
                losses = model.loss_function(**outputs)
                
                #For AE the AD score is the reco loss
                #For NF the AD score is the log prob (needs scaling over the whole of the input dataset)
                #Make an anomaly score
                anomaly_score = model.get_anomaly_score(losses)
                
                
                
                output.setdefault('data', []).extend(np.array(data))
                
                ae_results = ['reco', 'mu']
                for re in ae_results:
                    if re in outputs:
                        output.setdefault(re, []).extend(np.array(outputs[re]))
                
                output.setdefault('loss', []).extend(losses['loss'].tolist())
                output.setdefault('ad_score', []).extend(anomaly_score.tolist())
                output.setdefault('weights', []).extend(weights.tolist())
                #output.setdefault('log_loss', []).append(math.log(losses['loss'].item()))   #Don't do this as different models might fail
                output.setdefault('eventNumber', []).extend(evtNumber.tolist())
                
                
                #We don't want to reorder the samples, so just flatten them and append?
                output.setdefault('samples', []).extend(sample)
                
                #For the groups we just want to map each sample in the list to it's group and then store it:
                #group_mapped_samples = list(map(lambda x: self.reversed_groupings.get(x, x), sample))
                group_mapped_samples = [self.reversed_groupings.get(s, s) for s in sample] #Seems to be slightly faster
                output.setdefault('groups', []).extend(group_mapped_samples)
            
        #output['ad_score'] = np.array(output['ad_score'])
        for key, val in output.items():
            output[key] = np.array(val)
        return output
        
        
        
    def analyse_vae(self, bkg_output, sig_output=None, **kwargs):
        
        num_logloss_bins = 50
        logloss_bins = np.linspace(-10,10, num_logloss_bins)
        
        bkg_output['log_losses'] = np.log(bkg_output['loss'])
        if sig_output:
            sig_output['log_losses'] = np.log(sig_output['loss'])
            
        
        #Want plots of log anomaly scores
        #Want plots of all sigs vs background
        #
        
        per_sample_counts, per_sample_edges, per_sample_names = [], [], []
        test_logloss_counts, test_logloss_edges = None,None
        
        p = Plotter()
        
        #Do the same but plot background by group?
        if kwargs.get('trexfitter_plot'):
            
            #Set Chi2 and weights outputs
            chi2_out = []
            sum_weights = []
            sum_weights.append(f"Background: {sum(bkg_output['weights'])}\n")
            
            print("Making background plot...")
            groups = []
            group_names = []
            group_weights = []
            for group in set(bkg_output['groups']):
                inds = np.where(np.isin(bkg_output['groups'], [group]))[0]
                groups.append(bkg_output['log_losses'][inds])
                group_names.append(group)
                group_weights.append(bkg_output['weights'][inds])
                sum_weights.append(f"{group}: {sum(group_weights[-1])}\n")
                
            #Sort to plot the largest first
            group_sums = [sum(weights) for weights in group_weights]
            sorted_indices = np.argsort(group_sums)[::-1]
            groups = [groups[i] for i in sorted_indices]
            group_names = [group_names[i] for i in sorted_indices]
            group_weights = [group_weights[i] for i in sorted_indices]
            
            plt.figure()
            plt.hist(groups, bins=logloss_bins,alpha=0.8, density=1, stacked=True, label=group_names, weights=group_weights)
            plt.xlabel('Anomaly score')
            plt.ylabel('Counts')
            plt.legend()
            plt.savefig(os.path.join(self.out_dir, f"Bkg_distribution_normalised.png"))
            plt.savefig(os.path.join(self.out_dir, f"Bkg_distribution_normalised.pdf"))
            plt.close()

            plt.figure()
            plt.hist(groups, bins=logloss_bins,alpha=0.8, stacked=True, label=group_names, weights=group_weights)
            plt.xlabel('Anomaly score')
            plt.ylabel('Counts')
            plt.legend()
            plt.savefig(os.path.join(self.out_dir, f"Bkg_distribution.png"))
            plt.savefig(os.path.join(self.out_dir, f"Bkg_distribution.pdf"))
            plt.close()
            
            plt.figure()
            plt.hist(groups, bins=logloss_bins,alpha=0.8, stacked=True, label=group_names, weights=group_weights, log=True)
            plt.xlabel('Anomaly score')
            plt.ylabel('Counts')
            plt.legend()
            plt.savefig(os.path.join(self.out_dir, f"Bkg_distribution_log.png"))
            plt.savefig(os.path.join(self.out_dir, f"Bkg_distribution_log.pdf"))
            plt.close()
            
            if not os.path.exists(os.path.join(self.out_dir, 'sig_vs_bkg', 'log_plots')):
                os.makedirs(os.path.join(self.out_dir, 'sig_vs_bkg', 'log_plots'))
            
            if not os.path.exists(os.path.join(self.out_dir, 'sig_vs_bkg', 'separation_plots')):
                os.makedirs(os.path.join(self.out_dir, 'sig_vs_bkg','separation_plots'))
            
           
            
            for sig_sample in set(sig_output['groups']):
                print(f"Running: {sig_sample}")

                inds = np.where(np.isin(sig_output['groups'], [sig_sample]))[0]
                sig_l = np.array(sig_output['log_losses'])[inds]
                sig_w = np.array(sig_output['weights'])[inds]
                
                sum_weights.append(f"{sig_sample}: {sum(sig_w)}\n")
                
                #Calculate separations
                bkg_ad_counts, bkg_ad_edges = np.histogram(bkg_output['log_losses'], bins=logloss_bins, weights=bkg_output['weights'])
                
                sig_ad_counts, sig_ad_edges = np.histogram(sig_l, bins=logloss_bins, weights=sig_w)
                sep = self.get_chi2distance(bkg_ad_counts/np.sum(bkg_ad_counts), sig_ad_counts/np.sum(sig_ad_counts))
                
                out_str = f"{sig_sample}:{sep}\n"
                chi2_out.append(out_str)
                print(out_str)
                
                
                plt.figure()
                plt.hist(bkg_output['log_losses'], bins=logloss_bins, alpha=0.5, label='Bkg', density=True, weights=bkg_output['weights'])
                plt.hist(sig_l, bins=logloss_bins, alpha=0.5, label=sig_sample, density=True, weights=sig_w)
                plt.xlabel('Anomaly score')
                plt.ylabel('Counts')
                plt.legend()
                plt.savefig(os.path.join(self.out_dir, 'sig_vs_bkg', 'separation_plots', f"Sig_vs_bkg_{sig_sample}.png"))
                plt.savefig(os.path.join(self.out_dir,'sig_vs_bkg' ,'separation_plots', f"Sig_vs_bkg_{sig_sample}.pdf"))
                plt.close()
                
                #Get Chi2
                
                
                g_plots = groups.copy()
                g_plots.append(sig_l)
                g_weights = group_weights.copy()
                g_weights.append(sig_w)
                g_names = group_names.copy()
                g_names.append(sig_sample)
                
                plt.figure()
                plt.hist(g_plots, bins=logloss_bins,alpha=0.6, stacked=True, label=g_names, weights=g_weights, log=True)
                plt.xlabel('Anomaly score')
                plt.ylabel('Counts')
                plt.legend()
                plt.savefig(os.path.join(self.out_dir, 'sig_vs_bkg','log_plots', f"sig_vs_bkg_{sig_sample}_log.png"))
                plt.savefig(os.path.join(self.out_dir, 'sig_vs_bkg','log_plots', f"sig_vs_bkg_{sig_sample}_log.pdf"))
                plt.close()
                
                plt.figure()
                plt.hist(g_plots, bins=logloss_bins,alpha=0.6, stacked=True, label=g_names, weights=g_weights, density=True)
                plt.xlabel('Anomaly score')
                plt.ylabel('Counts')
                plt.legend()
                plt.savefig(os.path.join(self.out_dir, 'sig_vs_bkg', f"sig_vs_bkg_{sig_sample}.png"))
                plt.savefig(os.path.join(self.out_dir, 'sig_vs_bkg', f"sig_vs_bkg_{sig_sample}.pdf"))
                plt.close()
                
                
            with open(os.path.join(self.out_dir, 'All_Separations.txt'),'w') as f:
                f.writelines(sorted(chi2_out))
            with open(os.path.join(self.out_dir, 'Sum_weights.txt'), 'w') as f:
                f.writelines(sum_weights)
            
            #Plot signal samples with each of these
        
       
                 
            
            
        #Log loss per sample
        if kwargs.get('logloss_sample_hist',False):
            for group, samples in self.config['groupings'].items():
                index = np.where(np.isin(bkg_output['samples'],samples))
                weights = bkg_output['weights'][index] / np.sum(bkg_output['weights'][index])
                c, e = np.histogram(bkg_output['log_losses'][index], bins=logloss_bins,
                                            weights=weights)
                per_sample_counts.append(c)
                per_sample_edges.append(e)
                per_sample_names.append(group)
            
            p.plot_hist_stack(per_sample_edges, per_sample_counts, 
                              labels=per_sample_names, xlab='Log loss', ylab='Counts',
                             save_name=os.path.join(self.out_dir,'Logloss_bySampleNormalised.png'))
                
        
        #Overlay background and signal
        if kwargs.get('logloss_BkgvsSig_hist',False) and sig_output:
            if len(per_sample_counts) == 0:
                for group, samples in self.config['groupings'].items():
                    index = np.where(np.isin(bkg_output['samples'],samples))
                    weights = bkg_output['weights'][index] / np.sum(bkg_output['weights'][index])
                    c, e = np.histogram(bkg_output['log_losses'][index], bins=logloss_bins,
                                                weights=weights)
                    per_sample_counts.append(c)
                    per_sample_edges.append(e)
                    per_sample_names.append(group)
            
            sig_weights = sig_output['weights'] / np.sum(sig_output['weights'])
            sig_counts, sig_edges = np.histogram(sig_output['log_losses'], bins=logloss_bins,
                                                weights=sig_weights)
            
            p.plot_hist_stack(per_sample_edges+[sig_edges], per_sample_counts+[sig_counts],
                             labels=per_sample_names+['VLLs'], xlab='Log loss', ylab='Counts',
                            save_name=os.path.join(self.out_dir,'Logloss_bySampleNormalised_VLL.png'))
            
        
        
        
        #Overlay bkg and signal per sample
        '''
        norm_test_logloss_c = test_logloss_counts/np.sum(test_logloss_counts)
        
        #Calculate separation + make plot
        if kwargs.get('chi2_plots', False) and sig_output:
            
            separation_samples = self.config['separation_samples']
            sample_labels = self.config['sample_labels']
            sample_cols = self.config['sample_cols']
            
            histos = []
            edges = []
            histos.append(norm_test_logloss_c)
            edges.append(test_logloss_edges)
                                                         
            for sample in separation_samples:
                index = np.where(np.isin(sig_output['samples'],[sample]))
                weights = np.array(sig_output['weights'])[index] /  np.sum(np.array(sig_output['weights'])[index])
                loss_counts, loss_edges = np.histogram(sig_output['log_losses'][index], bins=logloss_bins, weights=weights)
               
                histos.append(loss_counts)
                edges.append(loss_edges)
                
                                                         
            chi2_out = []
            for i, sample in enumerate(separation_samples):
                chi2 = self.get_chi2distance(norm_test_logloss_c, histos[i+1])
                out_str = f"Histogram: {sample}, chi2 distance from background: {chi2}"
                chi2_out.append(out_str + '\n')
                
            with open(os.path.join(self.out_dir, 'Chi2_Distances.txt'),'w') as f:
                f.writelines(chi2_out)
                
            p.plot_hist_stack(edges, histos, xlab='Log loss', ylab='Counts', labels=['Bkg']+sample_labels, save_name=os.path.join(self.out_dir,'Separation_Hist.png'))


        
        
        #Plot cdf function
        if kwargs.get('cdf_hist',False):
            test_logloss_counts, test_logloss_edges = np.histogram(bkg_output['log_losses'], bins=logloss_bins, 
                                                                   weights=bkg_output['weights'])
            p.plot_cdf(test_logloss_edges, test_logloss_counts, xlab='Log loss', ylab='Cum. sum', 
                        save_name=os.path.join(self.out_dir, 'CumSumPlot_All.png'))
            
            
        #Plot Nsig vs Nbackground curves and SAVE outputs
        if kwargs.get('nsig_vs_nbkg', False) and sig_output:
            
            cumsum_hists = []
            loss_count_hists = []
            
            bkg_cumsum = np.cumsum(np.flip(norm_test_logloss_c))/sum(norm_test_logloss_c)
            cumsum_hists.append(bkg_cumsum)
            
            for sample in separation_samples:
                index = np.where(np.isin(sig_output['samples'],[sample]))
                #weights = np.array(sig_output['weights'])[index] /  np.sum(np.array(sig_output['weights'])[index])
                weights = np.array(sig_output['weights'])[index]
                loss_counts, loss_edges = np.histogram(sig_output['log_losses'][index], bins=logloss_bins, weights=weights)
                loss_count_hists.append(loss_counts)
                cum_sum = np.cumsum(np.flip(loss_counts))/sum(loss_counts)
                cumsum_hists.append(cum_sum)
                
         
            p.plot_Nsig_vs_Nbkg(cumsum_hists, xlab='$N_{bkg}$', ylab='$N_{sig}$',labels=['Bkg']+sample_labels, colors=['b']+sample_cols,
                               save_name=os.path.join(self.out_dir, 'Nsig_vs_Nbkg_curves.png'))
            
            if kwargs.get('sig_vs_nbkg', False):
                
                significances  = []
                bkg = np.cumsum(np.flip(test_logloss_counts)) / sum(test_logloss_counts)
                significances.append(bkg[:-20])
                for i in range(len(loss_count_hists)):
                    sig = np.cumsum(np.flip(loss_count_hists[i])) / np.sqrt(np.flip(test_logloss_counts))
                    significances.append(sig[:-20])
                
                p.plot_significances(significances, xlab='$N_{bkg}$', ylab='$\sigma$',labels=['Bkg']+sample_labels, colors=['b']+sample_cols,
                               save_name=os.path.join(self.out_dir, 'SigvsBkg_curves.png'))
            
        '''
        #Plot input vs output distributions
        if kwargs.get('inp_vs_out', False):
            bins = np.linspace(-5,5,20)
            inp = np.array(bkg_output['data']).reshape(len(bkg_output['data']), -1)
            out = np.array(bkg_output['reco']).reshape(len(bkg_output['data']), -1)

            vars = [n for n in self.config['training_variables'] if n not in ['weight','sample','eventNumber']]
            for i in range(inp.shape[-1]):
                
                x = inp[:,i]
                reco = out[:,i]
                plt.figure()
                plt.hist(x, bins=bins, label='Input', alpha=0.5)
                plt.hist(reco, bins=bins, label='Output',alpha=0.5)
                plt.title(f"{vars[i]}")
                plt.legend(title=self.config['Region_name'])
                if not os.path.exists(os.path.join(self.out_dir, f"input_vs_output")):
                    os.makedirs(os.path.join(self.out_dir, f"input_vs_output"))
                plt.savefig(os.path.join(self.out_dir, f"input_vs_output/{vars[i]}.png"))
                plt.close()
        ...
        
        
    def analyse_nf(self, bkg_output, sig_output=None, **kwargs):   
        
        anomaly_plot_settings = self.config.get('anomaly_score_plot', None)
        
        
        
        
        bkg_scores, min_prob, max_prob = self.model.scale_log_prob(bkg_output['ad_score'])
        
        
        
        if anomaly_plot_settings:
            num_bins = anomaly_plot_settings['num_bins']
            bin_min = anomaly_plot_settings['bin_min']
            bin_max = anomaly_plot_settings['bin_max']
            bins = np.linspace(bin_min, bin_max, num_bins)
            
            sig_sample_folder = 'sig_vs_bkg'
            if not os.path.exists(os.path.join(self.out_dir, sig_sample_folder)):
                os.makedirs(os.path.join(self.out_dir, sig_sample_folder))
            if not os.path.exists(os.path.join(self.out_dir, sig_sample_folder, 'log_plots')):
                os.makedirs(os.path.join(self.out_dir, sig_sample_folder, 'log_plots'))
            
            sig_scores, min_prob, max_prob = self.model.scale_log_prob(sig_output['ad_score'], min_prob, max_prob)
            
            chi2_out = []
            sum_weights = []
            sum_weights.append(f"Background: {sum(bkg_output['weights'])}\n")
            #Need to save the min prob and max prob for future scaling
            for sig_sample in set(sig_output['groups']):
                print(f"Running: {sig_sample}")

                inds = np.where(np.isin(sig_output['groups'], [sig_sample]))[0]

                plt.figure()
                plt.hist(bkg_scores, bins=bins, alpha=0.5, label='Bkg', density=True, weights=bkg_output['weights'])
                plt.hist(sig_scores[inds], bins=bins, alpha=0.5, label=sig_sample, density=True, weights=sig_output['weights'][inds])
                plt.xlabel('Anomaly score')
                plt.ylabel('Counts')
                plt.legend()
                plt.savefig(os.path.join(self.out_dir, sig_sample_folder, f"Sig_vs_bkg_{sig_sample}.png"))
                plt.savefig(os.path.join(self.out_dir, sig_sample_folder, f"Sig_vs_bkg_{sig_sample}.pdf"))
                plt.close()
                
                plt.figure()
                plt.hist(bkg_scores, bins=bins, alpha=0.5, label='Bkg', density=True, weights=bkg_output['weights'], log=True)
                plt.hist(sig_scores[inds], bins=bins, alpha=0.5, label=sig_sample, density=True, weights=sig_output['weights'][inds], log=True)
                plt.xlabel('Anomaly score')
                plt.ylabel('Counts')
                plt.legend()
                plt.savefig(os.path.join(self.out_dir, sig_sample_folder, 'log_plots', f"Sig_vs_bkg_{sig_sample}.png"))
                plt.savefig(os.path.join(self.out_dir, sig_sample_folder,'log_plots', f"Sig_vs_bkg_{sig_sample}.pdf"))
                plt.close()
                #Save
                
                #Calculate separations
                bkg_ad_counts, bkg_ad_edges = np.histogram(bkg_scores, bins=bins, weights=bkg_output['weights'])
                sig_ad_counts, sig_ad_edges = np.histogram(sig_scores[inds], bins=bins, weights=sig_output['weights'][inds])
                sep = self.get_chi2distance(bkg_ad_counts/np.sum(bkg_ad_counts), sig_ad_counts/np.sum(sig_ad_counts))
                out_str = f"{sig_sample}: {sep}\n"
                print(out_str)
                weight_str = f"{sig_sample}: {sum(sig_output['weights'][inds])}\n"
                sum_weights.append(weight_str)
                chi2_out.append(out_str)
            
            with open(os.path.join(self.out_dir, 'Chi2_Distances.txt'),'w') as f:
                f.writelines(sorted(chi2_out))
            
            
        #Do the same but plot background by group?
        if kwargs.get('bkg_plot'):
            
            
            #Make plot of unscaled values? 
            #Use min max of the distributions
            
            print("Making background plot...")
            groups = []
            unsc_groups = []
            group_names = []
            group_weights = []
            for group in set(bkg_output['groups']):
                inds = np.where(np.isin(bkg_output['groups'], [group]))[0]
                groups.append(bkg_scores[inds])
                unsc_groups.append(bkg_output['ad_score'][inds])
                group_names.append(group)
                group_weights.append(bkg_output['weights'][inds])
                sum_weights.append(f"{group}: {sum(bkg_output['weights'][inds])}\n")
                
            #Sort to plot the largest first
            group_sums = [sum(weights) for weights in group_weights]
            sorted_indices = np.argsort(group_sums)[::-1]
            groups = [groups[i] for i in sorted_indices]
            group_names = [group_names[i] for i in sorted_indices]
            group_weights = [group_weights[i] for i in sorted_indices]
            unsc_groups = [unsc_groups[i] for i in sorted_indices]
            
            plt.figure()
            plt.hist(groups, bins=bins,alpha=0.8, density=1, stacked=True, label=group_names, weights=group_weights)
            plt.xlabel('Anomaly score')
            plt.ylabel('Counts')
            plt.legend()
            plt.savefig(os.path.join(self.out_dir, f"Bkg_distribution_normalised.png"))
            plt.savefig(os.path.join(self.out_dir, f"Bkg_distribution_normalised.pdf"))
            plt.close()
            
            plt.figure()
            #Make 200 bins from min to max 
            unsc_bins = np.linspace(min(bkg_output['ad_score']), max(bkg_output['ad_score']), 200)
            plt.hist(groups, bins=unsc_bins,alpha=0.8, density=1, stacked=True, label=group_names, weights=group_weights)
            plt.xlabel('Anomaly score')
            plt.ylabel('Counts')
            plt.legend()
            plt.savefig(os.path.join(self.out_dir, f"Bkg_distribution_unscaled.png"))
            plt.savefig(os.path.join(self.out_dir, f"Bkg_distribution_unscaled.pdf"))
            plt.close()

            plt.figure()
            plt.hist(groups, bins=bins,alpha=0.8, stacked=True, label=group_names, weights=group_weights)
            plt.xlabel('Anomaly score')
            plt.ylabel('Counts')
            plt.legend()
            plt.savefig(os.path.join(self.out_dir, f"Bkg_distribution.png"))
            plt.savefig(os.path.join(self.out_dir, f"Bkg_distribution.pdf"))
            plt.close()
            
            plt.figure()
            plt.hist(groups, bins=bins,alpha=0.8, stacked=True, label=group_names, weights=group_weights, log=True)
            plt.xlabel('Anomaly score')
            plt.ylabel('Counts')
            plt.legend()
            plt.savefig(os.path.join(self.out_dir, f"Bkg_distribution_log.png"))
            plt.savefig(os.path.join(self.out_dir, f"Bkg_distribution_log.pdf"))
            plt.close()
            
            with open(os.path.join(self.out_dir, 'Sum_weights.txt'), 'w') as f:
                f.writelines(sum_weights)
            print("Finished making background plot")
            
        if kwargs.get('inp_vs_out', None):
            
            bins = np.linspace(-5,5,50)
            inp = np.array(bkg_output['data']).reshape(len(bkg_output['data']), -1)
            out = np.array(sig_output['data']).reshape(len(sig_output['data']), -1)
    
            vars = [n for n in self.config['training_variables'] if n not in ['weight','sample','eventNumber']]
            for i in range(inp.shape[-1]):
               
                x = inp[:,i]
                sig = out[:,i]
                plt.figure()
                plt.hist(x, bins=bins, label='Background', alpha=0.5, density=True)
                plt.hist(sig, bins=bins, label='All Signals', alpha=0.5, density=True)
                plt.xlabel(f"{vars[i]}")
                plt.ylabel("Counts")
                plt.legend(title=self.config['Region_name'])
                if not os.path.exists(os.path.join(self.out_dir, f"inputs_vs_sig")):
                    os.makedirs(os.path.join(self.out_dir, f"inputs_vs_sig"))
                plt.savefig(os.path.join(self.out_dir, f"inputs_vs_sig/{vars[i]}.png"))
                plt.savefig(os.path.join(self.out_dir, f"inputs_vs_sig/{vars[i]}.pdf"))
                plt.close()
            
      
        ...
        
    def analyse_results(self, bkg_output, sig_output=None, **kwargs):
        
        if self.config['model_type'] == 'AE' or self.config['model_type'] == 'VAE':
            self.analyse_vae(bkg_output, sig_output, **kwargs)
        
        elif self.config['model_type'] == 'NF':
            self.analyse_nf(bkg_output, sig_output, **kwargs)
            
        else:
            print("Not implemented an analyse function for this model type.")
        
        
if __name__ == '__main__':
    
    
    from model.model_getter import get_model
    from run.train import Trainer
    import os
    from preprocessing.dataset_io import DatasetHandler
    from preprocessing.dataset import data_set
    from torch.utils.data import DataLoader
    
    #Read config input
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config",default="configs/training_config.yaml", help="Choose the master config to use")
    args = parser.parse_args()
    
    train_conf = args.config
    
    out_dir = '/nfs/pic.es/user/j/jharriso/IFAE_ML/results/TestRun/1Z_0b_2SFOS_NFs/Run_0057-13-05-2023'

    out_dir = '/nfs/pic.es/user/j/jharriso/IFAE_ML/results/TestRun/1Z_0b_2SFOS_AllSigs/Run_1050-16-05-2023'
    
    out_dir = '/nfs/pic.es/user/j/jharriso/IFAE_ML/results/NF_NewTrainTestSplit_Odd/0Z_0b_1SFOS_NFs/Run_1517-22-05-2023'
    
    #vll_conf = 'configs/training_configs/VLL_VAE_config.yaml'
    
    
    t = Tester(config=train_conf)
    
    
    t.out_dir = out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    load = False
    save = True
    
    
    #Get model
    model = get_model(conf=train_conf)
    model.load_state_dict(torch.load(os.path.join(t.out_dir,'model_state_dict.pt')))
    print("Successfully loaded model...")
    
    if load:
        with open(os.path.join(out_dir, 'saved_outputs.pkl'), 'rb') as f:
            bkg_output = pickle.load(f)
        with open(os.path.join(out_dir, 'saved_signal_outputs.pkl'), 'rb') as f:
            sig_output = pickle.load(f)
        
    else:
        
        dh = DatasetHandler(train_conf, job_name='TestRun', out_dir=out_dir)
        train, val, test = dh.split_dataset(use_val=dh.config['validation_set'], 
                use_eventnumber=dh.config.get('use_eventnumber', None))
    
        train_data = data_set(train)
        test_data = data_set(test)
        train_loader = DataLoader(train_data, batch_size=dh.config['batch_size'], shuffle=True)    
        test_loader = DataLoader(test_data, batch_size=2056)
    
        if len(val) != 0:
            val_data = data_set(val)
            val_loader = DataLoader(val_data, batch_size=2056)
        else:
            val_loader=None
        
        
        sig_dh  =  DatasetHandler(train_conf, scalers=dh.scalers, out_dir=out_dir)
        sig_data = data_set(sig_dh.data)
        sig_loader = DataLoader(sig_data, batch_size=2056, shuffle=True)
    
        #Evaluate bkg+signal
        #Just use even model to evaluate on odd bkg data
        bkg_output = t.evaluate(model, test_loader)
        sig_output = t.evaluate(model, sig_loader)

  
    if save:
        print("Saving evaluation outputs to: ", out_dir)
        
        with open(os.path.join(out_dir, 'saved_outputs.pkl'), 'wb') as f:
            pickle.dump(bkg_output, f)
        with open(os.path.join(out_dir, 'saved_signal_outputs.pkl'), 'wb') as f:
            pickle.dump(sig_output, f)
        
    out_plots = t.config.get('output_plots',None)
    
    if out_plots:
        t.analyse_results(bkg_output, sig_output=sig_output, **out_plots)
    else:
        print("No output plots set in config... Ending script.")
    
    '''
    analysis = {
        'loss_hist' : True,
        'logloss_hist' : True,
        'logloss_sample_hist' : True,
        'logloss_BkgvsSig_hist' : True,
        'cdf_hist' : True,
        'chi2_plots' : True,
        'nsig_vs_nbkg' : True,
        'sig_vs_nbkg' : False
    }
    
    t.out_dir = out_dir
    t.analyse_results(bkg_output, sig_output=sig_output, **analysis)
    '''
    