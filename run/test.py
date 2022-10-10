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

#Put the anomaly score back into the ROOT files

#---------------------------------------

import torch

import numpy as np
import os
import yaml
import math

#Plotting
from plotting.plot_results import Plotter
from tqdm import tqdm

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
    
    def evaluate_vae(self, model, testloader):
        
        output = {
            'data' : [],
            'mu' : [],
            'loss' : [],
            'samples' : [],
            'groups' : [],
            'weights' : [],
            'log_losses' : [],
            'eventNumber' : [],
        }
        
        with torch.no_grad():
            
            #for idx, (data, weights, sample, sc_weight) in enumerate(testloader):
            for idx, data_dict in tqdm(enumerate(testloader)):
               
                data = data_dict['data']
                sample = data_dict['sample']
                sc_weight = data_dict['scaled_weight']
                evtNumber = data_dict['eventNumber']
                weights = data_dict['weight']
                
                # Feeding a batch of images into the network to obtain the output image, mu, and logVar
                out, mu, logVar = model(data)
                loss, mse, kld = model.loss_function(out, data, mu, logVar)
                
                output['data'].append(np.array(data))
                output['mu'].append(mu.tolist())
                output['loss'].append(loss.item())
                output['samples'].append(sample[0])
                output['weights'].append(weights.item())
                output['log_losses'].append(math.log(loss.item()))
                output['groups'].append(self.reversed_groupings.get(sample[0],'All'))
                output['eventNumber'].append(evtNumber.item())
        
        output['loss'] = np.array(output['loss'])
        output['log_losses'] = np.array(output['log_losses'])
        output['weights'] = np.array(output['weights'])
        return output
        
        
    def analyse_results(self, bkg_output, sig_output=None, **kwargs):
        
        num_logloss_bins = 50
        logloss_bins = np.linspace(-10,10,num_logloss_bins)
        
        per_sample_counts, per_sample_edges, per_sample_names = [], [], []
        test_logloss_counts, test_logloss_edges = None,None
        
        p = Plotter()
        
        #Loss histogram
        if kwargs.get('loss_hist',False):
            loss_counts, loss_edges = np.histogram(bkg_output['loss'], bins=50, weights=bkg_output['weights'])
            p.plot_hist(loss_edges, loss_counts, label='All', save_name=os.path.join(self.out_dir, 'Loss_hist_All.png'))
            
        
        #Log loss histogram
        if kwargs.get('logloss_hist',False):
            test_logloss_counts, test_logloss_edges = np.histogram(bkg_output['log_losses'], bins=logloss_bins, 
                                                                   weights=bkg_output['weights'])
            p.plot_hist(test_logloss_edges, test_logloss_counts, xlab='Log loss', ylab='Counts', 
                        label='All', save_name=os.path.join(self.out_dir, 'Log_Ascore_All.png'))
            
            
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
        if kwargs.get('logloss_BkgvsSig_hist',False):
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
                             

        
                
        #Overlay per sample vs VLL
        
        
        #Overlay bkg and signal per sample
        
        norm_test_logloss_c = test_logloss_counts/np.sum(test_logloss_counts)
        
        #Calculate separation + make plot
        if kwargs.get('chi2_plots', False) and sig_output is not None:
            
            separation_samples = ['Esinglet300', 'Mdoublet700']
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
            p.plot_hist_stack(edges, histos, xlab='Log loss', ylab='Counts', labels=['Bkg','E(300)','M(700)'], save_name=os.path.join(self.out_dir,'Separation_Hist.png'))# colours=['r','g','b']

    
    
        #Plot 2D histograms vs variables
        if kwargs.get('2d_hist', False):
            ...
        
        #Plot latent spaces
        
        #Plot per production + decay
        
        #Plot cdf function
        if kwargs.get('cdf_hist',False):
            test_logloss_counts, test_logloss_edges = np.histogram(bkg_output['log_losses'], bins=logloss_bins, 
                                                                   weights=bkg_output['weights'])
            p.plot_cdf(test_logloss_edges, test_logloss_counts, xlab='Log loss', ylab='Cum. sum', 
                        save_name=os.path.join(self.out_dir, 'CumSumPlot_All.png'))
        
        ...
        
        
    def update_root_files(self):
        ...
        
    def run_vae(self):
        ...
        
        
if __name__ == '__main__':
    
    
    
    #Get sample dataset
    
    #Load a model
    
    #Evaluate bkg+signal
    
    analysis = {
        'loss_hist' : True,
        'logloss_hist' : True,
        'logloss_sample_hist' : True,
        'logloss_BkgvsSig_hist' : True,
        'cdf_hist' : True
    }
    
    N=10000
    x = np.random.normal(loc=10,size=N)
    logx = np.log(x)
    y = np.random.normal(loc=100,size=N)
    z = np.concatenate((x,y))
    logz = np.log(z)
    samples = ['VV' for i in x]
    samples2 = ['VVV' for i in y]
    s = samples+samples2

    output = {'loss' : z,
             'log_losses' : logz,
             'weights' : np.array([1 for i in range(len(z))]),
              'samples' : s,
             }
    
    output2 = {'loss' : x,
             'log_losses' : logx,
             'weights' : np.array([1 for i in range(len(x))]),
              'samples' : ['Sig' for i in x],
             }
    
    t = Tester(config='configs/training_config.yaml')
    t.out_dir = 'outputs/test_dump'
    
    t.analyse_results(output, sig_output=output2, **analysis)
    