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
import matplotlib.pyplot as plt

import pickle

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
            'reco' : [],
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
            #for idx, data_dict in tqdm(enumerate(testloader)):
            for idx, data_dict in enumerate(testloader):   
            
                data = data_dict['data']
                sample = data_dict['sample']
                sc_weight = data_dict['scaled_weight']
                evtNumber = data_dict['eventNumber']
                weights = data_dict['weight']
                
                # Feeding a batch of images into the network to obtain the output image, mu, and logVar
                out, mu, logVar = model(data)
                loss, mse, kld = model.loss_function(out, data, mu, logVar)
                
                output['data'].append(np.array(data))
                output['reco'].append(np.array(out))
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
            
            
            if kwargs.get('trexfitter_plot', False) and sig_output:
                
                sample_cs = []
                sample_es = []
                sample_ns = []
               
                for group, samples in self.config['groupings'].items():
                    index = np.where(np.isin(bkg_output['samples'],samples))
                    weights = bkg_output['weights'][index]
                    c, e = np.histogram(bkg_output['log_losses'][index], bins=logloss_bins,
                                                weights=weights)
                    sample_cs.append(c)
                    sample_es.append(e)
                    sample_ns.append(group)
            
                
                sig_weights = sig_output['weights']
                sig_counts, sig_edges = np.histogram(sig_output['log_losses'], bins=logloss_bins,
                                                    weights=sig_weights)

                p.plot_bar_stack(sample_es+[sig_edges], sample_cs+[sig_counts],
                                 labels=sample_ns+['VLLs'], xlab='Log loss', ylab='Counts',
                                save_name=os.path.join(self.out_dir,'Logloss_bySample.png'))
            
        
        #Plot input vs output distributions
        if kwargs.get('inp_vs_out', False):
            bins = np.linspace(-5,5,20)
            inp = np.array(bkg_output['data']).reshape(len(bkg_output['data']), -1)
            out = np.array(bkg_output['reco']).reshape(len(bkg_output['data']), -1)
            print(out.shape)
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
        
        
    def update_root_files(self):
        ...
        
    def run_vae(self):
        ...
        
        
if __name__ == '__main__':
    
    
    from model.model_getter import get_model
    from run.train import Trainer
    import os
    from preprocessing.dataset_io import DatasetHandler
    from preprocessing.dataset import data_set
    from torch.utils.data import DataLoader
    
    train_conf = 'configs/training_configs/training_config.yaml'
    vll_conf = 'configs/training_configs/VLL_VAE_config.yaml'
    
    #NEED TO UPDATE THIS TO BE ABLE TO TEST A TRAINED MODEL EASILY
    t = Tester(config=train_conf)
    
    #Load a model
    #even_load_dir = "outputs/EVEN_FINAL_VAE_1318-23-09-2022"
    #odd_load_dir = "outputs/ODD_FINAL_VAE_1414-23-09-2022"
    
    path10 = 'outputs/good_runs/LongRun_VAE_1Z_0b_2SFOS_10GeV/Run_1507-22-12-2022_B'
    path25 = 'outputs/good_runs/LongRun_VAE_1Z_0b_2SFOS_25GeV/Run_1402-17-01-2023'
    
    
    #even_load_dir = "outputs/LongRun_VAE_1Z_0b_2SFOS_10GeV/Run_1626-23-12-2022"
    even_load_dir = path10
    
    out_dir =  'outputs/good_runs/eval_25GeV_training'
    
    load = False
    save = True
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    #Just use Even model
    model = get_model(conf=train_conf)
    model.load_state_dict(torch.load(os.path.join(even_load_dir,'model_state_dict.pt')))
    
    if load:
        with open(os.path.join(out_dir, 'odd_bkg_data.pkl'), 'rb') as f:
            bkg_output = pickle.load(f)
        with open(os.path.join(out_dir, 'sig_data.pkl'), 'rb') as f:
            sig_output = pickle.load(f)
        
    else:
        
        #Get the even dataset and signal dataset
        
        dh = DatasetHandler(train_conf)
        train, val, test = dh.split_dataset(val=dh.config['validation_set'], 
                    use_eventnumber=dh.config.get('use_eventnumber',None))

        train_data = data_set(train)
        test_data = data_set(test)
        train_loader = DataLoader(train_data, batch_size=dh.config['batch_size'], shuffle=True)    
        test_loader = DataLoader(test_data, batch_size=1)
        
        
        sig_dh  =  DatasetHandler(vll_conf, scalers=dh.scalers)
        sig_data = data_set(sig_dh.data)
        sig_loader = DataLoader(sig_data, batch_size=1, shuffle=True)
    
        #Evaluate bkg+signal
        #Just use even model to evaluate on odd bkg data
        bkg_output = t.evaluate_vae(model, test_loader)
        sig_output = t.evaluate_vae(model, sig_loader)
    
  
    if save:
        with open(os.path.join(out_dir, 'saved_outputs.pkl'), 'wb') as f:
            pickle.dump(bkg_output, f)
        with open(os.path.join(out_dir, 'saved_signal_outputs.pkl'), 'wb') as f:
            pickle.dump(sig_output, f)
        

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

    