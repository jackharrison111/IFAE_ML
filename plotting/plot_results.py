import matplotlib.pyplot as plt
import numpy as np
import mplhep as hep
import math

class Plotter():
    
    def __init__(self):
        
        plt.style.use(hep.style.ATLAS)
        self.grid = True
        ...
        
        
    def plot_loss(self, x, y, xlab=None, ylab=None,title=None,label=None,save_name=None):
        hfont = {'fontname':'Helvetica'}
        
        plt.plot(x,y,label=label)
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.legend()
        plt.grid(self.grid)
        
        plt.title(title)
        if save_name:
            plt.savefig(save_name)
            if '.png' in save_name:
                plt.savefig(save_name.replace(".png", ".pdf"))
        plt.close()
        
        
    def plot_loss_overlay(self, x_list, y_list=None, xlab=None, ylab=None,title=None,labels=None,
                             save_name=None, val_frequency=1):
        
        if type(x_list) == dict:
            for key, val in x_list.items():
                freq=val_frequency
                if key == 'Train':
                    freq=1
                print(f"Plotting {key} using freq {freq}, length: {len(val)}")
                plt.plot([i*freq for i in range(len(val))], val, label=key)
        else:
            for i, (x, y) in enumerate(zip(x_list, y_list)):
                plt.plot(x, y, label=labels[i])
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.grid(self.grid)
        plt.legend()
        plt.title(title)
        if save_name:
            plt.savefig(save_name)
            if '.png' in save_name:
                plt.savefig(save_name.replace(".png", ".pdf"))
        plt.close()
        
        
    def plot_hist(self, edges, counts, xlab=None, ylab=None,title=None,label=None,
                             save_name=None):
     
        #plt.bar(edges[:-1], counts, width=edges[1]-edges[0], label=key, fill=fill, color=colour, edgecolor=colour)
        plt.step(edges[:-1], counts, label=label)
        #plt.stairs(counts, edges, color=colour, label=key)
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.title(title)
        plt.legend()
        plt.grid(self.grid)
        if save_name:
            plt.savefig(save_name)
            if '.png' in save_name:
                plt.savefig(save_name.replace(".png", ".pdf"))
        plt.close()
    
    def plot_hist_stack(self, edge_list, count_list, xlab=None, ylab=None, title=None,
                       labels=None, save_name=None):
        
        for i, (edge,count) in enumerate(zip(edge_list,count_list)):
            plt.step(edge[:-1], count, label=labels[i])
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.title(title)
        plt.legend()
        plt.grid(self.grid)
        if save_name:
            plt.savefig(save_name)
            if '.png' in save_name:
                plt.savefig(save_name.replace(".png", ".pdf"))
        plt.close()
        
    def plot_cdf(self, edge_list, count_list, xlab=None, ylab=None, title=None,
                save_name=None, line_values=[0.05]):
        
        cum_sum = 1 - np.cumsum(count_list)/sum(count_list)
        plt.step(edge_list[:-1], cum_sum)
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.title(title)
        plt.legend()
        for l in line_values:
            idx = (np.abs(cum_sum - l)).argmin()
            plt.axvline(x=edge_list[idx], ymin = 0, ymax = 1,
            color ='red', label=f'{l*100}% Bkg')
        plt.legend()
        plt.grid(self.grid)
        if save_name:
            plt.savefig(save_name)
            if '.png' in save_name:
                plt.savefig(save_name.replace(".png", ".pdf"))
        plt.close()
        
    def plot_2d(self):
        ...

    def plot_Nsig_vs_Nbkg(self, cumsum_hists, xlab=None, ylab=None,labels=None,
                               save_name=None, title=None, colors=[]):
        
        for i in range(1,len(cumsum_hists)):
            plt.plot(cumsum_hists[0], cumsum_hists[i], label=labels[i], color=colors[i])
            
        even_bins = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
        plt.plot(even_bins, even_bins, '--', color='blue')
        
        plt.xticks(even_bins)
        plt.yticks(even_bins)
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.title(title)
        plt.legend()
        plt.grid(self.grid)
        if save_name:
            plt.savefig(save_name)
            if '.png' in save_name:
                plt.savefig(save_name.replace(".png", ".pdf"))
        plt.close()
        
    def plot_significances(self, cumsum_hists, xlab=None, ylab=None,labels=None,
                               save_name=None, title=None, colors=[]):
        
        for i in range(1,len(cumsum_hists)):
            plt.plot(cumsum_hists[0], cumsum_hists[i], label=labels[i], color=colors[i])
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.title(title)
        plt.legend()
        plt.grid(self.grid)
        if save_name:
            plt.savefig(save_name)
            if '.png' in save_name:
                plt.savefig(save_name.replace(".png", ".pdf"))
        plt.close()
        ...
    
    def plot_bar_stack(self, edge_list, count_list, xlab=None, ylab=None, title=None,
                       labels=None, save_name=None):
        
        for i, (edge,count) in enumerate(zip(edge_list,count_list)):
            plt.bar(edge[:-1], count, label=labels[i], alpha=1)
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.title(title)
        plt.legend()
        plt.grid(self.grid)
        if save_name:
            plt.savefig(save_name)
            if '.png' in save_name:
                plt.savefig(save_name.replace(".png", ".pdf"))
        plt.close()
    

    def plot_sig_bkg(self, var_choice, hist, sig_hist=None,
                figsize=(10, 6),
                bkg_label='Bkg',
                sig_label='Sig',
                alpha=0.5,
                density=False,
                title=None,
                edgecol="black",
                save_name=None,
                sig_hist_type='fill',
                ):
    
        hep.style.use("ATLAS") 
    
        xlabels = {
        'met_met' : r"$E_{T}^{miss}$ [GeV]",
        'Mllll0123' : "$M_{4\ell}$ [GeV]",
        'HT_lep' : "$H_{T_{4\ell}}$ [GeV]",
        'HT_jets' : "$H_{T_{jets}}$ [GeV]",
        'nJets_Continuous' : "$N_{jets}$",
        'best_mZll' : "$M_{Z}$ [GeV]",
        'other_mZll' : "$M_{\ell\ell}$ [GeV]",
        'M3l_high' : "$M_{3\ell_{high}}$ [GeV]",
        'M3l_low' : "$M_{3\ell_{low}}$ [GeV]",
        'best_ptZll' : "$p_{T_{Z}}$ [GeV]",
        'other_ptZll' : "$p_{T_{\ell\ell}}$ [GeV]",
        'MtLepMet' : "$M_{T_{4\ell,E_{T}^{miss}}}$ [GeV]",
        'MT_otherllMET' : "$M_{T_{\ell\ell,E_{T}^{miss}}}$ [GeV]",
        'MT_ZllMET' : "$M_{T_{Z,E_{T}^{miss}}}$ [GeV]",
        'ad_score' : "Anomaly score",
        'sumPsbtag' : "sumPCB",
        'weight' : "MC weight",
        }
    
        ylabels = {
            'met_met' : "Counts",
            'Mllll0123' : "Counts",
            'HT_lep' : "Counts",
            'HT_jets' : "Counts",
            'nJets_Continuous' : "Counts",
            'best_mZll' : "Counts",
            'other_mZll' : "Counts",
            'M3l_high' : "Counts",
            'M3l_low' : "Counts",
            'best_ptZll' : "Counts",
            'other_ptZll' : "Counts",
            'MtLepMet' : "Counts",
            'MT_otherllMET' : "Counts",
            'MT_ZllMET' : "Counts",
            'ad_score' : "Counts",
            'sumPsbtag' : "Counts",
            'weight' : "Counts",
        }
    
        fig, ax = plt.subplots(figsize=figsize)
        hep.histplot(hist,histtype="fill",
            label=bkg_label,
            alpha=alpha,
            edgecolor=edgecol,
            density=density,
            ax=ax)
    

        ax.set_ylim(bottom=0)
    
        if sig_hist:
            hep.histplot(sig_hist,histtype=sig_hist_type,
            label=sig_label,
            alpha=alpha,
            edgecolor=edgecol,
            density=density,
            ax=ax)
            if math.isfinite(max(hist[0])) and math.isfinite(max(sig_hist[0])):
                ax.set_ylim(0,max(max(hist[0]), max(sig_hist[0]))*1.25)
    
        hep.atlas.label(" Internal", data=True, lumi=139)
        ax.legend(title=title)
        ax.set_xlabel(xlabels[var_choice], fontsize=18)
        ax.set_ylabel(ylabels[var_choice], fontsize=18)
        ax.tick_params(axis='x', which='major', pad=8.5)

        if save_name:
            #plt.savefig(os.path.join(save_dir, f"{var_choice}.png"))
            #plt.savefig(os.path.join(save_dir, f"{var_choice}.pdf"))
            plt.savefig(save_name+".png")
            plt.savefig(save_name+".pdf")
        
        plt.close()
    
        
    
if __name__ == '__main__':
    
    N = 500
    x = np.random.normal(size=N)
    y = np.random.normal(size=N)
    p = Plotter()
    p.plot_scatter(x,y, save_name='outputs/test_scatter.png')
    
    a, b = np.random.rand(N), np.random.rand(N)
    p.plot_scatter_overlay([x,a], [y,b], labels=['One', 'Two'], save_name='outputs/test_scatter_overlay.png')
    
    c, e = np.histogram(x)
    p.plot_cdf(e,c, save_name='outputs/test_cdf.png')
    