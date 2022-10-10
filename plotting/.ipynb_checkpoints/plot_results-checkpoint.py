import matplotlib.pyplot as plt
import numpy as np

class Plotter():
    
    def __init__(self):
        
        ...
        
        
    def plot_scatter(self, x, y, xlab=None, ylab=None,title=None,label=None,save_name=None):
        plt.scatter(x,y,label=label)
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.legend()
        plt.title(title)
        if save_name:
            plt.savefig(save_name)
        plt.close()
        
        
    def plot_scatter_overlay(self, x_list, y_list=None, xlab=None, ylab=None,title=None,labels=None,
                             save_name=None, val_frequency=1):
        
        if type(x_list) == dict:
            for key, val in x_list.items():
                freq=val_frequency
                if key == 'Train':
                    freq=1
                plt.scatter([i*freq for i in range(len(val))], val, label=key)
        else:
            for i, (x, y) in enumerate(zip(x_list, y_list)):
                plt.scatter(x, y, label=labels[i])
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.legend()
        plt.title(title)
        if save_name:
            plt.savefig(save_name)
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
        plt.grid(True)
        if save_name:
            plt.savefig(save_name)
        plt.close()
    
    def plot_hist_stack(self, edge_list, count_list, xlab=None, ylab=None, title=None,
                       labels=None, save_name=None):
        
        for i, (edge,count) in enumerate(zip(edge_list,count_list)):
            plt.step(edge[:-1], count, label=labels[i])
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        if save_name:
            plt.savefig(save_name)
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
        plt.grid(True)
        if save_name:
            plt.savefig(save_name)
        plt.close()
        
    def plot_2d(self):
        ...

        
    
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
    