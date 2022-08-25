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
        plt.show()
        
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
        plt.show()
        
    
    
    
if __name__ == '__main__':
    
    N = 50
    x = np.random.rand(N)
    y = np.random.rand(N)
    p = Plotter()
    p.plot_scatter(x,y, save_name='outputs/test_scatter.png')
    
    a, b = np.random.rand(N), np.random.rand(N)
    p.plot_scatter_overlay([x,a], [y,b], labels=['One', 'Two'], save_name='outputs/test_scatter_overlay.png')
    