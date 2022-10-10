

from preprocessing.dataset_io import DatasetHandler
import pandas as pd
from time import perf_counter
import numpy as np
from preprocessing.dataset import data_set
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.model_selection import KFold 
import torch

from tqdm import tqdm


def scale_weights(df):
    sum_sig = sum(df.loc[df['label']==1]['weight'])
    sum_bkg = sum(df.loc[df['label']==0]['weight'])
    df.loc[df['label']==1,'weight'] = df.loc[df['label']==1,'weight']*(0.5/sum_sig)
    df.loc[df['label']==0,'weight'] = df.loc[df['label']==0,'weight']*(0.5/sum_bkg)
    

    sum_wpos_check = sum(df.loc[df['label']==1,'weight'])
    sum_wneg_check = sum(df.loc[df['label']==0,'weight'])
    
    print("Weight check: ", sum_wpos_check, sum_wneg_check)
    
    return df
      
      
def train(model, dataloader, val_loader, num_epochs, optimizer, val_freq=3):
    
    epoch_losses = []
    val_losses = []
    for j in range(num_epochs):
        print(f"Running epoch {j}.")
        e_l = 0
        for i, data_dict in tqdm(enumerate(train_loader)):
            model.train()
            data = data_dict['data']
            scaled_weights = data_dict['scaled_weight']
            weights = data_dict['weight']
            sample = data_dict['sample']
            label = data_dict['label']

            optimizer.zero_grad()
            output = model(data)
            loss_dict = model.loss_function(output, label.view(output.shape[0],-1), w=weights.view(output.shape[0],-1))
            loss = loss_dict['loss']
            loss.backward()
            optimizer.step()
            e_l += loss.item()/len(train_loader)
        epoch_losses.append(e_l)
            
        val_l = 0
        val_losses.append(val_l+j)
        print(f"Epoch loss: {e_l}, Val loss: {val_losses[-1]}")
        continue
        if j % val_freq != 0:
            continue
        for i, data_dict in enumerate(val_loader):
            model.eval()
            data = data_dict['data']
            scaled_weights = data_dict['scaled_weight']
            weights = data_dict['weight']
            sample = data_dict['sample']
            label = data_dict['label']
            output = model(data)
            loss_dict = model.loss_function(output.view(label.shape), label, w=weights)
            val_l += loss_dict['loss'].item()
            
        val_losses.append(val_l/len(val_loader))
        
        

    return model, epoch_losses, val_losses


@torch.no_grad()
def evaluate(model, dataloader):
    
    
    out_dict = {
        'out' : [],
        'loss' : [],
        'sample' : [],
        'label' : [],
        
    }
    for i, data_dict in enumerate(dataloader):
        model.eval()
        data = data_dict['data']
        scaled_weights = data_dict['scaled_weight']
        sample = data_dict['sample']
        label = data_dict['label']

        output = model(data)
        #loss_dict = model.loss_function(output, label.view(output.shape[0],-1))
        #loss = loss_dict['loss']

        out_dict['out'].append(output.item())
        out_dict['label'].append(label.item())
        
    out_dict['label'] = np.array(out_dict['label'])
    out_dict['out'] = np.array(out_dict['out'])
    return out_dict



if __name__ == '__main__':
    
    s = perf_counter()
    out_folder = 'mdoublet_2'
    
    #Get datasets
    dh = DatasetHandler('configs/linear_network_config.yaml')
    sig_dh = DatasetHandler('configs/VLL_signal_config.yaml', scalers=dh.scalers)
    
    size = -1
    dh.data = dh.data[:size]
    sig_dh.data = sig_dh.data[:size]
    dh.data['label'] = 0
    sig_dh.data['label'] = 1
    print(f"Signals are using: {sig_dh.data['sample'].unique()}")
    dh.data = pd.concat([dh.data,sig_dh.data])
    
    #scale the weights
    dh.data = scale_weights(dh.data)  
    
    data, test = dh.split_per_sample(val=False, use_eventnumber=dh.config.get('use_eventnumber',None))
  
    
    #Train test split
    k = 3
    kf = KFold(n_splits=k)
    
    e_l, v_l = [], []
    
    best_model = None
    best_val_loss = None
    for k, (train_index, test_index) in enumerate(kf.split(data)):
        if k > 0:
            continue
        #Make dataset
        print(f"Running k-fold {k}")
        print(data.columns)
        train_data = data_set(data.iloc[train_index].copy())
        val_data = data_set(data.iloc[test_index].copy())

        #Get model
        from model.linear_network import NN
        useful_columns = [col for col in dh.config['training_variables'] if col not in ['sample','weight', 'scaled_weight']]
        dimensions = [len(useful_columns),8 , 4, 1]
        model = NN(dimensions)
    
        train_loader = DataLoader(train_data, batch_size=dh.config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_data, batch_size=1)

        optimizer = optim.SGD(model.parameters(), lr=dh.config['learning_rate'])
        model, epoch_losses, val_losses = train(model, train_loader, val_loader, dh.config['num_epochs'], optimizer)
        e_l.append(epoch_losses)
        v_l.append(val_losses)
        
        if not best_model or val_losses[-1] < best_val_loss:
            best_model = model
            best_val_loss = val_losses[-1]

        import matplotlib.pyplot as plt
        plt.scatter([i for i in range(len(epoch_losses))], epoch_losses, label='Train')
        plt.scatter([i for i in range(len(val_losses))], val_losses, label='Val')
        plt.legend()
        plt.savefig(f'outputs/linear_NN/{out_folder}/epoch_losses_{k}.png')
        plt.close()
    
    
    e_l = np.array(e_l)
    v_l = np.array(v_l)
    av_epoch_l = e_l.mean(axis=0)
    std_epoch_l = e_l.std(axis=0)
    av_val_l = v_l.mean(axis=0)
    std_val_l = v_l.std(axis=0)
    
    plt.errorbar([i for i in range(len(av_epoch_l))], av_epoch_l, yerr=std_epoch_l,
                 label='Train', capsize=8)
    plt.errorbar([i for i in range(len(av_val_l))], av_val_l, yerr=std_val_l,
                 label='Val', capsize=8)
    plt.legend()
    plt.savefig(f'outputs/linear_NN/{out_folder}/cv_results.png')
    plt.close()
    
    
    #Test the model
    test_data = data_set(test)
    test_loader = DataLoader(test_data, batch_size=1)
    
    out = evaluate(best_model, test_loader)
    
    bkg_ind = np.where(out['label'] == 0)
    sig_ind = np.where(out['label'] == 1)
    
    plt.hist(out['out'][bkg_ind], alpha=0.5, label='Bkg')
    plt.hist(out['out'][sig_ind], alpha=0.5, label='Sig')
    plt.xlabel('NN score')
    plt.ylabel('Counts')
    plt.legend()
    plt.savefig(f'outputs/linear_NN/{out_folder}/Sig_comparison.png')
    plt.close()
        

    import matplotlib.pyplot as plt
    num_bins=25
    bins = np.linspace(0,1,num_bins)
    bkg_vals, bkg_bins, _ = plt.hist(out['out'][bkg_ind], label='Bkg',bins=bins, density=True,alpha=0.5)
    vals, bins, _ = plt.hist(out['out'][sig_ind], label='Sig', bins=bins, density=True,alpha=0.5)
    vals = vals/sum(vals)
    bkg_vals = bkg_vals/sum(bkg_vals)
    plt.xlabel('NN Score')
    plt.ylabel('Counts')
    plt.legend()
    plt.savefig(f'outputs/linear_NN/{out_folder}/Sig_comparison_Norm.png')
    plt.close()
    
    def get_chi2distance(x,y):
        ch2 = np.nan_to_num(((x-y)**2)/(x+y), copy=True, nan=0.0, posinf=None, neginf=None)
        ch2 = 0.5 * np.sum(ch2)
        return ch2

    chi2 = get_chi2distance(vals, bkg_vals)
    with open(f'outputs/linear_NN/{out_folder}/chi2.txt', 'w') as f:
        f.writelines([f"Chi2 value: {chi2}"])
    print(f"Chi2 value: {chi2}")
            
    f = perf_counter()
    print(f"Time taken: {round((f-s)/60,3)}mins.")
    
    
    