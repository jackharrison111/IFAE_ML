
from xgboost import XGBClassifier
from preprocessing.dataset_io import DatasetHandler
import pandas as pd
from time import perf_counter
import numpy as np

if __name__ == '__main__':

    
    dh = DatasetHandler('configs/training_config.yaml')
    
    sig_dh = DatasetHandler('configs/VLL_signal_config.yaml', scalers=dh.scalers)
    size=-1
    bkg_data = dh.data[:size]
    sig_data = sig_dh.data[:size]
    
    bkg_data['label'] = 0
    sig_data['label'] = 1
    
    sig_lowMass = sig_data.loc[sig_data['sample']=='Esinglet300']
    print(bkg_data.head())
    dh.data = pd.concat([bkg_data,sig_lowMass])
    train_lowMass, test_lowMass = dh.split_per_sample()
    
    df.data = pd.concat([bkg_data,sig_highMass])
    
    train_y = train['label']
    test_y = test['label']
    cols_to_drop = ['weight',
       'sample','scaled_weight', 'label']
    
    train.drop(columns=cols_to_drop, inplace=True)
    test.drop(columns=cols_to_drop, inplace=True)
    
    print(len(train))
    s = perf_counter()
    print("Making XGBoost")
    xgb_classifier = XGBClassifier(n_estimators=50, learning_rate=1e-2)#, tree_method = "hist")
    l1 = perf_counter()
    print("Fitting XGBoost")
    xgb_classifier.fit(train, train_y)
    l2 = perf_counter()
    print(f"Predicting XGBoost... ({round(l2-l1,2)}s to fit.)")
    predictions = xgb_classifier.predict_proba(test)
    l3 = perf_counter()
    print(f"Finished predicting... Time taken {round(l3-l2,2)}s.")

    sig_probs = [pred[1] for pred in predictions]
    bkg_index = np.where(test_y==0)[0]
    sig_index = np.where(test_y==1)[0]
    bkg_pred = np.array(sig_probs)[bkg_index]
    sig_pred = np.array(sig_probs)[sig_index]
    
    import matplotlib.pyplot as plt
    num_bins=25
    bins = np.linspace(0,1,num_bins)
    bkg_vals, bkg_bins, _ = plt.hist(bkg_pred, label='0',bins=bins, density=True,histtype='step')
    vals, bins, _ = plt.hist(sig_pred, label='1', bins=bins, density=True,histtype='step')
    vals = vals/sum(vals)
    bkg_vals = bkg_vals/sum(bkg_vals)
    plt.xlabel('Score')
    plt.ylabel('Counts')
    plt.legend()
    plt.savefig(f'outputs/XGBoost/test_hist{size}_2.png')
    
  
    def get_chi2distance(x,y):
        ch2 = np.nan_to_num(((x-y)**2)/(x+y), copy=True, nan=0.0, posinf=None, neginf=None)
        ch2 = 0.5 * np.sum(ch2)
        return ch2

    chi2 = get_chi2distance(vals, bkg_vals)
    print(f"Chi2 value: {chi2}")
    
    
    #from sklearn.metrics import f1_score
    #f1 = f1_score(test_y, predictions)
    #print(f"F1 : {f1}")