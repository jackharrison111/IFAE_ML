
from xgboost import XGBClassifier
from preprocessing.dataset_io import DatasetHandler

if __name__ == '__main__':

    
    dh = DatasetHandler('configs/training_config.yaml')
    
    sig_dh = DatasetHandler('configs/VLL_signal_config.yaml')
    
    xgb_classifier = XGBClassifier(n_estimators=200, learning_rate=1e-2)
    xgb_classifier.fit(X_svd, y)


    print("The model is ready.")