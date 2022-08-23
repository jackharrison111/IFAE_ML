'''
Class to perform the splitting of the dataset for train/test.
Class should be used after the dataset has been cleaned, and
produce outputs ready to be passed to a data_set class.

Jack Harrison 23/08/2022
'''

class DatasetSplitter():
    
    def __init__(self, df):
        
        self.data = df
        ...
        
        
    def split_per_sample(self, df, fraction=0.1):
        
        self.test_data = pd.DataFrame()
        print("Length of dataset: ", len(df))
        
        for sample in list(df['sample'].unique()):
            sample_data = df.loc[df['sample'] == sample]
            if len(sample_data) == 0:
                continue
            if len(sample_data) * fraction < 1:
                continue
                
            train, test = train_test_split(sample_data, test_size=fraction)
            
            self.test_data = pd.concat([self.test_data, test])
            self.data.drop(index=test.index.values, axis=0, inplace=True)
            
            print(f"Split {sample} by removing {len(test_data)} events.")
        
        print(f"Removed: {len(self.test_data}")
        print(f"Remaining length: {len(self.data)}")
                          
        return self.data, self.test_data