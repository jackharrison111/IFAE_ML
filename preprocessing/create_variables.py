'''
File to contain the class that calculates additional variables
that we do not have in ntuples.

Class should be initialised and then take a dataframe,
and return the dataframe again with the new columns added.

Jack Harrison 22/08/22
'''

import ROOT
from tqdm import tqdm

class VariableMaker():
    
    def __init__(self):
        ...
        
        
    def find_bestZll_pair(self, df):
        
        pairings=  {
            'Mll01' : 'Mll23', 
            'Mll02':'Mll13',
            'Mll03':'Mll12',
            'Mll12':'Mll03',
            'Mll13':'Mll02',
            'Mll23':'Mll01'
             }
        mll_columns = list(pairings.keys())
        df['best_Zllpair'] = (abs(df[mll_columns]-91.2e3)).idxmin(axis=1)
        df['other_Zllpair'] = df['best_Zllpair'].map(pairings)
        df = df.reset_index(drop=True)
        return df
    
    
    def calc_4lep_mZll(self, df):
    
        df['best_mZll'] = [df.loc[i,col] for i, col in enumerate(df['best_Zllpair'])]
        df['other_mZll'] = [df.loc[i,col] for i, col in enumerate(df['other_Zllpair'])]

        return df
    
    
    def calc_4lep_pTll(self, df):
        
        best_worst = {'best_Zllpair' : 'best_ptZll',
                    'other_Zllpair': 'other_ptZll'}
        
        for pair_choice, output_col in best_worst.items():
            print(df.loc[0,pair_choice])
            df['l0'] = df[pair_choice].str[-2]
            df['l1'] = df[pair_choice].str[-1]

            for i, (id0,id1) in tqdm(enumerate(zip(df['l0'],df['l1']))):

                lv0 = ROOT.TLorentzVector()
                lv0.SetPtEtaPhiE(df.loc[i,f"lep_Pt_{id0}"],df.loc[i,f"lep_Eta_{id0}"],
                                 df.loc[i,f"lep_Phi_{id0}"],df.loc[i,f"lep_E_{id0}"])
                lv1 = ROOT.TLorentzVector()
                lv1.SetPtEtaPhiE(df.loc[i,f"lep_Pt_{id1}"],df.loc[i,f"lep_Eta_{id1}"],
                                 df.loc[i,f"lep_Phi_{id1}"],df.loc[i,f"lep_E_{id1}"])

                df.loc[i, output_col] = (lv0+lv1).Pt()

            df.drop(columns=['l0','l1'],inplace=True)
        return df
    
    
    def calc_m3l(self, df):
        
        df['l0'] = df['best_Zllpair'].str[-2]
        df['l1'] = df['best_Zllpair'].str[-1]


        for i, (id0,id1) in tqdm(enumerate(zip(df['l0'],df['l1']))):

            lv0 = ROOT.TLorentzVector()
            lv0.SetPtEtaPhiE(df.loc[i,f"lep_Pt_{id0}"],df.loc[i,f"lep_Eta_{id0}"],
                             df.loc[i,f"lep_Phi_{id0}"],df.loc[i,f"lep_E_{id0}"])
            lv1 = ROOT.TLorentzVector()
            lv1.SetPtEtaPhiE(df.loc[i,f"lep_Pt_{id1}"],df.loc[i,f"lep_Eta_{id1}"],
                             df.loc[i,f"lep_Phi_{id1}"],df.loc[i,f"lep_E_{id1}"])
            sum2 = lv0+lv1

            choices = [0,1,2,3]
            choices.remove(int(id0))
            choices.remove(int(id1))
            mlls = []
            for lep in choices:
                lv3 = ROOT.TLorentzVector()
                lv3.SetPtEtaPhiE(df.loc[i,f"lep_Pt_{lep}"],df.loc[i,f"lep_Eta_{lep}"],
                             df.loc[i,f"lep_Phi_{lep}"],df.loc[i,f"lep_E_{lep}"])
                mlls.append((sum2+lv3).M())
            if mlls[0] > mlls[1]:
                df.loc[i, 'M3l_low'] = mlls[1]
                df.loc[i, 'M3l_high'] = mlls[0]
            else:
                df.loc[i, 'M3l_low'] = mlls[0]
                df.loc[i, 'M3l_high'] = mlls[1]
                
        df.drop(columns=['l0','l1'],inplace=True)
        return df
    
    #Function to get all the member functions
    #TODO
    def return_functions(self):
        ...
        