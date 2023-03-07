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
        
        
    def find_Z_pairs_1Z(self, df):
        pairings=  {
                'Mll01' : 'Mll23', 
                'Mll02':'Mll13',
                'Mll03':'Mll12',
                'Mll12':'Mll03',
                'Mll13':'Mll02',
                'Mll23':'Mll01'
                 }
        leptons = {
                'Mll01' : (0,1), 
                'Mll02':(0,2),
                'Mll03':(0,3),
                'Mll12':(1,2),
                'Mll13':(1,3),
                'Mll23':(2,3)
                 }
        cols = ['Mll01','Mll02','Mll03','Mll12','Mll13','Mll23']
        cols+=['lep_ID_0','lep_ID_1','lep_ID_2','lep_ID_3']
        cols+= ['best_Zllpair','other_Zllpair','best_mZll','other_mZll']
        
        mll_columns = list(pairings.keys())
        if len(df) == 0:
            return df
        
        for i in range(len(df)):
            best_pair = None
            best_mass = None
            for col in mll_columns:
                if abs(df.loc[i,col]-91.2e3)<10e3 and abs(df.loc[i, pairings[col]]-91.2e3)>10e3:
                    if best_pair is None:
                        best_pair = col
                        best_mass = df.loc[i,col]
                    else:
                        if abs(df.loc[i,col]-91.2e3) < abs(best_mass-91.2e3):
                            best_pair = col
                            best_mass = df.loc[i,col]

            df.loc[i, 'best_Zllpair'] = best_pair
            df.loc[i, 'best_mZll'] = best_mass
            df.loc[i, 'other_Zllpair'] = pairings.get(best_pair,None)
            if best_pair is None:
                df.loc[i, 'other_mZll'] = None
            else:
                df.loc[i, 'other_mZll'] = df.loc[i, pairings[best_pair]]

        print(f"Found {len(df.loc[df['other_mZll'].isna()])} events that don't match a 1Z event. Dropping them!")
        df.dropna(subset=['best_Zllpair','best_mZll','other_Zllpair','other_mZll'], inplace=True)
        
        return df
    
    def find_Z_pairs_2Z(self, df):
        pairings=  {
                'Mll01' : 'Mll23', 
                'Mll02':'Mll13',
                'Mll03':'Mll12',
                'Mll12':'Mll03',
                'Mll13':'Mll02',
                'Mll23':'Mll01'
                 }
        leptons = {
                'Mll01' : (0,1), 
                'Mll02':(0,2),
                'Mll03':(0,3),
                'Mll12':(1,2),
                'Mll13':(1,3),
                'Mll23':(2,3)
                 }
        cols = ['Mll01','Mll02','Mll03','Mll12','Mll13','Mll23']
        cols+=['lep_ID_0','lep_ID_1','lep_ID_2','lep_ID_3']
        cols+= ['best_Zllpair','other_Zllpair','best_mZll','other_mZll']
        
        mll_columns = list(pairings.keys())
        for i in range(len(df)):
            best_pair = None
            best_mass = None
            for col in mll_columns:
                if abs(df.loc[i,col]-91.2e3)<10e3 and abs(df.loc[i, pairings[col]]-91.2e3)<10e3:
                    if best_pair is None:
                        if abs(df.loc[i,col]-91.2e3) < abs(df.loc[i, pairings[col]]-91.2e3):
                            best_pair = col
                            best_mass = df.loc[i,col]
                        else:
                            best_pair = pairings[col]
                            best_mass = df.loc[i,pairings[col]]
                    else:
                        if min(abs(df.loc[i,col]-91.2e3),abs(df.loc[i, pairings[col]]-91.2e3)) < abs(best_mass-91.2e3):
                            if abs(df.loc[i,col]-91.2e3) < abs(df.loc[i, pairings[col]]-91.2e3):
                                best_pair = col
                                best_mass = df.loc[i,col]
                            else:
                                best_pair = pairings[col]
                                best_mass = df.loc[i,pairings[col]]

            df.loc[i, 'best_Zllpair'] = best_pair
            df.loc[i, 'best_mZll'] = best_mass
            df.loc[i, 'other_Zllpair'] = pairings.get(best_pair,None)
            if best_pair is None:
                df.loc[i, 'other_mZll'] = None
            else:
                df.loc[i, 'other_mZll'] = df.loc[i, pairings[best_pair]]

        print(f"Found {len(df.loc[df['other_mZll'].isna()])} events that don't match a 2Z event. Dropping them!")
        df.dropna(subset=['best_Zllpair','best_mZll','other_Zllpair','other_mZll'], inplace=True)
        
        return df
    
    '''
    Function to find the lepton pairings in a 0Z, 0b, 2SFOS region
    Strategy is to either pair by flavour (if 2e2m) or to pair
    by minimising the product of Mll**2. 
    Then fill the final output variables with a 'closest' Z
    '''
    def find_pairings_0Z_2SFOS(self, df):
    
        pairings=  {
            'Mll01' : 'Mll23', 
            'Mll02':'Mll13',
            'Mll03':'Mll12',
            'Mll12':'Mll03',
            'Mll13':'Mll02',
            'Mll23':'Mll01'
             }

        #Loop over dataframe
        for i in tqdm(range(len(df))):

            min_mass = None
            min_pair = None

            #For the 2e2m case
            if df.loc[i,'quadlep_type'] == 3:
                #If SFOS:
                if df.loc[i,'lep_ID_0'] == -df.loc[i,'lep_ID_1']:
                    #Pair into the closest Z mass
                    #if df.loc[i,'Mll01'] > df.loc[i,pairings['Mll01']]:
                    if abs(df.loc[i,'Mll01']-91.2e3)<abs(df.loc[i,pairings['Mll01']]-91.2e3):
                        min_pair = ('Mll01', pairings['Mll01'])
                    else:
                        min_pair =  (pairings['Mll01'],'Mll01')

                elif df.loc[i,'lep_ID_0'] == -df.loc[i,'lep_ID_2']:

                    #if df.loc[i,'Mll02'] > df.loc[i,pairings['Mll02']]:
                    if abs(df.loc[i,'Mll02']-91.2e3)<abs(df.loc[i,pairings['Mll02']]-91.2e3):
                        min_pair = ('Mll02', pairings['Mll02'])
                    else:
                        min_pair =  (pairings['Mll02'],'Mll02')

                elif df.loc[i,'lep_ID_0'] == -df.loc[i,'lep_ID_3']:

                    #if df.loc[i,'Mll03'] > df.loc[i,pairings['Mll03']]:
                    if abs(df.loc[i,'Mll03']-91.2e3)<abs(df.loc[i,pairings['Mll03']]-91.2e3):
                        min_pair = ('Mll03', pairings['Mll03'])
                    else:
                        min_pair =  (pairings['Mll03'],'Mll03')

            #For the 4e and 4m case
            elif df.loc[i,'quadlep_type'] in [1,5]:
                for mll, pair in pairings.items():
                    #Only count pairings if they're oppositely charged
                    if df.loc[i,f"lep_ID_{mll[-2]}"]!= -df.loc[i,f"lep_ID_{mll[-1]}"]:
                        continue
                    msqr = pow(df.loc[i,mll],2) * pow(df.loc[i,pair],2)
                    if min_mass is None or msqr < min_mass:
                        min_mass = msqr
                        #if df.loc[i, mll] > df.loc[i,pair]:
                        if abs(df.loc[i,mll]-91.2e3)<abs(df.loc[i,pair]-91.2e3):
                            min_pair = (mll, pair)
                        else:
                            min_pair = (pair, mll)
            else:
                print("Found non-2SFOS event!")

            df.loc[i,'best_Zllpair'] = min_pair[0]
            df.loc[i,'other_Zllpair'] = min_pair[1]
            df.loc[i,'best_mZll'] = df.loc[i,min_pair[0]]
            df.loc[i,'other_mZll'] = df.loc[i,min_pair[1]]

        return df
    
    def find_Z_pairs_0Z_1SFOS(self, df):
        pairings=  {
                'Mll01' : 'Mll23', 
                'Mll02':'Mll13',
                'Mll03':'Mll12',
                'Mll12':'Mll03',
                'Mll13':'Mll02',
                'Mll23':'Mll01'
                 }
        leptons = {
                'Mll01' : (0,1), 
                'Mll02':(0,2),
                'Mll03':(0,3),
                'Mll12':(1,2),
                'Mll13':(1,3),
                'Mll23':(2,3)
                 }
        cols = ['Mll01','Mll02','Mll03','Mll12','Mll13','Mll23']
        cols+=['lep_ID_0','lep_ID_1','lep_ID_2','lep_ID_3']
        cols+= ['best_Zllpair','other_Zllpair','best_mZll','other_mZll']
        
        mll_columns = list(pairings.keys())
        for i in range(len(df)):
            best_pair = None
            best_mass = None
            for col in mll_columns:
                if abs(df.loc[i,col]-91.2e3)>10e3 and df.loc[i,f"lep_ID_{leptons[col][0]}"]==-df.loc[i,f"lep_ID_{leptons[col][1]}"] and df.loc[i,f"lep_ID_{leptons[pairings[col]][0]}"]!=-df.loc[i,f"lep_ID_{leptons[pairings[col]][1]}"]:
                    if best_pair is None:
                        best_pair = col
                        best_mass = df.loc[i,col]
                    else:
                        if abs(df.loc[i,col]-91.2e3) < abs(best_mass-91.2e3):
                            best_pair = col
                            best_mass = df.loc[i,col]

            df.loc[i, 'best_Zllpair'] = best_pair
            df.loc[i, 'best_mZll'] = best_mass
            df.loc[i, 'other_Zllpair'] = pairings.get(best_pair,None)
            if best_pair is None:
                df.loc[i, 'other_mZll'] = None
            else:
                df.loc[i, 'other_mZll'] = df.loc[i, pairings[best_pair]]

        print(f"Found {len(df.loc[df['other_mZll'].isna()])} events that don't match a 0Z 1SFOS event. Dropping them!")
        df.dropna(subset=['best_Zllpair','best_mZll','other_Zllpair','other_mZll'], inplace=True)
        return df

    
    
    def find_Z_pairs_1Z_1SFOS(self, df):
        pairings=  {
                'Mll01' : 'Mll23', 
                'Mll02':'Mll13',
                'Mll03':'Mll12',
                'Mll12':'Mll03',
                'Mll13':'Mll02',
                'Mll23':'Mll01'
                 }
        leptons = {
                'Mll01' : (0,1), 
                'Mll02':(0,2),
                'Mll03':(0,3),
                'Mll12':(1,2),
                'Mll13':(1,3),
                'Mll23':(2,3)
                 }
        cols = ['Mll01','Mll02','Mll03','Mll12','Mll13','Mll23']
        cols+=['lep_ID_0','lep_ID_1','lep_ID_2','lep_ID_3']
        cols+= ['best_Zllpair','other_Zllpair','best_mZll','other_mZll']
        
        mll_columns = list(pairings.keys())
        for i in range(len(df)):
            best_pair = None
            best_mass = None
            for col in mll_columns:
                if abs(df.loc[i,col]-91.2e3)<10e3 and df.loc[i,f"lep_ID_{leptons[col][0]}"]==-df.loc[i,f"lep_ID_{leptons[col][1]}"] and df.loc[i,f"lep_ID_{leptons[pairings[col]][0]}"]!=-df.loc[i,f"lep_ID_{leptons[pairings[col]][1]}"]:
                    if best_pair is None:
                        best_pair = col
                        best_mass = df.loc[i,col]
                    else:
                        if abs(df.loc[i,col]-91.2e3) < abs(best_mass-91.2e3):
                            best_pair = col
                            best_mass = df.loc[i,col]

            df.loc[i, 'best_Zllpair'] = best_pair
            df.loc[i, 'best_mZll'] = best_mass
            df.loc[i, 'other_Zllpair'] = pairings.get(best_pair,None)
            if best_pair is None:
                df.loc[i, 'other_mZll'] = None
            else:
                df.loc[i, 'other_mZll'] = df.loc[i, pairings[best_pair]]

        print(f"Found {len(df.loc[df['other_mZll'].isna()])} events that don't match a 1Z 1SFOS event. Dropping them!")
        df.dropna(subset=['best_Zllpair','best_mZll','other_Zllpair','other_mZll'], inplace=True)
        return df
    
    #Function for finding bestZll in 1Z, 0b, 2SFOS
    def find_bestZll_pair(self, df):
        
        if len(df) == 0:
            return df
        
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
        
        if len(df) == 0:
            return df
    
        df['best_mZll'] = [df.loc[i,col] for i, col in enumerate(df['best_Zllpair'])]
        df['other_mZll'] = [df.loc[i,col] for i, col in enumerate(df['other_Zllpair'])]

        return df
    
    
    def calc_4lep_pTll(self, df):
        
        if len(df) == 0:
            return df
        
        best_worst = {'best_Zllpair' : 'best_ptZll',
                    'other_Zllpair': 'other_ptZll'}
        
        for pair_choice, output_col in best_worst.items():
            df['l0'] = df[pair_choice].str[-2]
            df['l1'] = df[pair_choice].str[-1]

            for i, (id0,id1) in enumerate(zip(df['l0'],df['l1'])):

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
        
        if len(df) == 0:
            return df
        
        df['l0'] = df['best_Zllpair'].str[-2]
        df['l1'] = df['best_Zllpair'].str[-1]


        for i, (id0,id1) in enumerate(zip(df['l0'],df['l1'])):

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
        