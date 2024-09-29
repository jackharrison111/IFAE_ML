'''
File to contain the class that calculates additional variables
that we do not have in ntuples.

Class should be initialised and then take a dataframe,
and return the dataframe again with the new columns added.

In order to evaluate on data, we want the functions only to care about
what affects the output variables.
Eg. the functions can be charge agnostic as long as it doesn't damage the
possibility of making Z's

Jack Harrison 22/08/22
'''

import ROOT
from tqdm import tqdm
import numpy as np

class VariableMaker():
    
    def __init__(self):
        
        self.pairings =  {
                'Mll01' : 'Mll23', 
                'Mll02':'Mll13',
                'Mll03':'Mll12',
                'Mll12':'Mll03',
                'Mll13':'Mll02',
                'Mll23':'Mll01'
                 }
        
        self.leptons = {
                'Mll01' : (0,1), 
                'Mll02':(0,2),
                'Mll03':(0,3),
                'Mll12':(1,2),
                'Mll13':(1,3),
                'Mll23':(2,3)
                 }
        
        self.num_leptons = 4
        
        #Columns that the code needs to be able to make variables
        self.varcols = ['Mll01','Mll02','Mll03','Mll12','Mll13','Mll23']
        self.varcols += ['lep_ID_0','lep_ID_1','lep_ID_2','lep_ID_3']
        self.varcols += ['quadlep_type', 'total_charge', 'eventNumber']
        self.varcols += [f"lep_Pt_{id0}" for id0 in range(self.num_leptons)]
        self.varcols += [f"lep_Eta_{id0}" for id0 in range(self.num_leptons)]
        self.varcols += [f"lep_Phi_{id0}" for id0 in range(self.num_leptons)]
        self.varcols += [f"lep_E_{id0}" for id0 in range(self.num_leptons)]
        self.varcols += ['met_met', 'met_phi', 'nJets_OR']
        
        #Names of the outputs of the code
        self.created_cols = ['best_Zllpair','best_mZll','other_Zllpair','other_mZll']
        self.created_cols += ['best_ptZll', 'other_ptZll']
        self.created_cols += ['M3l_high', 'M3l_low']
        self.created_cols += ['MT_ZllMET', 'MT_otherllMET', 'nJets_Continuous']
        
        
    #Used for finding 1Z 2SFOS
    def find_Z_pairs_1Z(self, df):

        if len(df) == 0:
            return df

        df['best_Zllpair'] = "-999"
        df['best_mZll'] = -999
        df['other_Zllpair'] = "-999"
        df['other_mZll'] = -999
        
        # Filter rows based on selections
        valid_rows = df[(df['quadlep_type'] >= 1) & 
                    (df['total_charge'] == 0) & 
                    (~df['quadlep_type'].isin([2, 4]))].copy()


        # Precompute the SFOS and mass differences
        for col, pair in self.pairings.items():
            valid_rows[f'{col}_diff'] = abs(valid_rows[col] - 91.2e3)
            valid_rows[f'{pair}_diff'] = abs(valid_rows[pair] - 91.2e3)

        for i in valid_rows.index:
            best_pair = "-999"
            best_mass = -999
    
            #Loop over the Mlls
            for col, pair in self.pairings.items():
                
                # SFOS check
                if (valid_rows.loc[i, f'lep_ID_{col[-2]}'] != -valid_rows.loc[i, f'lep_ID_{col[-1]}']) and \
                   (valid_rows.loc[i, f'lep_ID_{pair[-2]}'] != -valid_rows.loc[i, f'lep_ID_{pair[-1]}']):
                    continue

                # Check for 1Z
                if valid_rows.loc[i, f'{col}_diff'] < 10e3 and valid_rows.loc[i, f'{pair}_diff'] > 10e3:
                    if best_pair == "-999":
                        best_pair = col
                        best_mass = valid_rows.loc[i, col]
                    else:
                        if valid_rows.loc[i, f'{col}_diff'] < abs(best_mass-91.2e3):
                            best_pair = col
                            best_mass = valid_rows.loc[i,col]

            # Update values for the valid rows
            df.loc[i, 'best_Zllpair'] = best_pair
            df.loc[i, 'best_mZll'] = best_mass
            df.loc[i, 'other_Zllpair'] = self.pairings.get(best_pair, "-999")
            if best_pair == "-999":
                df.loc[i, 'other_mZll'] = -999
            else:
                df.loc[i, 'other_mZll'] = valid_rows.loc[i, self.pairings[best_pair]]
        return df

    
    def find_Z_pairs_2Z(self, df):

        df['best_Zllpair'] = "-999"
        df['best_mZll'] = -999
        df['other_Zllpair'] = "-999"
        df['other_mZll'] = -999

        # Filter rows based on selections
        valid_rows = df[(df['quadlep_type'] >= 1) & 
                    (df['total_charge'] == 0) & 
                    (~df['quadlep_type'].isin([2, 4]))].copy()


        # Precompute the SFOS and mass differences
        for col, pair in self.pairings.items():
            valid_rows[f'{col}_diff'] = abs(valid_rows[col] - 91.2e3)
            valid_rows[f'{pair}_diff'] = abs(valid_rows[pair] - 91.2e3)

        for i in valid_rows.index:
            best_pair = "-999"
            best_mass = -999
    
            #Loop over the Mlls
            for col, pair in self.pairings.items():
                
                # SFOS check
                if (valid_rows.loc[i, f'lep_ID_{col[-2]}'] != -valid_rows.loc[i, f'lep_ID_{col[-1]}']) and \
                   (valid_rows.loc[i, f'lep_ID_{pair[-2]}'] != -valid_rows.loc[i, f'lep_ID_{pair[-1]}']):
                    continue

                # Check for 2Z's
                if valid_rows.loc[i, f'{col}_diff'] < 10e3 and valid_rows.loc[i, f'{pair}_diff'] < 10e3:
                    if best_pair == "-999":
                        if valid_rows.loc[i, f'{col}_diff'] < valid_rows.loc[i, f'{pair}_diff']:
                            best_pair = col
                            best_mass = valid_rows.loc[i, col]
                        else:
                            best_pair = pair
                            best_mass = valid_rows.loc[i, pair]
                    else:
                        if min(valid_rows.loc[i, f'{col}_diff'], valid_rows.loc[i, f'{pair}_diff']) < abs(best_mass - 91.2e3):
                            if valid_rows.loc[i, f'{col}_diff'] < valid_rows.loc[i, f'{pair}_diff']:
                                best_pair = col
                                best_mass = valid_rows.loc[i, col]
                            else:
                                best_pair = pair
                                best_mass = valid_rows.loc[i, pair]

            # Update values for the valid rows
            df.loc[i, 'best_Zllpair'] = best_pair
            df.loc[i, 'best_mZll'] = best_mass
            df.loc[i, 'other_Zllpair'] = self.pairings.get(best_pair, "-999")
            if best_pair == "-999":
                df.loc[i, 'other_mZll'] = -999
            else:
                df.loc[i, 'other_mZll'] = valid_rows.loc[i, self.pairings[best_pair]]
        return df
    
    
    '''
    Function to find the lepton pairings in a 0Z, 2SFOS region
    Strategy is to either pair by flavour (if 2e2m) or to pair
    by minimising the product of Mll**2. 
    Then fill the final output variables with a 'closest' Z
    '''
    def find_pairings_0Z_2SFOS(self, df):

        df['best_Zllpair'] = "-999"
        df['best_mZll'] = -999
        df['other_Zllpair'] = "-999"
        df['other_mZll'] = -999

        # Filter rows based on selections
        valid_rows = df[(df['quadlep_type'] >= 1) & 
                    (df['total_charge'] == 0) & 
                    (~df['quadlep_type'].isin([2, 4]))].copy()


        # Precompute the SFOS and mass differences
        for col, pair in self.pairings.items():
            valid_rows[f'{col}_diff'] = abs(valid_rows[col] - 91.2e3)
            valid_rows[f'{pair}_diff'] = abs(valid_rows[pair] - 91.2e3)

        for i in valid_rows.index:
            best_pair = "-999"
            best_mass = -999

            #Loop over the Mlls
            for col, pair in self.pairings.items():
                
                # SFOS check
                if (valid_rows.loc[i, f'lep_ID_{col[-2]}'] != -valid_rows.loc[i, f'lep_ID_{col[-1]}']) and \
                   (valid_rows.loc[i, f'lep_ID_{pair[-2]}'] != -valid_rows.loc[i, f'lep_ID_{pair[-1]}']):
                    continue

                # Check for 0Z's
                if valid_rows.loc[i, f'{col}_diff'] > 10e3 and valid_rows.loc[i, f'{pair}_diff'] > 10e3:

                    if best_pair == "-999":
                        if valid_rows.loc[i, f'{col}_diff'] < valid_rows.loc[i, f'{pair}_diff']:
                            best_pair = col
                            best_mass = valid_rows.loc[i, col]
                        else:
                            best_pair = pair
                            best_mass = valid_rows.loc[i, pair]
                    else:
                        if min(valid_rows.loc[i, f'{col}_diff'], valid_rows.loc[i, f'{pair}_diff']) < abs(best_mass - 91.2e3):
                            if valid_rows.loc[i, f'{col}_diff'] < valid_rows.loc[i, f'{pair}_diff']:
                                best_pair = col
                                best_mass = valid_rows.loc[i, col]
                            else:
                                best_pair = pair
                                best_mass = valid_rows.loc[i, pair]

            # Update values for the valid rows
            df.loc[i, 'best_Zllpair'] = best_pair
            df.loc[i, 'best_mZll'] = best_mass
            df.loc[i, 'other_Zllpair'] = self.pairings.get(best_pair, "-999")
            if best_pair == "-999":
                df.loc[i, 'other_mZll'] = -999
            else:
                df.loc[i, 'other_mZll'] = valid_rows.loc[i, self.pairings[best_pair]]
                   
        return df
        

    def find_Z_pairs_0Z_1SFOS(self, df):
            
        df['best_Zllpair'] = "-999"
        df['best_mZll'] = -999
        df['other_Zllpair'] = "-999"
        df['other_mZll'] = -999

        # Filter rows based on selections
        valid_rows = df[(df['quadlep_type'] >= 1) & 
                    (df['total_charge'] == 0)].copy()

        # Precompute the SFOS and mass differences
        for col, pair in self.pairings.items():
            valid_rows[f'{col}_diff'] = abs(valid_rows[col] - 91.2e3)
            valid_rows[f'{pair}_diff'] = abs(valid_rows[pair] - 91.2e3)


        for i in valid_rows.index:
            best_pair = "-999"
            best_mass = -999

            #Loop over the Mlls
            for col, pair in self.pairings.items():

                if valid_rows.loc[i, f'{col}_diff'] > 10e3 and valid_rows.loc[i,f"lep_ID_{self.leptons[col][0]}"]==-valid_rows.loc[i,f"lep_ID_{self.leptons[col][1]}"] and valid_rows.loc[i,f"lep_ID_{self.leptons[pair][0]}"]!=-valid_rows.loc[i,f"lep_ID_{self.leptons[pair][1]}"]:
                    if best_pair == "-999":
                        best_pair = col
                        best_mass = valid_rows.loc[i,col]
                    else:
                        if valid_rows.loc[i, f'{col}_diff'] < abs(best_mass-91.2e3):
                            best_pair = col
                            best_mass = valid_rows.loc[i,col]

            # Update values for the valid rows
            df.loc[i, 'best_Zllpair'] = best_pair
            df.loc[i, 'best_mZll'] = best_mass
            df.loc[i, 'other_Zllpair'] = self.pairings.get(best_pair, "-999")
            if best_pair == "-999":
                df.loc[i, 'other_mZll'] = -999
            else:
                df.loc[i, 'other_mZll'] = valid_rows.loc[i, self.pairings[best_pair]]
        return df
            

    
    def find_Z_pairs_1Z_1SFOS(self, df):
        
        for i in range(len(df)):
            
            best_pair = "-999"
            best_mass = -999
            df.loc[i, 'best_Zllpair'] = "-999"
            df.loc[i, 'best_mZll'] = -999
            df.loc[i, 'other_Zllpair'] = "-999"
            df.loc[i, 'other_mZll'] = -999
            
            #Check that there are 4 leptons
            if df.loc[i, 'quadlep_type'] < 1:
                continue
            
            for col, pair in self.pairings.items():
                
                if abs(df.loc[i,col]-91.2e3)<10e3 and df.loc[i,f"lep_ID_{self.leptons[col][0]}"]==-df.loc[i,f"lep_ID_{self.leptons[col][1]}"] and df.loc[i,f"lep_ID_{self.leptons[pair][0]}"]!=-df.loc[i,f"lep_ID_{self.leptons[pair][1]}"]:
                    if best_pair == "-999":
                        best_pair = col
                        best_mass = df.loc[i,col]
                    else:
                        if abs(df.loc[i,col]-91.2e3) < abs(best_mass-91.2e3):
                            best_pair = col
                            best_mass = df.loc[i,col]

            if best_pair == "-999":
                continue
                
            df.loc[i, 'best_Zllpair'] = best_pair
            df.loc[i, 'best_mZll'] = best_mass
            df.loc[i, 'other_Zllpair'] = self.pairings.get(best_pair,"-999")
            if best_pair == "-999":
                df.loc[i, 'other_mZll'] = -999
            else:
                df.loc[i, 'other_mZll'] = df.loc[i, self.pairings[best_pair]]

            df.loc[i,'best_Zllpair'] = str(df.loc[i,'best_Zllpair'])
            df.loc[i,'other_Zllpair'] = str(df.loc[i,'other_Zllpair'])

        return df
    
    
    #Function for finding bestZll in 0Z 0SFOS
    def find_bestZll_pair(self, df):
        for i in range(len(df)):
            
            best_pair = "-999"
            best_mass = -999
            df.loc[i, 'best_Zllpair'] = "-999"
            df.loc[i, 'best_mZll'] = -999
            df.loc[i, 'other_Zllpair'] = "-999"
            df.loc[i, 'other_mZll'] = -999
            
            #Check that there are 4 leptons, Q=0
            if df.loc[i, 'quadlep_type'] < 1:
                continue
            
            for col, pair in self.pairings.items():
                
                #Check for 0SFOS
                if df.loc[i,f"lep_ID_{col[-2]}"]==-df.loc[i,f"lep_ID_{col[-1]}"]:
                    continue
                if df.loc[i,f"lep_ID_{pair[-2]}"]==-df.loc[i,f"lep_ID_{pair[-1]}"]:
                    continue
                
                #Pair by closest to Z. 
                if best_pair == "-999":
                    best_pair = col
                    best_mass = df.loc[i,col]
                else:
                    if abs(df.loc[i,col]-91.2e3) < abs(best_mass-91.2e3):
                        best_pair = col
                        best_mass = df.loc[i,col]
                        
            df.loc[i, 'best_Zllpair'] = best_pair
            df.loc[i, 'best_mZll'] = best_mass
            df.loc[i, 'other_Zllpair'] = self.pairings.get(best_pair,"-999")
            if best_pair == "-999":
                df.loc[i, 'other_mZll'] = -999
            else:
                df.loc[i, 'other_mZll'] = df.loc[i, self.pairings[best_pair]]

            df.loc[i,'best_Zllpair'] = str(df.loc[i,'best_Zllpair'])
            df.loc[i,'other_Zllpair'] = str(df.loc[i,'other_Zllpair'])
            
        return df



    def calc_4lep_pTll(self, df):
        
        if len(df) == 0:
            return df
        
        best_worst = {'best_Zllpair' : 'best_ptZll',
                    'other_Zllpair': 'other_ptZll'}

        for pair_choice, output_col in best_worst.items():

            df[output_col] = -999

            valid_rows = df[df[pair_choice]!="-999"].copy()
            
            valid_rows['l0'] = valid_rows[pair_choice].astype('str').str[-2]
            valid_rows['l1'] = valid_rows[pair_choice].astype('str').str[-1]

            for i in valid_rows.index:
                
                id0 = valid_rows.loc[i,'l0']
                id1 = valid_rows.loc[i,'l1']
                
                lv0 = ROOT.TLorentzVector()
                lv0.SetPtEtaPhiE(valid_rows.loc[i,f"lep_Pt_{id0}"],valid_rows.loc[i,f"lep_Eta_{id0}"],
                                 valid_rows.loc[i,f"lep_Phi_{id0}"],valid_rows.loc[i,f"lep_E_{id0}"])
                lv1 = ROOT.TLorentzVector()
                lv1.SetPtEtaPhiE(valid_rows.loc[i,f"lep_Pt_{id1}"],valid_rows.loc[i,f"lep_Eta_{id1}"],
                                 valid_rows.loc[i,f"lep_Phi_{id1}"],valid_rows.loc[i,f"lep_E_{id1}"])

                df.loc[i, output_col] = (lv0+lv1).Pt()
                
        return df
                
    
    
    def calc_m3l(self, df):
        
        if len(df) == 0:
            return df

        df['M3l_low'] = -999
        df['M3l_high'] = -999

        valid_rows = df[df['best_Zllpair']!="-999"].copy()

        valid_rows['l0'] = valid_rows['best_Zllpair'].astype('str').str[-2]
        valid_rows['l1'] = valid_rows['best_Zllpair'].astype('str').str[-1]

        for i in valid_rows.index:
            
            id0 = valid_rows.loc[i,'l0']
            id1 = valid_rows.loc[i,'l1']

            lv0 = ROOT.TLorentzVector()
            lv0.SetPtEtaPhiE(valid_rows.loc[i,f"lep_Pt_{id0}"],valid_rows.loc[i,f"lep_Eta_{id0}"],
                             valid_rows.loc[i,f"lep_Phi_{id0}"],valid_rows.loc[i,f"lep_E_{id0}"])
            lv1 = ROOT.TLorentzVector()
            lv1.SetPtEtaPhiE(valid_rows.loc[i,f"lep_Pt_{id1}"],valid_rows.loc[i,f"lep_Eta_{id1}"],
                             valid_rows.loc[i,f"lep_Phi_{id1}"],valid_rows.loc[i,f"lep_E_{id1}"])

            sum2 = lv0+lv1

            choices = [0,1,2,3]
            choices.remove(int(id0))
            choices.remove(int(id1))
            mlls = []
            for lep in choices:
                lv3 = ROOT.TLorentzVector()
                lv3.SetPtEtaPhiE(valid_rows.loc[i,f"lep_Pt_{lep}"],valid_rows.loc[i,f"lep_Eta_{lep}"],
                             valid_rows.loc[i,f"lep_Phi_{lep}"],valid_rows.loc[i,f"lep_E_{lep}"])
                mlls.append((sum2+lv3).M())

            df.loc[i, 'M3l_low'] = min(mlls[0],mlls[1])
            df.loc[i, 'M3l_high'] = max(mlls[0],mlls[1])
            
        return df
    

    def calc_m4l(self, df):
        
        if len(df) == 0:
            return df
        
        df['Mllll0123'] = -999

        for i in range(len(df)):
            
            lv0 = ROOT.TLorentzVector()
            lv0.SetPtEtaPhiE(df.loc[i,f"lep_Pt_0"],df.loc[i,f"lep_Eta_0"],
                             df.loc[i,f"lep_Phi_0"],df.loc[i,f"lep_E_0"])

            lv1 = ROOT.TLorentzVector()
            lv1.SetPtEtaPhiE(df.loc[i,f"lep_Pt_1"],df.loc[i,f"lep_Eta_1"],
                             df.loc[i,f"lep_Phi_1"],df.loc[i,f"lep_E_1"])

            lv2 = ROOT.TLorentzVector()
            lv2.SetPtEtaPhiE(df.loc[i,f"lep_Pt_2"],df.loc[i,f"lep_Eta_2"],
                             df.loc[i,f"lep_Phi_2"],df.loc[i,f"lep_E_2"])

            lv3 = ROOT.TLorentzVector()
            lv3.SetPtEtaPhiE(df.loc[i,f"lep_Pt_3"],df.loc[i,f"lep_Eta_3"],
                             df.loc[i,f"lep_Phi_3"],df.loc[i,f"lep_E_3"])

            df.loc[i,'Mllll0123'] = (lv0+lv1+lv2+lv3).M()
            
        return df
    
    
    def make_jets_continuous(self, df):
        
        df['randNumCol'] = np.random.random(size=len(df))
        df['nJets_Continuous'] = df['nJets_OR'] + df['randNumCol']
        df.drop(columns=['randNumCol'], inplace=True)
        return df
    
    
    def get_MTLepMet(self, df):
        
        if len(df) == 0:
            return df
        
        if 'MtLepMet' in df.columns:
            return df
        
        df['MtLepMet'] = -999
        
        for i in range(len(df)):

            
            met = ROOT.TLorentzVector()
            met.SetPtEtaPhiM(df.loc[i,'met_met'], 0, df.loc[i,'met_phi'], 0);

            lv0 = ROOT.TLorentzVector()
            lv0.SetPtEtaPhiE(df.loc[i,f"lep_Pt_0"],df.loc[i,f"lep_Eta_0"],
                             df.loc[i,f"lep_Phi_0"],df.loc[i,f"lep_E_0"])
            lv1 = ROOT.TLorentzVector()
            lv1.SetPtEtaPhiE(df.loc[i,f"lep_Pt_1"],df.loc[i,f"lep_Eta_1"],
                             df.loc[i,f"lep_Phi_1"],df.loc[i,f"lep_E_1"])
            
            lv2 = ROOT.TLorentzVector()
            lv2.SetPtEtaPhiE(df.loc[i,f"lep_Pt_0"],df.loc[i,f"lep_Eta_0"],
                             df.loc[i,f"lep_Phi_0"],df.loc[i,f"lep_E_0"])
            lv3 = ROOT.TLorentzVector()
            lv3.SetPtEtaPhiE(df.loc[i,f"lep_Pt_1"],df.loc[i,f"lep_Eta_1"],
                             df.loc[i,f"lep_Phi_1"],df.loc[i,f"lep_E_1"])

            df.loc[i, 'MtLepMet'] = (lv0+lv1+lv2+lv3+met).Mt()
            
        return df
        
        
    def get_MTLepLepMET(self, df):
 
        best_worst = {'best_Zllpair' : 'MT_ZllMET',
                    'other_Zllpair': 'MT_otherllMET'}
        
        for pair_choice, output_col in best_worst.items():

            df[output_col] = -999

            valid_rows = df[df[pair_choice]!="-999"].copy()
            
            valid_rows['l0'] = valid_rows[pair_choice].astype('str').str[-2]
            valid_rows['l1'] = valid_rows[pair_choice].astype('str').str[-1]

            for i in valid_rows.index:

                id0 = valid_rows.loc[i,'l0']
                id1 = valid_rows.loc[i,'l1']

                met = ROOT.TLorentzVector()
                met.SetPtEtaPhiM(valid_rows.loc[i,'met_met'], 0, valid_rows.loc[i,'met_phi'], 0);

                lv0 = ROOT.TLorentzVector()
                lv0.SetPtEtaPhiE(valid_rows.loc[i,f"lep_Pt_{id0}"],valid_rows.loc[i,f"lep_Eta_{id0}"],
                                 valid_rows.loc[i,f"lep_Phi_{id0}"],valid_rows.loc[i,f"lep_E_{id0}"])
                lv1 = ROOT.TLorentzVector()
                lv1.SetPtEtaPhiE(valid_rows.loc[i,f"lep_Pt_{id1}"],valid_rows.loc[i,f"lep_Eta_{id1}"],
                                 valid_rows.loc[i,f"lep_Phi_{id1}"],valid_rows.loc[i,f"lep_E_{id1}"])

                df.loc[i,output_col] = (lv0+lv1+met).Mt()
            
        return df
        
    def get_Qblind_pairs(self, df):
            
        #Take the closest same flavour pair to a Z mass as the Z pair
        #There will always be a same flavour pair
        for i in range(len(df)):
            
            best_pair = "-999"
            best_mass = -999
            other_pair = "-999"
            other_mass = -999
            
            '''
            #Comment out if running over data-driven estimates?
            if abs(df.loc[i, "total_charge"]) != 2:
                df.loc[i, 'best_Zllpair'] = best_pair
                df.loc[i, 'best_mZll'] = best_mass
                df.loc[i, 'other_Zllpair'] = other_pair
                df.loc[i, 'other_mZll'] = other_mass
                continue
            '''
            for col, pair in self.pairings.items():
                
                #Set the best pair and best mass if the lepton is the same flavour 
                if abs(df.loc[i, f"lep_ID_{col[-2]}"]) == abs(df.loc[i,f"lep_ID_{col[-1]}"]):
                    
                    #Check if the mass is closer to the Z than the best_mass
                    if best_mass == -999 or abs(best_mass - 91.2e3) > abs(df.loc[i,col]):
                        
                        best_mass = df.loc[i, col]
                        best_pair = col
                        other_mass = df.loc[i,pair]
                        other_pair = pair
              
            df.loc[i, 'best_Zllpair'] = best_pair
            df.loc[i, 'best_mZll'] = best_mass
            df.loc[i, 'other_Zllpair'] = other_pair
            df.loc[i, 'other_mZll'] = other_mass
                
            
        return df
