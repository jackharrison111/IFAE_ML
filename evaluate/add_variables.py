""" pyROOT script based on RDataFrame to add a new branch from an existing branch. Input a root file, creates a new rootfile. Variables to change are hardcoded """
#Setting lsetup "root 6.24.06-x86_64-centos7-gcc8-opt"

import ROOT, sys, os

'''
ROOT.EnableImplicitMT() #enable MT for faster computation


#Use 

opts = ROOT.RDF.RSnapshotOptions()
opts.fLazy = False
#opts.fMode="UPDATE" #option to update rootfiles
opts.fMode = "RECREATE"
'''


#Bring the CPP function implementations
#ROOT.gInterpreter.Declare('#include "evaluate/functions.h"')



def DFUpdate(df, var_list, new_var, def_var, add_to_list=True):
    df = df.Define(new_var,def_var)
    if add_to_list:
        var_list.append(new_var)
    return df




def RunEverything(_rootfilename, _variablestoadd):
    
    #get the list of trees to apply the changes
    tree_list = []
    tmpfile = ROOT.TFile(_rootfilename)
    
    nominal = tmpfile.Get('nominal')
    for key in nominal.GetListOfBranches():
        #if key.ReadObj().ClassName()=="TTree":
        tree_list.append(key.GetName())
    print(tree_list[-1])
    
    rdf = ROOT.RDataFrame(nominal)
    
    #rdf = rdf.Define('TestVar', 'functions()')
    
    
    
    #Make the variables and add them back in
    
    
    
            
    '''
    #Loop over the trees to apply the changes
    for ichain in tree_list:
        #initialize Chain
        fletwood = ROOT.TChain(ichain)
        fletwood.Add(_rootfilename)
        fletwood.ls
        rdf = ROOT.RDataFrame(fletwood)
        
        
        #Define the branches
        #should check that branches exist
        for originalvar in _variablestoadd.keys():
            
            
            rdf = rdf.Define(_variablestoadd[originalvar],originalvar)
        #save to a new file or modify the existing one
        rootfilename_ = _rootfilename.replace(".root","_modified.root")
        if tree_list.index(ichain)==0:
            rdf.Snapshot(ichain,rootfilename_)
        else:
            rdf.Snapshot(ichain,rootfilename_,"",opts) 
    '''

if __name__ == "__main__":
    
    import argparse
    
    parser = argparse.ArgumentParser("Running VAE evaluation")
    
    parser.add_argument("-i","--inputDir",action="store", help="Set the input config to use, default is 'configs/training_config.yaml'", default=None, required=False)
    
    args = parser.parse_args()
    input_dir = args.inputDir
    
    input_dir = '/data/at3/scratch3/multilepton/VLL_production/nominal/mc16a/364140.root'
    
    
    variabledictionary = { #original -> newname
        "DeltaEta_lep": "DeltaEta_VHlep",
        "DeltaPhi_lep": "DeltaPhi_VHlep"}

    #RunEverything(input_dir,variabledictionary)



