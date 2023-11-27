""" pyROOT script based on RDataFrame to add a new branch from an existing branch. Input a root file, creates a new rootfile. Variables to change are hardcoded """
#Setting lsetup "root 6.24.06-x86_64-centos7-gcc8-opt"

import ROOT, sys, os
ROOT.EnableImplicitMT() #enable MT for faster computation

opts = ROOT.RDF.RSnapshotOptions()
opts.fMode="UPDATE" #option to update rootfiles

def RunEverything(_rootfilename, _variablestoadd):
    
    #get the list of trees to apply the changes
    tree_list = []
    tmpfile = ROOT.TFile(_rootfilename)
    for key in tmpfile.GetListOfKeys():
        if key.ReadObj().ClassName()=="TTree":
            tree_list.append(key.ReadObj().GetName())
            print(tree_list[-1])
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
            
if __name__ == "__main__":
    if sys.argv[-1] == __file__:
        print("call via \"python", __file__, "<path_rootfile_input>\"")
        raise SystemExit
    
    if os.path.exists(sys.argv[-1].replace(".root","_modified.root")):
        print("Output",sys.argv[-1].replace(".root","_modified.root")," exists, don't like it. Good bye")
        raise SystemExit

    variabledictionary = { #original -> newname
        "DeltaEta_lep": "DeltaEta_VHlep",
        "DeltaPhi_lep": "DeltaPhi_VHlep"}

    RunEverything(sys.argv[-1],variabledictionary)




