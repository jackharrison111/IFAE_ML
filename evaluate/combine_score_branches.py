#####################################
# Script to merge all anomaly scores
# from different regions into one set
# of ROOT files.
#
# Does this by making friends of the 
# 'nominal' tree.
#
# Caveats:
# Needs a first set of score ntuples
# with a 'nominal' branch, which we 
# will use as a base. This can then be
# friended with TRexFitter.
#
# Uses 2Z 0b as a base.
#
# Jack Harrison 02/05/2023
#####################################


import ROOT, os
#from ROOT import RDataFrame,RDF
import time
import argparse


def merge(basefile, merge, rename_branch=None ,tree_name='nominal'):
    
    start_time = time.time()
        
    f2 = ROOT.TFile.Open(merge,"OPEN")
    t2=f2.Get(tree_name)
    
    f1 = ROOT.TFile.Open(basefile,"UPDATE")
    t1=f1.Get(tree_name)

    print("Successfully opened input files: ",merge)
    
    destTree = ROOT.TTree("destTree", "Destination Tree")
    
    if rename_branch:
        t2.SetObject(rename_branch, rename_branch)
        
    destTree = t2.CloneTree()
    destTree.Write()
    t1.AddFriend(destTree)
    t1.Write()
    f1.Close()
    f2.Close()

    end_time=time.time()
    print("Took x seconds to run", end_time-start_time)
    
    
    
if __name__ == "__main__" :
    
    s = time.time()
    
    parser = argparse.ArgumentParser(description = "Friender")
    parser.add_argument("-i", "--input",help = "Provide input root file to be modified",required = False)
    parser.add_argument("-m", "--merge",help = "Provide file with new branches",required = False)
    parser.add_argument("-d", "--directory",help = "Root directory to look for files",required = False)
    parser.add_argument("-b", "--base",help = "Base directory to add files to",required = False)
    parser.add_argument("-first","--First",action="store", help="Set the first file to run over", 
                        default=-1, required=False, type=int)
    parser.add_argument("-last","--Last",action="store", help="Set the last file to run over", 
                        default=-1, required=False, type=int)
    args = parser.parse_args()

    in_file = args.input
    merge_file = args.merge
    base_dir = args.directory
    base_files = args.base
    first = args.First
    last = args.Last
    
    #Take a base path for the AllScores (Use 2Z 0b as a base point?)
    #base_dir = '/data/at3/scratch3/multilepton/VLL_production/evaluations'
    #base_files = '/data/at3/scratch3/multilepton/VLL_production/evaluations/AllRegions_InputVars'
    
    region_folders = [
        '0Z_0b_0SFOS',
        '0Z_0b_1SFOS',
        '0Z_0b_2SFOS',
        '1Z_0b_1SFOS',
        '1Z_0b_2SFOS',
        '0Z_1b_0SFOS',
        '0Z_1b_1SFOS',
        '0Z_1b_2SFOS',
        '1Z_1b_1SFOS',
        '1Z_1b_2SFOS',
        '2Z_1b'
    ]
    
    #Adding them all into 1
    region_folders += ['2Z_0b']
    
    region_folders += [
        "Q2_0b",
        "Q2_0b_e",
        "Q2_0b_eu",
        "Q2_0b_u",
        "Q2_1b",
        "Q2_1b_e",
        "Q2_1b_eu",
        "Q2_1b_u",
    ]
    
    from utils._utils import find_root_files
    
    #Use in_files as all the files in the AllSamples folder
    in_files = find_root_files(base_files, '')
    print(in_files)
    print("Running check for missing files...")
    missing_files = 0
    #Add a scan at the beginning that checks for all the right files
    for file in in_files:
        
        root_file_path = os.path.relpath(file, base_files)
        for region in region_folders:
            
            region_file = os.path.join(base_dir,region,root_file_path)
            if not os.path.exists(region_file):
                print(f"ERROR:: Couldn't find file {region_file}")
                missing_files+=1
                
                
    if missing_files > 0:
        raise Exception(f"TERMINATING:: {missing_files} missing input files.")
    
    print("Finished check, no missing files.")
    
    
    #For each file
    
    for i, file in enumerate(in_files):
        
        if i < first and first!=-1:
            continue
        if i > last and last!=-1:
            break
        
        
        #For each score (except the base score), by default use the 2Z 0b as the base score
        root_file_path = os.path.relpath(file, base_files)
        
        
        #Need to get each region file
        for region in region_folders:
            
            # basedir + region + root file path
            region_file = os.path.join(base_dir,region,root_file_path)
            
            print("Merging: ", file, " with ", region_file)
            merge(file, region_file, rename_branch=f"nominal_{region}")
        
        print(f"Done file {i} / {len(in_files)}")
            
    e = time.time()
    print("Time for whole script: ", e-s, "s")
            
