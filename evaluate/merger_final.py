import ROOT,os
#from ROOT import RDataFrame,RDF
import time
import argparse



def main(infile, merge, rename_branch=None ,tree_name='nominal'):
    
    start_time = time.time()
        
    f2 = ROOT.TFile.Open(merge,"OPEN")
    t2=f2.Get(tree_name)
    
    f1 = ROOT.TFile.Open(infile,"UPDATE")
    t1=f1.Get(tree_name)

    print("Got input files.")
    
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
    
    parser = argparse.ArgumentParser(description = "Friender")
    parser.add_argument("-i", "--input",help = "Provide input root file to be modified",required = False)
    parser.add_argument("-m", "--merge",help = "Provide file with new branches",required = False)
    args = parser.parse_args()

    in_file = args.input
    merge_file = args.merge

    in_file = '/data/at3/scratch3/multilepton/VLL_production/evaluations/364250.root'
    
    merge_file = '/data/at3/scratch3/multilepton/VLL_production/evaluations/364250_score.root'
    
    score_files = [
        '364250_0Z_0b_0SFOS.root',
        '364250_0Z_1b_0SFOS.root',
        '364250_0Z_0b_1SFOS.root',
        '364250_0Z_1b_1SFOS.root',
        '364250_0Z_0b_2SFOS.root',
        '364250_0Z_1b_2SFOS.root',
        '364250_1Z_0b_1SFOS.root',
        '364250_1Z_1b_1SFOS.root',
        '364250_1Z_0b_2SFOS.root',
        '364250_1Z_1b_2SFOS.root',
        '364250_2Z_0b.root',
        '364250_2Z_1b.root',
    ]
    
    root = '/data/at3/scratch3/multilepton/VLL_production/evaluations/'
    scores_files = [root + score_file for score_file in score_files]
    for f in score_files:
        main(in_file, f)