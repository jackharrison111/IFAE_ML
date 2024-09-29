'''
Script to combine several root files with different nominal trees
into a single set of files with one nominal tree and branches for 
each variable.

Note: Memory consumption is important here.
Make sure you check that you do not exhaust memory (I've not added
many checks!)

- Script uses the first region in the region list to check if there
  are any missing files in any of the folders. 

Jack Harrison 09/01/2024
'''

import time 
import uproot
import pandas as pd
import argparse
import os
import gc
import tracemalloc

# For the systematics, merge each one individually and then HADD the results?


def merge_files(file_list, tree, merge_col='eventNumber'):
    
    start = time.perf_counter()
    tracemalloc.start()
    
    with uproot.open(file_list[0]+f":{tree}") as f:
        events = f.arrays(library="pd")
        
    l1 = time.perf_counter()
    print("Time to open first file: ", round(l1-start,2), " s.")
        
    for i, file in enumerate(file_list[1:]):
        print(f"Merging region {i} / {len(file_list[1:])}")
        lap2 = time.perf_counter()
        
        with uproot.open(file+f":{tree}") as f:
            new_events = f.arrays(library="pd")
            lap3 = time.perf_counter()
            print(f"--- Retrieved events in {round(lap3-lap2,2)}s.")
            events = pd.merge(
                events,
                new_events,
                how="inner",
                on=merge_col,
                sort=False,
                copy=False,
                validate="one_to_one"
            )
            lap4 = time.perf_counter()
            print(f"--- Merged events in {round(lap4-lap3,2)}s.")
            #print("\n--- Memory usage: ---")
            #events.info(verbose = False, memory_usage = 'deep')
            #print("---------------------\n")
            print(f"Time for region: {round(lap4-lap2,2)}s.")
        
    print(f"Finished loop. Total time: {round(lap4-start,2)}s.")
    cur, peak = tracemalloc.get_traced_memory()
    print(f"Current mem: {cur*1e-6}MB, Peak: {peak*1e-6}MB")
    tracemalloc.stop()
    return events


def needs_reevaluating(nom_filename, eval_filename):

    #If no file already
    if not os.path.exists(eval_filename):
        print("Not found eval file, evaluating!")
        return True

    #If file
    try:
        nom_f = uproot.open(nom_filename)
    except:
        print("Couldn't open the nominal file, returning True to re-evaluate!")
        return True

    try:
        eval_f = uproot.open(eval_filename)
    except:
        print("Couldn't open the eval file, returning True to re-evaluate!")
        return True

    nom_keys = nom_f.keys()
    eval_keys = eval_f.keys()
    if len(nom_keys) != len(eval_keys):
        print("Got different numbers of trees in nom/eval, re-evaluating!")
        return True

    diff_flag = False
    for key in nom_keys:
        if ";" in key:
            key = key.split(';')[0]
        if nom_f[key].num_entries != eval_f[key].num_entries:
            diff_flag = True
            print("Found different entries in tree:", key)

    if not diff_flag:
        return False
    
    return True



if __name__ == "__main__" :
    
    s = time.time()
    
    parser = argparse.ArgumentParser(description = "Script to merge nominals into one.")
    parser.add_argument("-d", "--directory",help = "Root directory to look for files",required = False)
    parser.add_argument("-o", "--outDir",help = "Out directory to add files to",required = False)
    parser.add_argument("-r", "--rootDir",help = "Actual input directory to look for files",required = False)
    parser.add_argument("-first","--First",action="store", help="Set the first file to run over", 
                        default=-1, required=False, type=int)
    parser.add_argument("-last","--Last",action="store", help="Set the last file to run over", 
                        default=-1, required=False, type=int)
    args = parser.parse_args()
    
    base_dir = args.directory
    out_dir = args.outDir
    root_dir = args.rootDir
    first = args.First
    last = args.Last
    
    #Q0:
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
        '2Z_0b',
        '2Z_1b'
    ]
    #Q2:
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
    

    #ntuplePathIn = '/data/at3/common/multilepton/FinalSystProduction/nominal'
    ntuplePathIn = root_dir

    
    #Use in_files as all the files in the first region folder - INSTEAD USE THE NOMINAL
    in_files = find_root_files(ntuplePathIn, '')

    check_missing = False
    if check_missing:
        print("Running check for missing files...")
        missing_files = 0
        #Add a scan at the beginning that checks for all the right files
        
        rel_filepaths = []
        new_files = []
        new_relpaths = []
        already_present = []
        empty = []
        for file in in_files:
            
            root_file_path = os.path.relpath(file, ntuplePathIn)
            
            for region in region_folders:
                region_file = os.path.join(base_dir,region,root_file_path)
                if not os.path.exists(region_file):
                    if file in empty:
                        break
                    print(f"ERROR:: Couldn't find input evaluation: {region_file}")
                    try:
                        f = uproot.open(file + ':nominal',library="pd")
                        if len(f.keys()) == 0:
                            print(f"CAUSE:: Found no events in file: {file}.")
                            missing_files -=1
                            empty.append(file)
                    except:
                        print("Couldn't open file for some reason... ", file)
                    missing_files+=1
                    
            # Also check if there are any files already there
            outfile = os.path.join(out_dir, root_file_path)
            if not os.path.exists(outfile):
                if file not in empty:
                    new_files.append(file)
                    new_relpaths.append(root_file_path)
            else:
                already_present.append(file)
            
            rel_filepaths.append(root_file_path)
            
        print(f"Found {len(new_files)} missing:")
        for f in new_files:
            print("  '", f, "',")
            
        print(f"Got {len(already_present)} already done")
                    
        #if missing_files > 0:
        #    raise Exception(f"TERMINATING:: {missing_files} missing input files.")
        
        print(f"Finished check, {missing_files} missing files.")
    
    
    
    #Now run the loop for all the files found.
    if not check_missing:
        new_relpaths = [os.path.relpath(f, ntuplePathIn) for f in in_files]
    
    for i, file in enumerate(new_relpaths):
        
        if i < first and first!=-1:
            continue
        if i > last and last!=-1:
            break
        
            
        print(f"Running file {file}. {i} / {len(new_relpaths)}")
        
        
        #print(file)
        
        #TODO: Check if the length of the input file is zero
        #TODO: Check if this is actually needed??
        '''
        try:
            nom_file = uproot.open(os.path.join(ntuplePathIn, file))
            for key in nom_file.keys():
                num_events = nom_file[key].num_entries
                if num_events == 0:
                    print("Input nominal file is empty... Skipping!")
                    continue
        except:
            print("Couldn't open nominal file for some reason... skipping!")
        '''
        nom_file = in_files[i]
        outfile = os.path.join(out_dir, file)
        if os.path.exists(outfile):
            if not needs_reevaluating(nom_file, outfile):
                print("Found a file that doesn't need to be reproduced! Skipping...")
                continue
        else:
            print("Found evaluation file that doesn't exist, evaluating: ", outfile)

        #Make directory if it doesnt exist
        outfile_base = os.path.dirname(outfile)
        if not os.path.exists(outfile_base):
            os.makedirs(outfile_base)
        
        
        merge_list = [os.path.join(base_dir, r, file) for r in region_folders]

        skipper = False
        for file in merge_list:
            if not os.path.exists(file):
                print("Found file that doesn't exist: ", file)
                print("Skipping!")
                skipper = True
        if skipper:
            continue

        trees = uproot.open(merge_list[0]).keys()

        #Now read each of the trees individually and merge them into an output file
        for i, tree in enumerate(trees):

            if ";" in tree:
                tree = tree.split(';')[0]
                
            print(f"Running tree {tree}: {i} / {len(trees)}")
            events = merge_files(merge_list, tree)
            
            tracemalloc.start()
            tree_file_name = outfile.replace('.root', f"_{tree}.root")
            with uproot.recreate(tree_file_name) as rootfile:
                rootfile[tree]=events
            del events
            gc.collect()
            cur, peak = tracemalloc.get_traced_memory()
            print(f"Saved tree file to: {tree_file_name}")
            print(f"Current mem: {cur*1e-6}MB, Peak: {peak*1e-6}MB")
            tracemalloc.stop()

        
        #Hadd all the trees
        all_tree_files = outfile.replace('.root', '_*.root')

        
        print("Hadd-ing all trees to: ", outfile)

        #hadd -f targetfile source1 source2 ...
        os.system(f"hadd -f {outfile} {all_tree_files}")
        os.system(f"rm {all_tree_files}")
        
        print(f"Done file {i} / {len(new_relpaths)}")
        
        
    f = time.time()
    print(f"Finished script. \nTime for whole script: {round(f-s,2)}s.")
