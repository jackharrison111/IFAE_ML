import os
import argparse
from set_regions import define_regions
from utils._utils import find_root_files
import math

if __name__ == '__main__':
    

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--submit",default=False, help="Choose whether to submit the jobs or not", type=bool)
    parser.add_argument("-m", "--mergeTree",default=False, help="Choose whether to merge all to one tree or not.", type=bool)
    parser.add_argument("-p", "--parallel",default=False, help="Choose whether to submit the jobs in parallel", type=bool)
    args = parser.parse_args()

    parallel = args.parallel
    #parallel = False
    
    total_files = -1

    
    
    #If total_files = -1, get all the files in the folder?

    
    split_amount = 25
    memory = 16
    SYSTEMATICS = False
    
    USE_SINGLE_TREE = args.mergeTree

    if SYSTEMATICS:
        USE_SINGLE_TREE = False


    
    
    os.chdir("/nfs/pic.es/user/j/jharriso/IFAE_ML")

    
    job_name = "CombineNom_2009"
    
    
    #base_dir = '/data/at3/common/multilepton/VLL_production/evaluations'
    #base_files = '/data/at3/common/multilepton/VLL_production/evaluations/Eval_Nominal'  #also outdir

    #base_dir = '/data/at3/common/multilepton/SystProduction/evaluations/nominal'
    #base_files = '/data/at3/common/multilepton/SystProduction/evaluations/Eval_Nominal'

    base_dir = '/data/at3/common/multilepton/FinalSystProduction/evaluations/nominal'
    base_files = '/data/at3/common/multilepton/FinalSystProduction/evaluations/Eval_Nominal_v2'

    #Check this in combine_single_nominal.py too
    ntuplePathIn = '/data/at3/common/multilepton/FinalSystProduction/nominal'
    

    #Change this for each sys folder
    if SYSTEMATICS:
        base_dir = '/data/at3/common/multilepton/FinalSystProduction/evaluations/Sys1'
        base_files = '/data/at3/common/multilepton/FinalSystProduction/evaluations/Eval_Sys1'
        
    sys_base_dir = '/data/at3/common/multilepton/FinalSystProduction/Sys1'
    
    flavour = "long"
    
    scriptdir = f"evaluate/jobs/{job_name}"
    if not os.path.exists(scriptdir):
        os.makedirs(scriptdir)
        
    if not os.path.exists(os.path.join(scriptdir,'logs')):
        os.makedirs(os.path.join(scriptdir,'logs'))
    if not os.path.exists(os.path.join(scriptdir,'outs')):
        os.makedirs(os.path.join(scriptdir,'outs'))
    if not os.path.exists(os.path.join(scriptdir,'errs')):
        os.makedirs(os.path.join(scriptdir,'errs'))

    if total_files==-1:
        #Find a way to get all of the total files
        files = find_root_files(ntuplePathIn, '', [])
        total_files = len(files)
        print(f"Found {total_files} to run over.")
        rounded_total = math.ceil(total_files / split_amount) * split_amount
        total_files = rounded_total
        print(f"Rounded to {total_files} to run over.")
        
    if parallel:
        
        # Run one job for each split of files say 0-100 etc 
        
        s = 0
        for i in range(split_amount, total_files, split_amount):
            
            sh_name = os.path.join(scriptdir,f"{job_name}_{s}_{i}.sh")
            execute = open(sh_name, "w")
            execute.write('#!/bin/bash \n')
            #execute.write('export PATH="/data/at3/scratch3/jharrison/miniconda3/envs/ML_env/bin/:$PATH" \n')
            execute.write('export PATH="/data/at3/common/multilepton/miniforge3/envs/ML_env/bin/:$PATH" \n')
            execute.write('cd /nfs/pic.es/user/j/jharriso/IFAE_ML\n')               
            execute.write('eval "$(conda shell.bash hook)"\n')
            execute.write('mamba activate ML_env\n')
            
            if USE_SINGLE_TREE:
                func = f"python evaluate/combine_single_nominal.py -d {base_dir} -o {base_files} --First {s} --Last {i}"
            elif SYSTEMATICS:
                func = f"python evaluate/combine_systematics.py -d {base_dir} -o {base_files} --First {s} --Last {i} -r {sys_base_dir}"
            else:
                func = f"python evaluate/combine_score_branches.py -d {base_dir} -b {base_files} --First {s} --Last {i}"  
            
            execute.write(func)
            execute.write(' \n')
            execute.write('conda deactivate \n')
            execute.close()
            print(f"chmodding: {sh_name}")
            os.system(f"chmod +x {sh_name}")
            s = i
            
            
    else:
        
        #Make the executable file
        sh_name = os.path.join(scriptdir,f"{job_name}.sh")
        execute = open(sh_name, "w")
        execute.write('#!/bin/bash \n')
        #execute.write('export PATH="/data/at3/scratch3/jharrison/miniconda3/envs/ML_env/bin/:$PATH" \n')
        execute.write('export PATH="/data/at3/common/multilepton/miniforge3/envs/ML_env/bin/:$PATH" \n')
        execute.write('cd /nfs/pic.es/user/j/jharriso/IFAE_ML\n')               
        execute.write('eval "$(conda shell.bash hook)"\n')
        execute.write('mamba activate ML_env\n')
        
        if USE_SINGLE_TREE:
            func = f"python evaluate/combine_single_nominal.py -d {base_dir} -o {base_files}"
        elif SYSTEMATICS:
            func = f"python evaluate/combine_systematics.py -d {base_dir} -o {base_files} -r {sys_base_dir}"
        else:
            func = f"python evaluate/combine_score_branches.py -d {base_dir} -b {base_files}"  
            
        #func = f"python evaluate/combine_score_branches.py -d {base_dir} -b {base_files}"  
        #os.system(func)
        execute.write(func)

        execute.write(' \n')
        execute.write('conda deactivate \n')
        execute.close()
        print(f"chmodding: {sh_name}")
        os.system(f"chmod +x {sh_name}")

                
    #Make the submit file and submit it
    condor = open(os.path.join(scriptdir,"condor_submit.sub"), "w")
    name = f"name  = {job_name}\n"
    junk = f"executable  = $(name)\n"
    batch_name = f"JobBatchName = {job_name}\n"

    condor.write(junk)
    condor.write(batch_name)
        

        
    junk1 = f"output  =  evaluate/jobs/{job_name}/outs/$Fnx(name).out\n"
    junk2 = f"log  =  evaluate/jobs/{job_name}/logs/$Fnx(name).log\n"
    junk3 = f"error  =  evaluate/jobs/{job_name}/errs/$Fnx(name).err\n"


    condor.write(junk1)
    condor.write(junk2)
    condor.write(junk3)
    condor.write("getenv = True\n")
    condor.write('+flavour="long"\n')
    condor.write('request_cpus = 1\n')

    condor.write(f"request_memory = {memory} GB\n")
    condor.write('on_exit_remove          = (ExitBySignal == False) && (ExitCode == 0)\n')
    condor.write('requirements = !regexp("AMD EPYC 7452",CPU_MODEL)\n')
    condor.write('max_retries = 1\n')

    condor.write(f"queue name matching files (evaluate/jobs/{job_name}/*.sh)\n")
    condor.close()
    
    if args.submit:
        os.system(f"condor_submit {os.path.join(scriptdir, 'condor_submit.sub')}")

    

