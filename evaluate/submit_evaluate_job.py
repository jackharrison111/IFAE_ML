'''
Script to evaluate the ML model over a dataset

Takes as input:
- feather config:
    - Used for getting the functions to make the input variables with

'''


import os
import argparse
from set_regions import define_regions
import math 

from utils._utils import find_root_files

if __name__ == '__main__':
    

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--submit",default=False, help="Choose whether to submit the jobs or not", type=bool)
    parser.add_argument("-p", "--parallel",default=False, help="Choose whether to submit the jobs in parallel", type=bool)
    parser.add_argument("-i", "--indices",default=False, help="Choose whether to submit only selected indices", type=bool)
    args = parser.parse_args()
    split = args.parallel
    
    
    os.chdir("/nfs/pic.es/user/j/jharriso/IFAE_ML")

    SYSTS = True
    Q2 = False
    eval_DD = False

    run_selected_files = args.indices
    files_per_job = 5
    
    job_name = "SystFinalSub"
    job_suffix = "Sys4"   #Used to chose the sys group

    job_name += f"_{job_suffix}"
    
    split_amount = 15
    total_files = -1
    

    #Change here to set folder to read from
    ntuplePathIn = f"/data/at3/common/multilepton/FinalSystProduction/{job_suffix}"
    ntuplePathOut  = f"/data/at3/common/multilepton/FinalSystProduction/evaluations/{job_suffix}"
    
    #ntuplePathIn = "/data/at3/common/separi/nominal"
    #ntuplePathOut  = "/data/at3/common/multilepton/VLLemu/evaluations/nominal"

    if 'FinalSystProduction/Sys' in ntuplePathIn:
        SYSTS = True
    
    if Q2:
        train_run_file = 'nf_Q2.yaml'
    else:
        train_run_file = 'nf_NewYields.yaml'
    
    
        
    


    if Q2:
        chosen_regions = ["Q2_0b", "Q2_0b_e", "Q2_0b_eu", "Q2_0b_u",
                      "Q2_1b", "Q2_1b_e", "Q2_1b_eu", "Q2_1b_u"]

    else:
        chosen_regions =  ['0Z_0b_0SFOS', '0Z_0b_1SFOS', '0Z_0b_2SFOS',
                        '1Z_0b_1SFOS', '1Z_0b_2SFOS', '2Z_0b',
                        '0Z_1b_0SFOS', '0Z_1b_1SFOS', '0Z_1b_2SFOS',
                        '1Z_1b_1SFOS', '1Z_1b_2SFOS', '2Z_1b']
    

    regions_file = os.path.join('evaluate/region_settings',train_run_file)
    regions = define_regions(regions_file)
    
    flavour = "long"

    if total_files==-1 and not run_selected_files:
        #Find a way to get all of the total files
        files = find_root_files(ntuplePathIn, '', [])
        total_files = len(files)
        print(f"Found {total_files} to run over.")
        rounded_total = math.ceil(total_files / split_amount) * split_amount
        total_files = rounded_total
        print(f"Rounded to {total_files} to run over.")

    
    if total_files:
        regions['total_files'] = total_files

    for chosen_region in chosen_regions:

        
        region = chosen_region
        vals = regions['regions'][chosen_region]

        if run_selected_files:
            check_outdate = True
            check_reeval = True
            use_file_skipper = True
            
            from evaluate_model_Systs import file_past_date, needs_reevaluating
            files = find_root_files(ntuplePathIn, '', [])
            to_rerun_indices = []
    
            for i, file in enumerate(files):
    
                save_path = file.split(os.path.basename(ntuplePathIn))[1]
                if save_path[0] == '/':
                    save_path = save_path[1:]
                
                outdir = os.path.join(ntuplePathOut, region)
                whole_out_string = os.path.join(outdir, save_path)

                if use_file_skipper:
                    name = '/nfs/pic.es/user/j/jharriso/IFAE_ML/evaluate/needEval.txt'
                    with open(name,'r') as f:
                        files_to_skip = f.readlines()
                    files_to_skip = [(s.split(' ')[0],s.split(' ')[1])for s in files_to_skip]
                    file_to_run = False
                    for f in files_to_skip:
                        if f[0].strip().replace('\n', '') in file and f[1].strip().replace('\n', '') in file:
                            file_to_run=True
                    if not file_to_run:
                        continue
                if check_outdate:
                    if os.path.exists(whole_out_string):
                        #print("Skipping file as outfile already exists")
                        continue
                    if not file_past_date(whole_out_string):
                        #print("Found file that was created before September 1st... Skipping!")
                        continue
                if check_reeval:
                    if not needs_reevaluating(file, whole_out_string):
                        #print("Found file that doesn't need evaluating... skipping!")
                        continue
    
                to_rerun_indices.append(i)
            print(f"For region {region}, found {len(to_rerun_indices)} files. \nIndices to rerun: ", to_rerun_indices)
            print(f"Splitting this over {math.ceil(len(to_rerun_indices)/files_per_job)} jobs...")

        

        scriptdir = f"evaluate/jobs/{job_name}/{region}"
        if not os.path.exists(scriptdir):
            os.makedirs(scriptdir)
            
        if not os.path.exists(os.path.join(scriptdir,'logs')):
            os.makedirs(os.path.join(scriptdir,'logs'))
        if not os.path.exists(os.path.join(scriptdir,'outs')):
            os.makedirs(os.path.join(scriptdir,'outs'))
        if not os.path.exists(os.path.join(scriptdir,'errs')):
            os.makedirs(os.path.join(scriptdir,'errs'))
            
            
        even_path = os.path.join(regions['even_base_dir'],vals['even_path'])
        odd_path = os.path.join(regions['odd_base_dir'],vals['odd_path'])
        
        feather_conf = f'configs/feather_configs/10GeV/{region}.yaml'
            
        if Q2:
            feather_conf = f'configs/feather_configs/10GeV/Q2/{region}.yaml'
        
        if split:
            
            if split_amount and split_amount != -1:
                regions['split_amount'] = split_amount
            
            s = 0
            for i in range(regions['split_amount'], regions['total_files']+2*regions['split_amount'], regions['split_amount']):
                
                #Make the executable file
                sh_name = os.path.join(scriptdir,f"{region}_{s}_{i}.sh")
                execute = open(sh_name, "w")
                execute.write('#!/bin/bash \n')
                #execute.write('export PATH="/data/at3/scratch3/jharrison/miniforge3/envs/ML_env/bin/:$PATH" \n')
                execute.write('export PATH="/data/at3/common/multilepton/miniforge3/envs/ML_env/bin/:$PATH" \n')
                
                execute.write('cd /nfs/pic.es/user/j/jharriso/IFAE_ML\n')               
                execute.write('eval "$(conda shell.bash hook)"\n')
                execute.write('mamba activate ML_env\n')
                

                func = f"python evaluate/evaluate_model_v4.py -r {region} -e {even_path} -o {odd_path} --First {s} --Last {i} -f {feather_conf} -ni {ntuplePathIn} -no {ntuplePathOut}"
                
                
                if eval_DD:
                    func = f"python evaluate/evaluate_model_DDqmisID.py -r {region} -e {even_path} -o {odd_path} -f {feather_conf} -ni {ntuplePathIn} -no {ntuplePathOut} --First {s} --Last {i}"
                    
                if SYSTS:
                    func = f"python evaluate/evaluate_model_Systs.py -r {region} -e {even_path} -o {odd_path} -f {feather_conf} -ni {ntuplePathIn} -no {ntuplePathOut} --First {s} --Last {i}"
                

                execute.write(func)

                execute.write(' \n')
                execute.write('conda deactivate \n')
                execute.close()
                print(f"chmodding: {sh_name}")
                os.system(f"chmod +x {sh_name}")
                s = i

        elif run_selected_files:
            
            #Loop over indices in the list
            for i in range(0, len(to_rerun_indices), files_per_job):
                inds = to_rerun_indices[i:i + files_per_job]
                #Make the executable file
                sh_name = os.path.join(scriptdir,f"{region}_{int(i/files_per_job)}.sh")
                execute = open(sh_name, "w")
                execute.write('#!/bin/bash \n')
                #execute.write('export PATH="/data/at3/scratch3/jharrison/miniconda3/envs/ML_env/bin/:$PATH" \n')
                execute.write('export PATH="/data/at3/common/multilepton/miniforge3/envs/ML_env/bin/:$PATH" \n')
                #execute.write('#!/data/at3/scratch3/jharrison/miniconda3/envs/ML_env/bin/python')
                execute.write('cd /nfs/pic.es/user/j/jharriso/IFAE_ML\n')               
                execute.write('eval "$(conda shell.bash hook)"\n')
                execute.write('mamba activate ML_env\n')

                func = f"python evaluate/evaluate_model_v4.py -r {region} -e {even_path} -o {odd_path} -f {feather_conf} -ni {ntuplePathIn} -no {ntuplePathOut} --Inds "

                if eval_DD:
                    func = f"python evaluate/evaluate_model_DDqmisID.py -r {region} -e {even_path} -o {odd_path} -f {feather_conf} -ni {ntuplePathIn} -no {ntuplePathOut} --Inds "
                if SYSTS:
                    func = f"python evaluate/evaluate_model_Systs.py -r {region} -e {even_path} -o {odd_path} -f {feather_conf} -ni {ntuplePathIn} -no {ntuplePathOut} --Inds "

                for ind in inds:
                    func += f"{ind} "
                    
                execute.write(func)
                execute.write(' \n')
                execute.write('conda deactivate \n')
                execute.close()
                print(f"chmodding: {sh_name}")
                os.system(f"chmod +x {sh_name}")
                
        else:
            
            #Make the executable file
            sh_name = os.path.join(scriptdir,f"{region}.sh")
            execute = open(sh_name, "w")
            execute.write('#!/bin/bash \n')
            #execute.write('export PATH="/data/at3/scratch3/jharrison/miniconda3/envs/ML_env/bin/:$PATH" \n')
            execute.write('export PATH="/data/at3/common/multilepton/miniforge3/envs/ML_env/bin/:$PATH" \n')
            #execute.write('#!/data/at3/scratch3/jharrison/miniconda3/envs/ML_env/bin/python')
            execute.write('cd /nfs/pic.es/user/j/jharriso/IFAE_ML\n')               
            execute.write('eval "$(conda shell.bash hook)"\n')
            execute.write('mamba activate ML_env\n')

            func = f"python evaluate/evaluate_model_v4.py -r {region} -e {even_path} -o {odd_path} -f {feather_conf} -ni {ntuplePathIn} -no {ntuplePathOut}"
            
            #os.system(func)
            
            if eval_DD:
                func = f"python evaluate/evaluate_model_DDqmisID.py -r {region} -e {even_path} -o {odd_path} -f {feather_conf} -ni {ntuplePathIn} -no {ntuplePathOut}"
                
            if SYSTS:
                    func = f"python evaluate/evaluate_model_Systs.py -r {region} -e {even_path} -o {odd_path} -f {feather_conf} -ni {ntuplePathIn} -no {ntuplePathOut}"
            
            execute.write(func)

            execute.write(' \n')
            execute.write('conda deactivate \n')
            execute.close()
            print(f"chmodding: {sh_name}")
            os.system(f"chmod +x {sh_name}")
        
            
            
        #Make the submit file and submit it
        condor = open(os.path.join(scriptdir,"condor_submit.sub"), "w")
        name = f"name  = {region}\n"
        junk = f"executable  = $(name)\n"
        batch_name = f"JobBatchName = {job_suffix}{region}\n"
        #condor.write(name)
        condor.write(junk)
        condor.write(batch_name)
        
        #junk1 = f"output  =  evaluate/jobs/{job_name}/{region}/{region}.out\n"
        #junk2 = f"log  =  evaluate/jobs/{job_name}/{region}/{region}.log\n"
        #junk3 = f"error  =  evaluate/jobs/{job_name}/{region}/{region}.err\n"
        
        junk1 = f"output  =  evaluate/jobs/{job_name}/{region}/outs/$Fnx(name).out\n"
        junk2 = f"log  =  evaluate/jobs/{job_name}/{region}/logs/$Fnx(name).log\n"
        junk3 = f"error  =  evaluate/jobs/{job_name}/{region}/errs/$Fnx(name).err\n"


        condor.write(junk1)
        condor.write(junk2)
        condor.write(junk3)
        condor.write("getenv = True\n")
        condor.write('+flavour="long"\n')
        condor.write('request_cpus = 1\n')

        condor.write('request_memory = 8 GB\n')
        condor.write('on_exit_remove          = (ExitBySignal == False) && (ExitCode == 0)\n')
        condor.write('requirements = !regexp("AMD EPYC 7452",CPU_MODEL)\n')
        condor.write('max_retries = 1\n')
        
        condor.write(f"queue name matching files (evaluate/jobs/{job_name}/{region}/*.sh)\n")
        condor.close()
    
        if args.submit:
            os.system(f"condor_submit {os.path.join(scriptdir, 'condor_submit.sub')}")

    

