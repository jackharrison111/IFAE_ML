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
    args = parser.parse_args()
    split = args.parallel
    
    
    os.chdir("/nfs/pic.es/user/j/jharriso/IFAE_ML")

    SYSTS = True
    Q2 = True
    eval_DD = False
    
    job_name = "OldReeval_Sys1_v2"
    job_prefix = "Sys1"
    
    split_amount = 50
    total_files = -1
    

    #Change here to set folder to read from
    ntuplePathIn = "/data/at3/common/multilepton/FinalSystProduction/Sys1"
    ntuplePathOut  = "/data/at3/common/multilepton/FinalSystProduction/evaluations/Sys1"
    
    #ntuplePathIn = "/data/at3/common/separi/nominal"
    #ntuplePathOut  = "/data/at3/common/multilepton/VLLemu/evaluations/nominal"

    if 'FinalSystProduction/Sys' in ntuplePathIn:
        SYSTS = True
    
    if Q2:
        train_run_file = 'nf_Q2.yaml'
    else:
        train_run_file = 'nf_NewYields.yaml'
    
    
        
    

    
    #chosen_regions = ['0Z_0b_2SFOS', '0Z_1b_2SFOS', '0Z_0b_1SFOS']
    
    #chosen_regions = ['0Z_0b_0SFOS', '0Z_0b_1SFOS', '0Z_0b_2SFOS',
    #                '1Z_0b_1SFOS', '1Z_0b_2SFOS', '2Z_0b',
    #                '0Z_1b_0SFOS', '0Z_1b_1SFOS', '0Z_1b_2SFOS',
    #                '1Z_1b_1SFOS', '1Z_1b_2SFOS', '2Z_1b']
    
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

    if total_files==-1:
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
            for i in range(regions['split_amount'], regions['total_files']+regions['split_amount'], regions['split_amount']):
                
                #Make the executable file
                sh_name = os.path.join(scriptdir,f"{region}_{s}_{i}.sh")
                execute = open(sh_name, "w")
                execute.write('#!/bin/bash \n')
                #execute.write('export PATH="/data/at3/scratch3/jharrison/miniforge3/envs/ML_env/bin/:$PATH" \n')
                execute.write('export PATH="/data/at3/common/multilepton/miniforge3/envs/ML_env/bin/:$PATH" \n')
                
                execute.write('cd /nfs/pic.es/user/j/jharriso/IFAE_ML\n')               
                execute.write('eval "$(conda shell.bash hook)"\n')
                execute.write('mamba activate ML_env\n')
                
                

                #conf_file = f"configs/training_configs/Regions/{region}/training_config.yaml"
                #func = f"python evaluate/evaluate_model_v3.py -r {region} -e {even_path} -o {odd_path} --First {s} --Last {i} -f {feather_conf} -ni {ntuplePathIn} -no {ntuplePathOut}"
                func = f"python evaluate/evaluate_model_v4.py -r {region} -e {even_path} -o {odd_path} --First {s} --Last {i} -f {feather_conf} -ni {ntuplePathIn} -no {ntuplePathOut}"
                
                
               
                
                if eval_DD:
                    func = f"python evaluate/evaluate_model_DDqmisID.py -r {region} -e {even_path} -o {odd_path} -f {feather_conf} -ni {ntuplePathIn} -no {ntuplePathOut} --First {s} --Last {i}"
                    
                if SYSTS:
                    func = f"python evaluate/evaluate_model_Systs.py -r {region} -e {even_path} -o {odd_path} -f {feather_conf} -ni {ntuplePathIn} -no {ntuplePathOut} --First {s} --Last {i}"
                
                #Need to set the feather file, the config file to use, 
                
                execute.write(func)

                execute.write(' \n')
                execute.write('conda deactivate \n')
                execute.close()
                print(f"chmodding: {sh_name}")
                os.system(f"chmod +x {sh_name}")
                s = i
                
                
                
              
           
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

            #conf_file = f"configs/training_configs/Regions/{region}/training_config.yaml"
            #func = f"python evaluate/evaluate_model.py -r {region} -e {even_path} -o {odd_path}"
            #func = f"python evaluate/evaluate_model_v3.py -r {region} -e {even_path} -o {odd_path} -f {feather_conf} -ni {ntuplePathIn} -no {ntuplePathOut}"

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
        batch_name = f"JobBatchName = {job_prefix}{region}\n"
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

    

