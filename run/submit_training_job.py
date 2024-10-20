import os
import argparse
import shutil
import datetime as dt

def add_run_to_history(job_name, history_file):
    if not os.path.exists(history_file):
        f = open(history_file, "x")
    with open(history_file, 'r') as file:
        history = file.readlines()
    history.append(job_name + ' : {} \n'.format(dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    with open(history_file, 'w') as file_out:
        file_out.writelines(history)


if __name__ == '__main__':
    

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--submit",default=False, help="Choose whether to submit the jobs or not", type=bool)
    args = parser.parse_args()
    
    
    os.chdir("/nfs/pic.es/user/j/jharriso/IFAE_ML")

    #job_name = "BaseRun_Even_3kEpochs"
    job_name = "ConfigCheck_0Z2SFOS_Even"
    NORM_FLOW = True
    Q2 = False
    EVEN_ODD = 'Even'
    num_epochs = 3000
    
    regions = []
    regions += ['0Z_0SFOS',
                '0Z_1SFOS',
                '0Z_2SFOS',
                '1Z_1SFOS',
                '1Z_2SFOS',
                '2Z']
    regions = ['0Z_0b_2SFOS']#,'0Z_1SFOS']
    #regions += ['0Z_0b_0SFOS']
    #regions+=['0Z_1b_2SFOS']
    #regions += ['0Z_0b_0SFOS', '0Z_0b_1SFOS', '0Z_0b_2SFOS',
    #            '1Z_0b_1SFOS', '1Z_0b_2SFOS','2Z_0b',
    #            '0Z_1b_0SFOS', '0Z_1b_1SFOS', '0Z_1b_2SFOS',
    #            '1Z_1b_1SFOS', '1Z_1b_2SFOS', '2Z_1b']
    
    if Q2:
        #regions = ["Q2_0b_e", "Q2_0b_eu", "Q2_0b_u","Q2_1b_e", "Q2_1b_eu", "Q2_1b_u"]
        #regions = ["Q2_0b", "Q2_1b"]
        regions = ["Q2_1b","Q2_1b_e", "Q2_1b_eu", "Q2_1b_u"]
    
    #regions+=['1Z_0b_2SFOS']
    #regions += ['2Z_0b']
    #regions += ['1Z_1b_2SFOS', '1Z_0b_2SFOS']
    
    #regions = ['VLLs/'+r for r in regions]
    
    flavour = "testmatch"

    for region in regions:
        scriptdir = f"run/jobs/{job_name}/{region}"
        
        if not os.path.exists(scriptdir):
            os.makedirs(scriptdir)

        #Copy the config file to the scriptdir
        if NORM_FLOW:
            #conf_file = f"configs/training_configs/Regions/{region}/nf_config.yaml"

            conf_file = f"configs/training_configs/Regions/{region}/nf_config_Inclusive.yaml"
        else:
            conf_file = f"configs/training_configs/Regions/{region}/training_config.yaml"
        new_conf_file = os.path.join(scriptdir, 'training_config.yaml')
        shutil.copyfile(conf_file, new_conf_file)
            
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
              
        
        func = f"python run/run_all.py -i {new_conf_file} -j {job_name} -e {EVEN_ODD}"
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
        batch_name = f"JobBatchName = {region}_{EVEN_ODD}\n"
        condor.write(name)
        condor.write(junk)
        condor.write(batch_name)
        

        junk1 = f"output  =  run/jobs/{job_name}/{region}/{region}.out\n"
        junk2 = f"log  =  run/jobs/{job_name}/{region}/{region}.log\n"
        junk3 = f"error  =  run/jobs/{job_name}/{region}/{region}.err\n"

        condor.write(junk1)
        condor.write(junk2)
        condor.write(junk3)
        condor.write("getenv = True\n")
        condor.write(f"+flavour = 'long'\n")
        condor.write('request_cpus = 2\n')
        condor.write('request_memory = 8 GB\n')
        condor.write('on_exit_remove          = (ExitBySignal == False) && (ExitCode == 0)\n')
        condor.write('requirements = !regexp("AMD EPYC 7452",CPU_MODEL)\n')
        condor.write('max_retries = 1\n')
        condor.write(f"queue name matching files (run/jobs/{job_name}/{region}/*.sh)\n")
        condor.close()
    
        if args.submit:
            os.system(f"condor_submit {os.path.join(scriptdir, 'condor_submit.sub')}")

            add_run_to_history(job_name, 'run/jobs/history.txt')


    

