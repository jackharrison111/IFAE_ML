import os
import argparse
import shutil

if __name__ == '__main__':
    

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--submit",default=False, help="Choose whether to submit the jobs or not", type=bool)
    args = parser.parse_args()
    
    
    os.chdir("/nfs/pic.es/user/j/jharriso/IFAE_ML")

    job_name = "MTLepMET_AllSigs"
    SIGNALS = True
    
    regions = []
    '''
    regions += ['1Z_0b_2SFOS']
    '''
    regions += ['0Z_0b_0SFOS', '0Z_0b_1SFOS', '0Z_0b_2SFOS',
                '1Z_0b_1SFOS', '1Z_0b_2SFOS', '2Z_0b',
                '0Z_1b_0SFOS', '0Z_1b_1SFOS', '0Z_1b_2SFOS',
                '1Z_1b_1SFOS', '1Z_1b_2SFOS', '2Z_1b']
    
    #regions += ['1Z_0b_2SFOS']
    
    flavour = "testmatch"

    for region in regions:
        scriptdir = f"feather/submits/{job_name}/{region}"
        if not os.path.exists(scriptdir):
            os.makedirs(scriptdir)
            
        #Copy the config file to the submit dir and use that
        conf_file = f"configs/feather_configs/10GeV/{region}.yaml"
        
        if SIGNALS:
            conf_file = f"configs/feather_configs/10GeV/VLLs/{region}.yaml"
        
        new_conf_file = os.path.join(scriptdir, 'feather_config.yaml')
        shutil.copyfile(conf_file, new_conf_file)
            
        #Make the executable file
        sh_name = os.path.join(scriptdir,f"{region}.sh")
        execute = open(sh_name, "w")
        execute.write('#!/bin/bash \n')
        execute.write('export PATH="/data/at3/scratch3/jharrison/miniconda3/envs/ML_env/bin/:$PATH" \n')
        #execute.write('#!/data/at3/scratch3/jharrison/miniconda3/envs/ML_env/bin/python')
        execute.write('cd /nfs/pic.es/user/j/jharriso/IFAE_ML\n')               
        execute.write('eval "$(conda shell.bash hook)"\n')
        execute.write('mamba activate ML_env\n')
              
            
        func = f'python feather/make_feather.py -c "%s"'%new_conf_file
        execute.write(func)
                       
        execute.write(' \n')
        execute.write('mamba deactivate \n')
        execute.close()
        print(f"chmodding: {sh_name}")
        os.system(f"chmod +x {sh_name}")
        
            
            
        #Make the submit file and submit it
        condor = open(os.path.join(scriptdir,"condor_submit.sub"), "w")
        name = f"name  = {region}\n"
        junk = f"executable  = $(name)\n"
        condor.write(name)
        condor.write(junk)
        

        junk1 = f"output  =  feather/submits/{job_name}/{region}/{region}.out\n"
        junk2 = f"log  =  feather/submits/{job_name}/{region}/{region}.log\n"
        junk3 = f"error  =  feather/submits/{job_name}/{region}/{region}.err\n"
        
        batch_name = f"JobBatchName = feather_{region}\n"

        condor.write(junk1)
        condor.write(junk2)
        condor.write(junk3)
        condor.write(batch_name)
        condor.write("getenv = True\n")
        condor.write('+flavour = "long"\n')
        condor.write('request_cpus = 1\n')
        condor.write('request_memory = 16 GB\n')
        condor.write('on_exit_remove          = (ExitBySignal == False) && (ExitCode == 0)\n')
        condor.write('requirements = !regexp("AMD EPYC 7452",CPU_MODEL)\n')
        condor.write('max_retries = 1\n')
        
        condor.write(f"queue name matching files (feather/submits/{job_name}/{region}/*.sh)\n")
        condor.close()
    
        if args.submit:
            os.system(f"condor_submit {os.path.join(scriptdir, 'condor_submit.sub')}")
    
    
        
