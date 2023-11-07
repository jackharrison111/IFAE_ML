'''
Script to add a new variable to the dataset

Takes as input:
- input ntuple dir
- output save dir

'''


import os
import argparse

    
if __name__ == '__main__':
    

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--submit",default=False, help="Choose whether to submit the jobs or not", type=bool)
    parser.add_argument("-p", "--parallel",default=True, help="Choose whether to submit the jobs in parallel", type=bool)
    args = parser.parse_args()
    split = args.parallel
    
    
    os.chdir("/nfs/pic.es/user/j/jharriso/IFAE_ML")


    split_amount = 100      #number of files per job
    total_files = 1900      #total number of files
    split = True

    job_name = "Add_nIFFVars"
    
    input_dir = '/data/at3/common/multilepton/VLL_production/nominal'
    save_dir = '/data/at3/common/multilepton/VLL_production/nominal_AddVar'



    scriptdir = f"utils/jobs/{job_name}"
    if not os.path.exists(scriptdir):
        os.makedirs(scriptdir)
    if not os.path.exists(os.path.join(scriptdir,'logs')):
        os.makedirs(os.path.join(scriptdir,'logs'))
    if not os.path.exists(os.path.join(scriptdir,'outs')):
        os.makedirs(os.path.join(scriptdir,'outs'))
    if not os.path.exists(os.path.join(scriptdir,'errs')):
        os.makedirs(os.path.join(scriptdir,'errs'))
    

    
    flavour = "long"

        
    if split:
        
        s = 0
        for i in range(split_amount, total_files, split_amount):
                
                #Make the executable file
                sh_name = os.path.join(scriptdir,f"Files_{s}_{i}.sh")

                execute = open(sh_name, "w")
                execute.write('#!/bin/bash \n')
                execute.write('export PATH="/data/at3/common/multilepton/miniforge3/envs/ML_env/bin/:$PATH" \n')
                
                execute.write('cd /nfs/pic.es/user/j/jharriso/IFAE_ML\n')               
                execute.write('eval "$(conda shell.bash hook)"\n')
                execute.write('mamba activate ML_env\n')
                
                func = f"python utils/add_var.py --First {s} --Last {i} -i {input_dir} -o {save_dir}"

                execute.write(func)
                execute.write(' \n')
                execute.write('conda deactivate \n')
                execute.close()
                print(f"chmodding: {sh_name}")
                os.system(f"chmod +x {sh_name}")
                s = i
                
            
        else:
            
            #Make the executable file
            sh_name = os.path.join(scriptdir,f"AllFiles.sh")
            execute = open(sh_name, "w")
            execute.write('#!/bin/bash \n')
            execute.write('export PATH="/data/at3/common/multilepton/miniforge3/envs/ML_env/bin/:$PATH" \n')
            execute.write('cd /nfs/pic.es/user/j/jharriso/IFAE_ML\n')               
            execute.write('eval "$(conda shell.bash hook)"\n')
            execute.write('mamba activate ML_env\n')
            
            func = f"python utils/add_var.py  -i {input_dir} -o {save_dir}"
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
        
        
        junk1 = f"output  =  utils/jobs/{job_name}/outs/$Fnx(name).out\n"
        junk2 = f"log  =  utils/jobs/{job_name}/logs/$Fnx(name).log\n"
        junk3 = f"error  =  utils/jobs/{job_name}/errs/$Fnx(name).err\n"


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
        
        condor.write(f"queue name matching files (utils/jobs/{job_name}/*.sh)\n")
        condor.close()
    
        if args.submit:
            os.system(f"condor_submit {os.path.join(scriptdir, 'condor_submit.sub')}")

    

