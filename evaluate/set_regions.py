import os
import argparse
import yaml

#Write a function to get all the folders automatically - don't want to have to change it ! 
#Pass a job name, and then for each region, find the Run folder

def define_regions(region_file):

    
    with open(region_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config
'''
    regions = {}
    even_base_dir = 'results/AllSigs'
    odd_base_dir = 'results/AllSigs_Odd'
    
    
    regions['0Z_0b_0SFOS'] = {
        'even_load_dir' : os.path.join(even_base_dir,'0Z_0b_0SFOS_AllSigs/Run_1333-21-04-2023'),
        'odd_load_dir'  : os.path.join(odd_base_dir,'0Z_0b_0SFOS_AllSigs/Run_1250-23-04-2023'),
        'split_amount' : 100,
        'total_files' : 2000,
    }
    regions['0Z_0b_1SFOS'] = {
        'even_load_dir' : os.path.join(even_base_dir,'0Z_0b_1SFOS_AllSigs/Run_1334-21-04-2023'),
        'odd_load_dir'  : os.path.join(odd_base_dir,'0Z_0b_1SFOS_AllSigs/Run_1250-23-04-2023'),
        'split_amount' : 100,
        'total_files' : 2000,
    }
    regions['0Z_0b_2SFOS'] = {
        'even_load_dir' : os.path.join(even_base_dir,'0Z_0b_2SFOS_AllSigs/Run_1335-21-04-2023'),
        'odd_load_dir'  : os.path.join(odd_base_dir,'0Z_0b_2SFOS_AllSigs/Run_1251-23-04-2023'),
        'split_amount' : 100,
        'total_files' : 2000,
    }
    regions['0Z_1b_0SFOS'] = {
        'even_load_dir' : os.path.join(even_base_dir,'0Z_1b_0SFOS_AllSigs/Run_1405-21-04-2023'),
        'odd_load_dir'  : os.path.join(odd_base_dir,'0Z_1b_0SFOS_AllSigs/Run_1312-23-04-2023'),
        'split_amount' : 100,
        'total_files' : 2000,
    }  
    regions['0Z_1b_1SFOS'] = {
        'even_load_dir' : os.path.join(even_base_dir,'0Z_1b_1SFOS_AllSigs/Run_1407-21-04-2023'),
        'odd_load_dir'  : os.path.join(odd_base_dir,'0Z_1b_1SFOS_AllSigs/Run_1313-23-04-2023'),
        'split_amount' : 100,
        'total_files' : 2000,
    }
    regions['0Z_1b_2SFOS'] = {
        'even_load_dir' : os.path.join(even_base_dir,'0Z_1b_2SFOS_AllSigs/Run_1422-21-04-2023'),
        'odd_load_dir'  : os.path.join(odd_base_dir,'0Z_1b_2SFOS_AllSigs/Run_1319-23-04-2023'),
        'split_amount' : 100,
        'total_files' : 2000,
    } 
    regions['1Z_0b_1SFOS'] = {
        'even_load_dir' : os.path.join(even_base_dir,'1Z_0b_1SFOS_AllSigs/Run_1351-21-04-2023'),
        'odd_load_dir'  : os.path.join(odd_base_dir,'1Z_0b_1SFOS_AllSigs/Run_1258-23-04-2023'),
        'split_amount' : 100,
        'total_files' : 2000,
    }
    regions['1Z_0b_2SFOS'] = {
        'even_load_dir' : os.path.join(even_base_dir,'1Z_0b_2SFOS_AllSigs/Run_1159-21-04-2023'),
        'odd_load_dir'  : os.path.join(odd_base_dir,'1Z_0b_2SFOS_AllSigs/Run_1310-23-04-2023'),
        'split_amount' : 100,
        'total_files' : 2000,
    }
    regions['1Z_1b_1SFOS'] = {
        'even_load_dir' : os.path.join(even_base_dir,'1Z_1b_1SFOS_AllSigs/Run_1443-21-04-2023'),
        'odd_load_dir'  : os.path.join(odd_base_dir,'1Z_1b_1SFOS_AllSigs/Run_1350-23-04-2023'),
        'split_amount' : 100,
        'total_files' : 2000,
    }
    regions['1Z_1b_2SFOS'] = {
        'even_load_dir' : os.path.join(even_base_dir,'1Z_1b_2SFOS_AllSigs/Run_1456-21-04-2023'),
        'odd_load_dir'  : os.path.join(odd_base_dir,'1Z_1b_2SFOS_AllSigs/Run_1350-23-04-2023'),
        'split_amount' : 100,
        'total_files' : 2000,
    }
    regions['2Z_0b'] = {
        'even_load_dir' : os.path.join(even_base_dir,'2Z_0b_AllSigs/Run_1351-21-04-2023'),
        'odd_load_dir'  : os.path.join(odd_base_dir,'2Z_0b_AllSigs/Run_1310-23-04-2023'),
        'split_amount' : 100,
        'total_files' : 2000,
    }
    regions['2Z_1b'] = {
        'even_load_dir' : os.path.join(even_base_dir,'2Z_1b_AllSigs/Run_1515-21-04-2023'),
        'odd_load_dir'  : os.path.join(odd_base_dir,'2Z_1b_AllSigs/Run_1349-23-04-2023'),
        'split_amount' : 100,
        'total_files' : 2000,
    }
    
    return regions

'''
if __name__ == '__main__':
    
    define_regions()
    