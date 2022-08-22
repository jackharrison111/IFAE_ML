###########################################
# File to handle reading of TRex configs
# 
# Useful for choosing which samples + regions
# to use
#
# Jack Harrison 22/07/22
###########################################

import yaml

def format_trex_sample_file(file, outfile=None):
    if outfile==None:
        outfile=file
    with open(file,'r') as f:
        l = f.readlines()
    new_yaml = []
    for line in l:
        if 'Sample' in line:
            if '#Sample' in line:
                continue
            split = line.split(':')[-1]
            if split[0] == ' ':
                split = split[1:]
            split = split[:-1] + ':' + split[-1:]
            line = split
        new_yaml.append(line)
    with open(outfile, 'w') as f:
        f.writelines(new_yaml)
    return outfile


def read_trex_samples(file, outfile=None, format=True):
    if format:
        file = format_trex_sample_file(file, outfile)
    with open(file, 'r') as f:
        data = yaml.safe_load(f)
    return data