################################
#
# File to read trex-fitter samples
# and save them into json config 
# format
#
# Jack Harrison 15/07/22
# ###############################

import json 


'''
Function for extracting the list of DSIDs that each sample
should contain.
Format of file should be a text file with one sample per line
in the format:
    Sample_name: id1, id2, id3
'''

def extract_DSIDs(sample_file, output_json):
    
    with open(sample_file, 'r') as file:
        sample_txt = file.readlines()

    samples_dict = {}
    for line in sample_txt:
        all_samples = []
        sample_list = line.split(':')[-1]
        sample_name = line.split(':')[0]
        if sample_name[0]=='#':
            sample_name = sample_name[1:]
        samples = sample_list.split(',')
        DSIDs = [path.split('/')[-1][:6] for path in samples]
        for id in DSIDs:
            if id == '':
                continue
            all_samples.append(id)
        all_samples = list(set(all_samples))
        samples_dict[sample_name] = all_samples
    
    with open(output_json,'w') as f:
        json.dump(samples_dict, f)
        
        
def make_json_file(dictionary, path):
    with open(path, 'w') as f:
        json.dump(dictionary)

    
    
    
if __name__ == '__main__':
    sample_txt = 'configs/sample_files.txt'
    json_output = 'configs/samples.json'

    extract_DSIDs(sample_txt, json_output)
    
    
    
