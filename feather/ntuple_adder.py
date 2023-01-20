
import uproot
import pandas as pd
import numpy as np

print(uproot.__version__)

root_path = '/data/at3/scratch3/jharrison/nominal/mc16a/'
root_file = '364250.root'
mod_path = '/data/at3/scratch3/jharrison/test_mod_ntuples'


test_array = [0,1,2,3,4]
evtNums = [393587,390718,389753,389242,390970]
data = {'eventNumber' : evtNums,
	'score' : test_array
	}
data = pd.DataFrame.from_dict(data)
print(data.head())


other_data = [6,7,8,9]
other_evts = [15059124, 15059197, 15058788, 15257292]
other_dict = {'eventNumber' : other_evts,
             'score' : other_data
             }
n_data = pd.DataFrame.from_dict(other_dict)

update = True
if not update:
    with uproot.open(root_path+root_file+':nominal') as evts:
        #print(evts.keys())
        evtNum = evts.arrays(library='pd')
        print(evtNum['eventNumber'].tail())
        new_df = evtNum.merge(data, on='eventNumber', how='left')
        new_df['score'] = new_df['score'].replace(np.nan, -99)
        print(new_df.head())
        print(len(new_df))



if update:
    with uproot.open('364250_mod.root:nominal') as evts:
        evtNumScore = evts.arrays(library='pd')
        print(evtNumScore.head())
        print(evtNumScore.tail())
        
        df = evtNumScore.merge(n_data, on='eventNumber', how='left')
        df['score'] = df['score_y'].combine_first(df['score_x'])
        df.drop(columns=['score_x','score_y'],inplace=True)
        #evts['nominal']['score']=df['score']
        print("Head: \n", df.head())
        print("Tail: \n", df.tail())
        
    with uproot.recreate('364250_mod.root') as evts:
        evts['nominal'] = df
    #with uproot.update('364250_mod.root') as evts:
    #    evts['nominal/score'] = df['score']
        
        
    ...
    
else:
    with uproot.recreate('364250_mod.root') as f:
        f['nominal'] = new_df
 
#f1 = uproot.open(root_path+root_file+':nominal',library="pd")
#print(f1.head())




