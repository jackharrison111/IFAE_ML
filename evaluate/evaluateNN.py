import sys
import ROOT,os,uproot
import argparse
import sys, gc
import os
import pickle
import math
import numpy as np
seed = 400
np.random.seed(seed)
import pandas as pd
import tensorflow as tf
tf.random.set_seed(400)
#from root_numpy import array2tree
import awkward as ak
from sklearn.preprocessing import RobustScaler, StandardScaler
import joblib
import tensorflow.keras.models as mod
tf.config.run_functions_eagerly(False)
ROOT.gInterpreter.GenerateDictionary('std::vector<std::vector<int>>')
ROOT.gROOT.ProcessLine("gErrorIgnoreLevel = 2001;")
from concurrent.futures import ThreadPoolExecutor
import time
#runNumber

print("Loading aux functions")
start_time= time.time() #start timer

def load_model(train_path,cft) :
    arch_path = train_path + "/architecture_{}_2hdm.json".format(cft)
    weights_path = train_path + "/weights_{}_2hdm.h5".format(cft)
    print(arch_path)
    print(weights_path)
    json_file = open(os.path.abspath(arch_path), 'r')
    loaded_model = json_file.read()
    json_file.close()
    loaded_model = mod.model_from_json(loaded_model)
    loaded_model.load_weights(os.path.abspath(weights_path))
    return loaded_model

def scale_data(ds_noncs,train_path) :
    scaler_path = train_path + "/scaler_standard.bin"
    scaler1 = joblib.load(scaler_path)
    print("scaler path = ",scaler_path)
    ds_noncs=ds_noncs.replace(99,0)
    ds_noncs=ds_noncs.replace(-99,0)
    ds_noncs=ds_noncs.replace(9999,0)
    ds_noncs=ds_noncs.replace(-9999,0)
    df_sc = scaler1.transform(ds_noncs)
    return df_sc

#@njit(parallel=True)
def predict_NN(model, arr_frame):
    scores = np.empty([len(arr_frame), 5], dtype=np.float16)
    BATCH_INDICES = np.arange(start=0, stop=len(arr_frame), step=100000)  # row indices of batches
    BATCH_INDICES = np.append(BATCH_INDICES, len(arr_frame))  # add final batch_end row

    for index in np.arange(len(BATCH_INDICES) - 1):
        batch_start = BATCH_INDICES[index]  # first row of the batch
        batch_end = BATCH_INDICES[index + 1]  # last row of the batch
        scores[batch_start:batch_end] = model.predict(arr_frame[batch_start:batch_end])
        tf.keras.backend.clear_session()
        _ = gc.collect()
    return scores

#@njit(parallel=True)
def predict_NN_SSB(model, arr_frame):
    scores = np.empty([len(arr_frame), 6], dtype=np.float16)
    BATCH_INDICES = np.arange(start=0, stop=len(arr_frame), step=100000)  # row indices of batches
    BATCH_INDICES = np.append(BATCH_INDICES, len(arr_frame))  # add final batch_end row

    for index in np.arange(len(BATCH_INDICES) - 1):
        batch_start = BATCH_INDICES[index]  # first row of the batch
        batch_end = BATCH_INDICES[index + 1]  # last row of the batch
        scores[batch_start:batch_end] = model.predict(arr_frame[batch_start:batch_end])
        tf.keras.backend.clear_session()
        _ = gc.collect()
    return scores

#@njit(parallel=True)
def predict_NN3l(model, arr_frame):
    scores = np.empty([len(arr_frame), 3], dtype=np.float16)
    BATCH_INDICES = np.arange(start=0, stop=len(arr_frame), step=100000)  # row indices of batches
    BATCH_INDICES = np.append(BATCH_INDICES, len(arr_frame))  # add final batch_end row

    for index in np.arange(len(BATCH_INDICES) - 1):
        batch_start = BATCH_INDICES[index]  # first row of the batch
        batch_end = BATCH_INDICES[index + 1]  # last row of the batch
        scores[batch_start:batch_end] = model.predict(arr_frame[batch_start:batch_end])
        tf.keras.backend.clear_session()
        _ = gc.collect()
    return scores

#@njit(parallel=True)
def predict_NN3l_SSB(model, arr_frame):
    scores = np.empty([len(arr_frame), 4], dtype=np.float16)
    BATCH_INDICES = np.arange(start=0, stop=len(arr_frame), step=100000)  # row indices of batches
    BATCH_INDICES = np.append(BATCH_INDICES, len(arr_frame))  # add final batch_end row

    for index in np.arange(len(BATCH_INDICES) - 1):
        batch_start = BATCH_INDICES[index]  # first row of the batch
        batch_end = BATCH_INDICES[index + 1]  # last row of the batch
        scores[batch_start:batch_end] = model.predict(arr_frame[batch_start:batch_end])
        tf.keras.backend.clear_session()
        _ = gc.collect()
    return scores


#executor = ThreadPoolExecutor(8)
end_time = time.time()
print("++++++++++++++ Loaded aux functions", end_time-start_time)
print("Loaded aux functions")



#@profile
def main() :
    print("Loading aux functions")
    start_time= time.time() #start timer

    parser = argparse.ArgumentParser(description = "Evaluate NN")
    parser.add_argument("-i", "--input",
        help = "Provide input root files",
        required = True)
    parser.add_argument("-t2l", "--trainpath2l", help = "path of the trainings", required = True)
    parser.add_argument("-t3l", "--trainpath3l", help = "path of the trainings", required = True)
    parser.add_argument("-t2lSSB", "--trainpath2lSSB", help = "path of the trainings", required = True)
    parser.add_argument("-t3lSSB", "--trainpath3lSSB", help = "path of the trainings", required = True)
    #parser.add_argument("-t4l", "--trainpath4l", help = "path of the trainings", required = True)
    #parser.add_argument("-tSvB", "--trainpathSvB", help = "path of the trainings", required = True)
    parser.add_argument("--outdir", help = "Provide an output directory for plots", default = "./")
    args = parser.parse_args()
    #print("Starting load")
    

    end_time2 = time.time()
    print("++++++++++++++ Started loading models", end_time2-start_time)
	
    #SvsS NNs
    model2l_cift = load_model(args.trainpath2l,"cift")
    print("2c SS done")
    model2l_tek = load_model(args.trainpath2l,"tek")
    print("2t SS done")
    model3l_cift = load_model(args.trainpath3l,"cift")
    print("3c SS done")
    model3l_tek = load_model(args.trainpath3l,"tek")
    print("3l SS done")
    
    #SvsSvsB NNs
    model2l_SSB_cift = load_model(args.trainpath2lSSB,"cift")
    print("2c SSB done")
    model2l_SSB_tek = load_model(args.trainpath2lSSB,"tek")
    print("2t SSB done")
    model3l_SSB_cift = load_model(args.trainpath3lSSB,"cift")
    print("3c SSB done")
    model3l_SSB_tek = load_model(args.trainpath3lSSB,"tek")
    print("3l SSB done")
    
    end_time3 = time.time()
    print("++++++++++++++ Finished loading models", end_time3-start_time)
    #out_data = None
    print("Starting loop")
    
    f1 = uproot.open(args.input,library="pd")
    f=f1['nominal']
    

    end_time4 = time.time()
    print("++++++++++++++ Opened file", end_time4-start_time)	

    #all_data = uproot.open(args.input+':nominal',library="pd")
    
   
    #import pdb; pdb.set_trace()
        
    all_data_all = f.arrays(["nHiggs", "nWZhad","sumPsbtag","met_met","nJets_OR"],library="pd")

    all_data = f.arrays(["lqlq_decays"],library="pd")
	
    end_time5 = time.time()
    print("++++++++++++++ Filled array 1", end_time5-start_time) 

    
    events = f.arrays(["randomRunNumber"],library="np")
    #events = all_data[["randomRunNumber"]].to_numpy()
    #events = events.flatten()
        
    end_time6 = time.time()
    print("++++++++++++++ Filled array 2", end_time6-start_time)


    #scale
    df_sc2l = scale_data(all_data_all,args.trainpath2l)
    df_sc3l = scale_data(all_data_all,args.trainpath3l)
        
    df_SSB_sc2l = scale_data(all_data_all,args.trainpath2lSSB)
    df_SSB_sc3l = scale_data(all_data_all,args.trainpath3lSSB)
        
    end_time6 = time.time()
    print("++++ Scaled", end_time6-start_time)
        
    #score predictions
    scores2l = np.empty([events["randomRunNumber"].size,5],'<f4')
    scores3l = np.empty([events["randomRunNumber"].size,3],'<f4')
    scores2l_SSB = np.empty([events["randomRunNumber"].size,6],'<f4')
    scores3l_SSB = np.empty([events["randomRunNumber"].size,4],'<f4')
        
        
    #score predictions SS
    scores2l_tek = predict_NN(model2l_cift,df_sc2l[events["randomRunNumber"]%2==1])
    scores2l_cift = predict_NN(model2l_tek,df_sc2l[events["randomRunNumber"]%2==0])
    scores3l_tek = predict_NN3l(model3l_cift,df_sc3l[events["randomRunNumber"]%2==1])
    scores3l_cift = predict_NN3l(model3l_tek,df_sc3l[events["randomRunNumber"]%2==0])
        
    end_time7 = time.time()
    print("++++ Predicted", end_time7-start_time)
	
    scores2l[events["randomRunNumber"]%2==1] = scores2l_tek
    scores2l[events["randomRunNumber"]%2==0] = scores2l_cift
    scores3l[events["randomRunNumber"]%2==1] = scores3l_tek
    scores3l[events["randomRunNumber"]%2==0] = scores3l_cift
        
    max2l = np.argmax(scores2l, axis=1)
    max3l = np.argmax(scores3l, axis=1)
 
    #score predictions SSB
    scores2l_SSB_tek = predict_NN_SSB(model2l_SSB_cift,df_SSB_sc2l[events["randomRunNumber"]%2==1])
    scores2l_SSB_cift = predict_NN_SSB(model2l_SSB_tek,df_SSB_sc2l[events["randomRunNumber"]%2==0])
    scores3l_SSB_tek = predict_NN3l_SSB(model3l_SSB_cift,df_SSB_sc3l[events["randomRunNumber"]%2==1])
    scores3l_SSB_cift = predict_NN3l_SSB(model3l_SSB_tek,df_SSB_sc3l[events["randomRunNumber"]%2==0])
        
    scores2l_SSB[events["randomRunNumber"]%2==1] = scores2l_SSB_tek
    scores2l_SSB[events["randomRunNumber"]%2==0] = scores2l_SSB_cift
    scores3l_SSB[events["randomRunNumber"]%2==1] = scores3l_SSB_tek
    scores3l_SSB[events["randomRunNumber"]%2==0] = scores3l_SSB_cift
        
    max2l_SSB = np.argmax(scores2l_SSB, axis=1)
    max3l_SSB = np.argmax(scores3l_SSB, axis=1)
 
    end_time8 = time.time()
    print("++++ Predicted SSB", end_time8-start_time)


    #all_data=f.arrays(library="pd")
    
    end_time9 = time.time()
    print("++++ Opened file", end_time9-start_time)

    #import pdb; pdb.set_trace()

    all_data=pd.concat([all_data, pd.DataFrame({'max2l' :max2l})], axis=1, copy=False)
    all_data=pd.concat([all_data, pd.DataFrame({'max3l' :max3l})], axis=1, copy=False)
    all_data=pd.concat([all_data, pd.DataFrame({'max2l_SSB' :max2l_SSB})], axis=1, copy=False)
    all_data=pd.concat([all_data, pd.DataFrame({'max3l_SSB' :max3l_SSB})], axis=1, copy=False)
        
    #all_data=pd.concat([all_data, pd.DataFrame({'max2l_SSB_score_0' :scores2l_SSB[0]})], axis=1, copy=False)
    #all_data=pd.concat([all_data, pd.DataFrame({'max2l_SSB_score_1' :scores2l_SSB[1]})], axis=1, copy=False)
    #all_data=pd.concat([all_data, pd.DataFrame({'max2l_SSB_score_2' :scores2l_SSB[2]})], axis=1, copy=False)
    #all_data=pd.concat([all_data, pd.DataFrame({'max2l_SSB_score_3' :scores2l_SSB[3]})], axis=1, copy=False)
    #all_data=pd.concat([all_data, pd.DataFrame({'max2l_SSB_score_4' :scores2l_SSB[4]})], axis=1, copy=False)
    #all_data=pd.concat([all_data, pd.DataFrame({'max2l_SSB_score_5' :scores2l_SSB[5]})], axis=1, copy=False)
        
    #all_data=pd.concat([all_data, pd.DataFrame({'max3l_SSB_score_0' :scores3l_SSB[0]})], axis=1, copy=False)
    #all_data=pd.concat([all_data, pd.DataFrame({'max3l_SSB_score_1' :scores3l_SSB[1]})], axis=1, copy=False)
    #all_data=pd.concat([all_data, pd.DataFrame({'max3l_SSB_score_2' :scores3l_SSB[2]})], axis=1, copy=False)
    #all_data=pd.concat([all_data, pd.DataFrame({'max3l_SSB_score_3' :scores3l_SSB[3]})], axis=1, copy=False)

    #import pdb; pdb.set_trace()

    end_time9 = time.time()
    print("++++ CATTED", end_time9-start_time)
    
    #end_time = time.time()
    #rint("++++++++++++++")
    #print("++++++++++++++Time", start_time-end_time)
    #print("++++++++++++++")
   
    length = len(args.input.split("/"))
    print(len(args.input.split("/")))
    files = args.input.split("/")[length-1]
    file_name = args.outdir + args.input.split("/")[length-1] #11,10
    files=files.replace('./','')
    files=files.replace('.root','')
       
    end_time9 = time.time()
    print("++++ Gonna create", end_time9-start_time)
    with uproot.recreate(args.outdir+"/outGN2_%s.root"%(files)) as rootfile:
        rootfile["new_nominal"]=all_data
        gc.collect()

    end_time9 = time.time()
    print("++++ Recreated", end_time9-start_time)    
    print("rooo3")
    #start_time = time.time()
    #from ROOT import RDataFrame
    #print("File 1 is", f1)
    #print("File 2 is", f2)    
    
    #f1 = ROOT.TFile.Open(args.input,"READ")
    #f2 = ROOT.TFile.Open(args.outdir+"/outGN2_%s.root"%(files),"READ")
    #print("File 1 is", f1)
    #print("File 2 is", f2)	

    #t1=f1.Get("nominal")
    #t2=f2.Get("nominal")
    #t1.AddFriend(t2)
    #print("Made friends")
    #df = RDataFrame(t1)
    #end_time=time.time()
    #print("Took x mins to make friend+df", end_time-start_time)
    #print("Made df; gonna snapshot now")
    #print("Outfile is ", args.outdir+"/%s.root"%(files))
    #import pdb; pdb.set_trace()
    #df.Snapshot("nominal",args.outdir+"/%s.root"%(files))
    #end_time=time.time()
    #print("Took x mins to make a snapshot", end_time-start_time)
    #print("Done")	
    #os.system("rm "args.outdir+"/outGN2_%s*"%(files))
    
    #import pdb; pdb.set_trace()
    #os.system("hadd -ff %s"%(file_name)+" "+args.outdir+"/outGN2_%s*"%(files))
    #os.system("hadd -ff %s"%(file_name)+" "+args.outdir+"/outGN2_%s*"%(files))
    #from glob import glob
    #[os.unlink(f) for f in glob(args.outdir+'/outGN2*'+files+'*.root')]
    #end_time9 = time.time()
    #print("++++ Hadded", end_time9-start_time) 
    #print("done, stuff saved with: {}".format(file_name))


if __name__ == "__main__" :
    main()
