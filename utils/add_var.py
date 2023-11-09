# Add here
import ROOT
import os
from ROOT import RDataFrame
import argparse
from utils._utils import find_root_files

getNumOfIFFClassFlavour = '''
int getNumOfIFFClassFlavour(int classChoice, int flavourChoice,
    int IFFClass_lep_0, int IFFClass_lep_1, int IFFClass_lep_2, int IFFClass_lep_3, int IFFClass_lep_4, int IFFClass_lep_5,
    int lep_ID_0,  int lep_ID_1,  int lep_ID_2,  int lep_ID_3,  int lep_ID_4,  int lep_ID_5){

    int count = 0;
    if(lep_ID_0==0) return count;
    
    if (IFFClass_lep_0 == classChoice){
      if(abs(lep_ID_0)==flavourChoice){ 
        count++;
      }
      if(lep_ID_1==0) return count;   //Stop it counting 3lep events as 4lep with an unclassified
    }
    if (IFFClass_lep_1 == classChoice){
      if(abs(lep_ID_1)==flavourChoice){ 
        count++;
      }
      if(lep_ID_2==0) return count;
    } 
    if (IFFClass_lep_2 == classChoice){
      if(abs(lep_ID_2)==flavourChoice){ 
        count++;
      }
      if(lep_ID_3==0) return count;
    } 
    if (IFFClass_lep_3 == classChoice){
      if(abs(lep_ID_3)==flavourChoice){ 
        count++;
      }
      if(lep_ID_4==0) return count;
    } 
    if (IFFClass_lep_4 == classChoice){
      if(abs(lep_ID_4)==flavourChoice){ 
        count++;
      }
      if(lep_ID_5==0) return count;
    } 
    if (IFFClass_lep_5 == classChoice){
      if(abs(lep_ID_5)==flavourChoice){ 
        count++;
      }
    } 
    return count;
  }
'''



def get_function_choices():

  function_choices = {}

  function_choices['N_E_IFF_Unclassified'] = "getNumOfIFFClassFlavour(-1, 11, IFFClass_lep_0, IFFClass_lep_1, IFFClass_lep_2, IFFClass_lep_3, IFFClass_lep_4, IFFClass_lep_5, lep_ID_0, lep_ID_1, lep_ID_2, lep_ID_3, lep_ID_4, lep_ID_5)"
  function_choices['N_M_IFF_Unclassified'] = "getNumOfIFFClassFlavour(-1, 13, IFFClass_lep_0, IFFClass_lep_1, IFFClass_lep_2, IFFClass_lep_3, IFFClass_lep_4, IFFClass_lep_5, lep_ID_0, lep_ID_1, lep_ID_2, lep_ID_3, lep_ID_4, lep_ID_5)"
  function_choices['N_E_IFF_KnownUnknown'] = "getNumOfIFFClassFlavour(0, 11, IFFClass_lep_0, IFFClass_lep_1, IFFClass_lep_2, IFFClass_lep_3, IFFClass_lep_4, IFFClass_lep_5, lep_ID_0, lep_ID_1, lep_ID_2, lep_ID_3, lep_ID_4, lep_ID_5)"
  function_choices['N_M_IFF_KnownUnknown'] = "getNumOfIFFClassFlavour(0, 13, IFFClass_lep_0, IFFClass_lep_1, IFFClass_lep_2, IFFClass_lep_3, IFFClass_lep_4, IFFClass_lep_5, lep_ID_0, lep_ID_1, lep_ID_2, lep_ID_3, lep_ID_4, lep_ID_5)"
  function_choices['N_E_IFF_Bdecays'] = "getNumOfIFFClassFlavour(5, 11, IFFClass_lep_0, IFFClass_lep_1, IFFClass_lep_2, IFFClass_lep_3, IFFClass_lep_4, IFFClass_lep_5, lep_ID_0, lep_ID_1, lep_ID_2, lep_ID_3, lep_ID_4, lep_ID_5)"
  function_choices['N_M_IFF_Bdecays'] = "getNumOfIFFClassFlavour(5, 13, IFFClass_lep_0, IFFClass_lep_1, IFFClass_lep_2, IFFClass_lep_3, IFFClass_lep_4, IFFClass_lep_5, lep_ID_0, lep_ID_1, lep_ID_2, lep_ID_3, lep_ID_4, lep_ID_5)"
  function_choices['N_E_IFF_Cdecays'] = "getNumOfIFFClassFlavour(6, 13, IFFClass_lep_0, IFFClass_lep_1, IFFClass_lep_2, IFFClass_lep_3, IFFClass_lep_4, IFFClass_lep_5, lep_ID_0, lep_ID_1, lep_ID_2, lep_ID_3, lep_ID_4, lep_ID_5)"
  function_choices['N_M_IFF_Cdecays'] = "getNumOfIFFClassFlavour(6, 13, IFFClass_lep_0, IFFClass_lep_1, IFFClass_lep_2, IFFClass_lep_3, IFFClass_lep_4, IFFClass_lep_5, lep_ID_0, lep_ID_1, lep_ID_2, lep_ID_3, lep_ID_4, lep_ID_5)"
  function_choices['N_E_IFF_LightHadDecays'] = "getNumOfIFFClassFlavour(7, 11, IFFClass_lep_0, IFFClass_lep_1, IFFClass_lep_2, IFFClass_lep_3, IFFClass_lep_4, IFFClass_lep_5, lep_ID_0, lep_ID_1, lep_ID_2, lep_ID_3, lep_ID_4, lep_ID_5)"
  function_choices['N_M_IFF_LightHadDecays'] = "getNumOfIFFClassFlavour(7, 13, IFFClass_lep_0, IFFClass_lep_1, IFFClass_lep_2, IFFClass_lep_3, IFFClass_lep_4, IFFClass_lep_5, lep_ID_0, lep_ID_1, lep_ID_2, lep_ID_3, lep_ID_4, lep_ID_5)"

  return function_choices


if __name__ == '__main__':

  tree = 'nominal'
  parser = argparse.ArgumentParser("Running model evaluation")

  parser.add_argument("-i","--inputDir",action="store", help="Set the input directory", default="/data/at3/common/multilepton/VLL_production/nominal", required=False)
    
  parser.add_argument("-o","--outputDir",action="store", help="Set the output directory", default="/data/at3/common/multilepton/VLL_production/addVarTest", required=False)
   

  parser.add_argument("-first","--First",action="store", help="Set the first file to run over", 
                        default=-1, required=False, type=int)
    
  parser.add_argument("-last","--Last",action="store", help="Set the last file to run over", 
                        default=-1, required=False, type=int)
  args = parser.parse_args()

  first = args.First
  last = args.Last
  input_dir = args.inputDir
  save_dir = args.outputDir

  ROOT.gInterpreter.Declare(getNumOfIFFClassFlavour)
  #ROOT.EnableImplicitMT() Don't use as shuffles outputs
  
  #Loop over predefined number of files
  #input_dir = '/data/at3/common/multilepton/VLL_production/nominal'
  #save_dir = '/data/at3/common/multilepton/VLL_production/nominal_v2'

  opts = ROOT.RDF.RSnapshotOptions()
  opts.fLazy = False
  opts.fMode = "RECREATE"

  function_choices = get_function_choices()


  all_root_files = find_root_files(input_dir, '', [])
  
  for i, file in enumerate(all_root_files):
    
    if i < first and first!=-1:
        continue
    if i > last and last!=-1:
        break
    
    print(f"Running file {file}. {i} / {len(all_root_files)}")

    # Do script
    frame = RDataFrame(tree, file)
    start_branches=len([x for x in frame.GetColumnNames()])

    #Add the functions
    for var, func in function_choices.items():
      frame = frame.Define(var, func)

    end_branches=len([x for x in frame.GetColumnNames()])
    
    print(f"Finished running functions... added {end_branches-start_branches} branches.")
    print(end_branches, " branches.")
    finalOutVars = ROOT.vector('string')()
    nBrnch=len([finalOutVars.push_back(x) for x in frame.GetColumnNames()])



    save_path = file.split(os.path.basename(input_dir))[1]
    if save_path[0] == '/':
        save_path = save_path[1:]

    whole_out_string = os.path.join(save_dir, save_path)
    if not os.path.exists(os.path.split(whole_out_string)[0]):
      os.makedirs(os.path.split(whole_out_string)[0])

    print(f"Saving file: {whole_out_string}")
    frame.Snapshot("nominal", whole_out_string, finalOutVars, opts)
    del frame

print("Finished running script... closing.")

  

