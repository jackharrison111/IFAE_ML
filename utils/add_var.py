# Add here
import ROOT

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
ROOT.gInterpreter.Declare(getNumOfIFFClassFlavour)

#ROOT.gInterpreter.Declare('#include <functions.h>')

#ROOT.EnableImplicitMT()


#Use example file
#TODO - look how to run over all files
import os
file_root = '/data/at3/common/multilepton/VLL_production/nominal'
save_root = '/data/at3/common/multilepton/VLL_production/nominal_v2'
file = 'mc16a/364176.root'



from ROOT import RDataFrame
frame = RDataFrame("nominal", os.path.join(file_root,file))


if not os.path.exists(os.path.join(save_root,'mc16a')):
  os.makedirs(os.path.join(save_root,'mc16a'))

opts = ROOT.RDF.RSnapshotOptions()
opts.fLazy = False
opts.fMode = "RECREATE"

nBrnch=len([x for x in frame.GetColumnNames()])
print(nBrnch)

frame = frame.Define("N_E_IFF_Unclassified", "getNumOfIFFClassFlavour(-1, 11, IFFClass_lep_0, IFFClass_lep_1, IFFClass_lep_2, IFFClass_lep_3, IFFClass_lep_4, IFFClass_lep_5, lep_ID_0, lep_ID_1, lep_ID_2, lep_ID_3, lep_ID_4, lep_ID_5)")


finalOutVars = ROOT.vector('string')()
nBrnch=len([finalOutVars.push_back(x) for x in frame.GetColumnNames()])
print(nBrnch)


#Try making validation histograms of the variable? 


frame.Snapshot("nominal",os.path.join(save_root, file),finalOutVars,opts)

