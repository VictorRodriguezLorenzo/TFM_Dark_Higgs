import ROOT
import uproot
import pandas as pd
import numpy as np
import subprocess
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, roc_auc_score

import random

import pandas as pd
import numpy as np

# Modelling
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
from sklearn.datasets import make_classification

import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, roc_auc_score

import matplotlib.pyplot as plt

import tensorflow.keras
from keras.utils import np_utils
import tensorflow.keras.callbacks as cb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras import callbacks
from tensorflow.keras.models import load_model
ROOT.EnableImplicitMT()

####### CONTROL PARAMETERS FOR THE WHOLE CODE #######

ANALYSIS_NAME = "dnn_low_mass_model"
choose_RFC = False  # False sets it for the DNN
parametric = False
save_model = True

loaded_model = False 

#MODEL_NAME = "/afs/cern.ch/user/v/victorr/private/DarkHiggs/Full2018_v7/Machile_Learning/Models/model_parametric_model_DNN_.h5"
MODEL_NAME = "/afs/cern.ch/user/v/victorr/private/DarkHiggs/Full2018_v7/Machile_Learning/Models/random_forest_parametric_model_RFC_.pkl"
#####################################################


var = [
  'lep_pt1',
  'lep_pt2',
  #'lep_phi1',
  #'lep_phi2',
  'lep_eta1',
  'lep_eta2',#
  'mll',
  'mth',
  'mtw1',#
  'mtw2',
  'ptll',
  'drll',
  #'dphilmet1',
  #'dphilmet2',
  'dphill',#
  'PuppiMET_pt',
  #'PuppiMET_phi',
  'detall',
  'mpmet',
  'recoil',
  #'yll',
  'mR',
  'mT2',#
  'mTe',#
  'mTi',
  'upara',
  'uperp',#
  #'dphil1tkmet',
  #'dphil2tkmet',
  'dphilmet',#
  'dphillmet',
  'mcoll',#
  'mcollWW',
  
  'dPhillStar',#
  'dPhill_Zp',
  'Cos_theta',#
  'Theta_ll',
  'dPhill_MET',
  
  'first_btag_ID',
  'second_btag_ID',
  # 1-jet ---
  #'dphilep1jet1',
  #'dphilep2jet1',
  #'btagDeepFlavB',
  # 2-jet -----
  #'mjj',
  #'Ctot',
  #'detajj',
  #'dphilep1jet1', 
  #'dphilep2jet1', 
  #'dphilep1jet2',
  #'dphilep2jet2',
  #'btagDeepFlavB', 
  #'btagDeepFlavB_1',
  #'D_VBF_QCD',
  #'D_VBF_VH',
  #'D_QCD_VH',
  #'D_VBF_DY',
]

print("")
print("Starting to load the events for the analysis------------------------------------------------------------------------------------------")
print("")

files_DH_temp = {}

from mkShapesRDF.lib.searchDM_files import SearchDMFiles

searchDMFiles = SearchDMFiles()

limitFiles = -1
redirector = ""

def nanoGetSampleDMFiles(path, name):
    _files = searchDMFiles.searchDMFiles(path, name, redirector=redirector)
    print("\n", name, _files)
    if limitFiles != -1 and len(_files) > limitFiles:
        return _files[:limitFiles]
    else:
        return _files



signalDirectory_2016 = '/eos/user/r/rocio/MonoH/Summer16_102X_nAODv7_Full2016v7/MCl1loose2016v7__MCCorr2016v7__l2loose__l2tightOR2016v7'
signalDirectory_2017 = '/eos/user/r/rocio/MonoH/Fall2017_102X_nAODv7_Full2017v7/MCl1loose2017v7__MCCorr2017v7__l2loose__l2tightOR2017v7'
signalDirectory_2018 = '/eos/user/r/rocio/MonoH/Autumn18_102X_nAODv7_Full2018v7/MCl1loose2018v7__MCCorr2018v7__l2loose__l2tightOR2018v7'  


mcDirectory_2016 = '/eos/cms/store/group/phys_higgs/cmshww/amassiro/HWWNano/Summer16_102X_nAODv7_Full2016v7/MCl1loose2016v7__MCCorr2016v7__l2loose__l2tightOR2016v7'
mcDirectory_2017 = '/eos/cms/store/group/phys_higgs/cmshww/amassiro/HWWNano/Fall2017_102X_nAODv7_Full2017v7/MCl1loose2017v7__MCCorr2017v7__l2loose__l2tightOR2017v7'
mcDirectory_2018 = '/eos/cms/store/group/phys_higgs/cmshww/amassiro/HWWNano/Autumn18_102X_nAODv7_Full2018v7/MCl1loose2018v7__MCCorr2018v7__l2loose__l2tightOR2018v7'

"""
# ONE MASS
mhs = ['300']
mDM = ['150']
mZp = ['800']
"""
"""
# ALL MASSES
mhs = ['160', '180', '200', '300']
mDM = ['100', '150', '200', '300']
mZp = ['200', '300', '400', '500', '800', '1000', '1200', '1500', '2000', '2500']
"""
"""
# HIGS MASS OF ms
mhs = ['300']
mDM = ['100', '150', '200', '300']
mZp = ['200', '300', '400', '500', '800', '1000', '1200', '1500', '2000', '2500']
"""

# LOW MASS OF ms
mhs = ['160']
mDM = ['100', '150', '200', '300']
mZp = ['200', '300', '400', '500', '800', '1000', '1200', '1500', '2000', '2500']


for DM in mDM:
    for hs in mhs:
        for Zp in mZp:
            files_DH_temp[hs + "_" + DM + "_"+ Zp] = { "names": nanoGetSampleDMFiles(signalDirectory_2016, 'DarkHiggs_MonoHs_HsToWWTo2l2nu_mhs_' + hs + '_mx_' + DM  + '_mZp_' + Zp) + nanoGetSampleDMFiles(signalDirectory_2017, 'DarkHiggs_MonoHs_HsToWWTo2l2nu_mhs_' + hs + '_mx_' + DM  + '_mZp_' + Zp), "mhs": hs, "mDM": DM, "mZp": Zp}


files_DH = files_DH_temp.copy()
for key, value in files_DH_temp.items():
    if isinstance(value["names"], list) and len(value["names"]) == 0:
        del files_DH[key]



print("")
print("Files for Dark Higgs loaded----------------------------------------------------------------------------------------------------------")
print("")

#DATA FROM 2016 AND 2017
files_WW = nanoGetSampleDMFiles(mcDirectory_2016, 'WWTo2L2Nu') + nanoGetSampleDMFiles(mcDirectory_2017, 'WWTo2L2Nu')

files_WWewk = nanoGetSampleDMFiles(mcDirectory_2016, 'WpWmJJ_EWK') + nanoGetSampleDMFiles(mcDirectory_2017, 'WpWmJJ_EWK')

""" NOT FOUND IN 2016
files_ggWW = nanoGetSampleDMFiles(mcDirectory_2016, 'GluGluToWWToENEN') + \
    nanoGetSampleDMFiles(mcDirectory_2016, 'GluGluToWWToENMN') + \
    nanoGetSampleDMFiles(mcDirectory_2016, 'GluGluToWWToENTN') + \
    nanoGetSampleDMFiles(mcDirectory_2016, 'GluGluToWWToMNEN') + \
    nanoGetSampleDMFiles(mcDirectory_2016, 'GluGluToWWToMNMN') + \
    nanoGetSampleDMFiles(mcDirectory_2016, 'GluGluToWWToMNTN') + \
    nanoGetSampleDMFiles(mcDirectory_2016, 'GluGluToWWToTNEN') + \
    nanoGetSampleDMFiles(mcDirectory_2016, 'GluGluToWWToTNMN') + \
    nanoGetSampleDMFiles(mcDirectory_2016, 'GluGluToWWToTNTN') + \

files_ggWW =    nanoGetSampleDMFiles(mcDirectory_2017, 'GluGluToWWToENEN') + \
    nanoGetSampleDMFiles(mcDirectory_2017, 'GluGluToWWToENMN') + \
    nanoGetSampleDMFiles(mcDirectory_2017, 'GluGluToWWToENTN') + \
    nanoGetSampleDMFiles(mcDirectory_2017, 'GluGluToWWToMNEN') + \
    nanoGetSampleDMFiles(mcDirectory_2017, 'GluGluToWWToMNMN') + \
    nanoGetSampleDMFiles(mcDirectory_2017, 'GluGluToWWToMNTN') + \
    nanoGetSampleDMFiles(mcDirectory_2017, 'GluGluToWWToTNEN') + \
    nanoGetSampleDMFiles(mcDirectory_2017, 'GluGluToWWToTNMN') + \
    nanoGetSampleDMFiles(mcDirectory_2017, 'GluGluToWWToTNTN')
"""
files_top = nanoGetSampleDMFiles(mcDirectory_2016, 'TTTo2L2Nu') + \
    nanoGetSampleDMFiles(mcDirectory_2017, 'TTTo2L2Nu')
#    nanoGetSampleDMFiles(mcDirectory_2016,'ST_s-channel') + \
#    nanoGetSampleDMFiles(mcDirectory_2016,'ST_t-channel_top') +  \
#    nanoGetSampleDMFiles(mcDirectory_2016,'ST_tW_antitop') + \
#    nanoGetSampleDMFiles(mcDirectory_2016,'ST_tW_top') + \
#    nanoGetSampleDMFiles(mcDirectory_2017,'ST_s-channel') + \
#    nanoGetSampleDMFiles(mcDirectory_2017,'ST_t-channel_top') +  \
#    nanoGetSampleDMFiles(mcDirectory_2017,'ST_tW_antitop') + \
#    nanoGetSampleDMFiles(mcDirectory_2017,'ST_tW_top')
"""

files_WW = nanoGetSampleDMFiles(mcDirectory_2018, 'WWTo2L2Nu')

files_WWewk = nanoGetSampleDMFiles(mcDirectory_2018, 'WpWmJJ_EWK')

files_ggWW = nanoGetSampleDMFiles(mcDirectory_2018, 'GluGluToWWToENEN') + \
    nanoGetSampleDMFiles(mcDirectory_2018, 'GluGluToWWToENMN') + \
    nanoGetSampleDMFiles(mcDirectory_2018, 'GluGluToWWToENTN') + \
    nanoGetSampleDMFiles(mcDirectory_2018, 'GluGluToWWToMNEN') + \
    nanoGetSampleDMFiles(mcDirectory_2018, 'GluGluToWWToMNMN') + \
    nanoGetSampleDMFiles(mcDirectory_2018, 'GluGluToWWToMNTN') + \
    nanoGetSampleDMFiles(mcDirectory_2018, 'GluGluToWWToTNEN') + \
    nanoGetSampleDMFiles(mcDirectory_2018, 'GluGluToWWToTNMN') + \
    nanoGetSampleDMFiles(mcDirectory_2018, 'GluGluToWWToTNTN') 

files_top = nanoGetSampleDMFiles(mcDirectory_2018, 'TTTo2L2N') + \
    nanoGetSampleDMFiles(mcDirectory_2018,'ST_s-channel_ext1') + \
    nanoGetSampleDMFiles(mcDirectory_2018,'ST_s-channel_ext1') + \
    nanoGetSampleDMFiles(mcDirectory_2018,'ST_t-channel_top') +  \
    nanoGetSampleDMFiles(mcDirectory_2018,'ST_tW_antitop_ext1') + \
    nanoGetSampleDMFiles(mcDirectory_2018,'ST_tW_top_ext1') 
"""



files_BKG = files_WW + files_top 
print("")
print("Background files processed-----------------------------------------------------------------------------------------------------------")
print("")

dataframes = {}
for key, df in files_DH.items():
    dataframes[key] = ROOT.RDataFrame("Events", files_DH[key]["names"])

df_bkg = ROOT.RDataFrame("Events", files_BKG)

print("")
print("Dataframes created-------------------------------------------------------------------------------------------------------------------")
print("")

ROOT.gROOT.ProcessLine(
    """
    template
    <typename container>
    float Alt(container c, int index, float alt){
        if (index < c.size()) {
            return c[index];
        }
        else{
            return alt;
        }
    }
    """
)

######
###### Construct boosted angular variables
######

ROOT.gInterpreter.Declare(
"""
    #include <TMath.h>
    #include <algorithm>
    #include <TLorentzVector.h>
    #include <iostream>
    #include "ROOT/RVec.hxx"
    #include <Math/Vector3D.h>
    #include <Math/VectorUtil.h>


    using namespace ROOT;
    using namespace ROOT::VecOps;

    TLorentzVector boostinv(TLorentzVector q, TLorentzVector pboost) {
    TLorentzVector qprime(0.0, 0.0, 0.0, 0.0);

    double rmboost = pboost.E() * pboost.E() - pboost.X() * pboost.X() - pboost.Y() * pboost.Y() - pboost.Z() * pboost.Z();
    if (rmboost > 0.0) {
        rmboost = TMath::Sqrt(rmboost);
    }
    else {
        rmboost = 0.0;
    }

    double aux = (q.E() * pboost.E() - q.X() * pboost.X() - q.Y() * pboost.Y() - q.Z() * pboost.Z()) / rmboost;
    double aaux = (aux + q.E()) / (pboost.E() + rmboost);

    double qprimeE = aux;
    double qprimeX = q.X() - aaux * pboost.X();
    double qprimeY = q.Y() - aaux * pboost.Y();
    double qprimeZ = q.Z() - aaux * pboost.Z();

    qprime.SetPxPyPzE(qprimeX, qprimeY, qprimeZ, qprimeE);

    return qprime;
    }
 
    double dPhillStar(
                RVecF Lepton_pt,
                RVecF Lepton_eta,
                RVecF Lepton_phi,
                double PuppiMET_pt,
                double PuppiMET_phi) {
    TLorentzVector L1, L2, MET, Zp;

    L1.SetPtEtaPhiM(Lepton_pt[0], Lepton_eta[0], Lepton_phi[0], 0.);
    L2.SetPtEtaPhiM(Lepton_pt[1], Lepton_eta[1], Lepton_phi[1], 0.);
    MET.SetPtEtaPhiM(PuppiMET_pt, 0, PuppiMET_phi, 0.);


    ROOT::Math::PtEtaPhiEVector MET_vector;
    MET_vector.SetCoordinates(MET.Pt(), MET.Eta(), MET.Phi(), MET.E());

    Zp = L1 + L2 + MET;

    ROOT::Math::XYZVector METvector;
    METvector = MET_vector.BoostToCM();

    ROOT::Math::XYZVector L1_vector;
    ROOT::Math::XYZVector L2_vector;

    ROOT::Math::PtEtaPhiEVector l1;
    ROOT::Math::PtEtaPhiEVector l2;
    ROOT::Math::PtEtaPhiEVector met;

    l1.SetCoordinates(L1.Pt(), L1.Eta(), L1.Phi(), L1.E());
    l2.SetCoordinates(L2.Pt(), L2.Eta(), L2.Phi(), L2.E());
    met.SetCoordinates(MET.Pt(), MET.Eta(), MET.Phi(), MET.E());

    L1_vector = ROOT::Math::VectorUtil::boost(l1, METvector);
    L2_vector = ROOT::Math::VectorUtil::boost(l2, METvector);

    float metpt = MET.Pt();
    float metphi_val = MET.Phi();

    float dphillStar = ROOT::Math::VectorUtil::DeltaPhi(L1_vector, L2_vector);

    return dphillStar;
}
"""
)
ROOT.gInterpreter.Declare(
"""
    double dPhill_Zp(
                RVecF Lepton_pt,
                RVecF Lepton_eta,
                RVecF Lepton_phi,
                double PuppiMET_pt,
                double PuppiMET_phi) {
    TLorentzVector L1, L2, MET, Zp;

    L1.SetPtEtaPhiM(Lepton_pt[0], Lepton_eta[0], Lepton_phi[0], 0.);
    L2.SetPtEtaPhiM(Lepton_pt[1], Lepton_eta[1], Lepton_phi[1], 0.);
    MET.SetPtEtaPhiM(PuppiMET_pt, 0, PuppiMET_phi, 0.);


    ROOT::Math::PtEtaPhiEVector MET_vector;
    MET_vector.SetCoordinates(MET.Pt(), MET.Eta(), MET.Phi(), MET.E());

    Zp = L1 + L2 + MET;

    ROOT::Math::XYZVector METvector;
    METvector = MET_vector.BoostToCM();

    ROOT::Math::XYZVector L1_vector;
    ROOT::Math::XYZVector L2_vector;

    ROOT::Math::PtEtaPhiEVector l1;
    ROOT::Math::PtEtaPhiEVector l2;
    ROOT::Math::PtEtaPhiEVector met;

    l1.SetCoordinates(L1.Pt(), L1.Eta(), L1.Phi(), L1.E());
    l2.SetCoordinates(L2.Pt(), L2.Eta(), L2.Phi(), L2.E());
    met.SetCoordinates(MET.Pt(), MET.Eta(), MET.Phi(), MET.E());

    L1_vector = ROOT::Math::VectorUtil::boost(l1, METvector);
    L2_vector = ROOT::Math::VectorUtil::boost(l2, METvector);

    float metpt = MET.Pt();
    float metphi_val = MET.Phi();



    ROOT::Math::PtEtaPhiEVector Zp_vector;
    Zp_vector = l1+l2+met;

    ROOT::Math::XYZVector Zpvector;
    Zpvector = Zp_vector.BoostToCM();

    ROOT::Math::XYZVector L1_vector_toZp;
        ROOT::Math::XYZVector L2_vector_toZp;
    ROOT::Math::XYZVector MET_vector_toZp;

    L1_vector_toZp = ROOT::Math::VectorUtil::boost(l1, Zpvector);
    L2_vector_toZp = ROOT::Math::VectorUtil::boost(l2, Zpvector);

    MET_vector_toZp = ROOT::Math::VectorUtil::boost(met, Zpvector);

    float dphill_Zp = ROOT::Math::VectorUtil::Angle(L1_vector_toZp, L2_vector_toZp);

    return dphill_Zp;
}
"""
)

ROOT.gInterpreter.Declare(
"""
    double Cos_theta(
                RVecF Lepton_pt,
                RVecF Lepton_eta,
                RVecF Lepton_phi,
                double PuppiMET_pt,
                double PuppiMET_phi) {
    TLorentzVector L1, L2, MET, Zp;

    L1.SetPtEtaPhiM(Lepton_pt[0], Lepton_eta[0], Lepton_phi[0], 0.);
    L2.SetPtEtaPhiM(Lepton_pt[1], Lepton_eta[1], Lepton_phi[1], 0.);
    MET.SetPtEtaPhiM(PuppiMET_pt, 0, PuppiMET_phi, 0.);


    ROOT::Math::PtEtaPhiEVector MET_vector;
    MET_vector.SetCoordinates(MET.Pt(), MET.Eta(), MET.Phi(), MET.E());

    Zp = L1 + L2 + MET;

    ROOT::Math::XYZVector METvector;
    METvector = MET_vector.BoostToCM();

    ROOT::Math::XYZVector L1_vector;
    ROOT::Math::XYZVector L2_vector;

    ROOT::Math::PtEtaPhiEVector l1;
    ROOT::Math::PtEtaPhiEVector l2;
    ROOT::Math::PtEtaPhiEVector met;

    l1.SetCoordinates(L1.Pt(), L1.Eta(), L1.Phi(), L1.E());
    l2.SetCoordinates(L2.Pt(), L2.Eta(), L2.Phi(), L2.E());
    met.SetCoordinates(MET.Pt(), MET.Eta(), MET.Phi(), MET.E());

    L1_vector = ROOT::Math::VectorUtil::boost(l1, METvector);
    L2_vector = ROOT::Math::VectorUtil::boost(l2, METvector);

    float metpt = MET.Pt();
    float metphi_val = MET.Phi();

    ROOT::Math::PtEtaPhiEVector Zp_vector;
    Zp_vector = l1+l2+met;

    ROOT::Math::XYZVector Zpvector;
    Zpvector = Zp_vector.BoostToCM();

    ROOT::Math::XYZVector L1_vector_toZp;
        ROOT::Math::XYZVector L2_vector_toZp;
    ROOT::Math::XYZVector MET_vector_toZp;

    L1_vector_toZp = ROOT::Math::VectorUtil::boost(l1, Zpvector);
    L2_vector_toZp = ROOT::Math::VectorUtil::boost(l2, Zpvector);

    MET_vector_toZp = ROOT::Math::VectorUtil::boost(met, Zpvector);

    float theta_1 = ROOT::Math::VectorUtil::Angle(L1_vector_toZp, Zp);
    float theta_2 = ROOT::Math::VectorUtil::Angle(L2_vector_toZp, Zp);

    float cos_theta1 = TMath::Cos(theta_1);
    float cos_theta2 = TMath::Cos(theta_2);

    float cos_theta = min(cos_theta1, cos_theta2);
    return cos_theta;
}
"""
)

ROOT.gInterpreter.Declare(
"""
    double Theta_ll(
                RVecF Lepton_pt,
                RVecF Lepton_eta,
                RVecF Lepton_phi,
                double PuppiMET_pt,
                double PuppiMET_phi) {
    TLorentzVector L1, L2, MET, Zp;

    L1.SetPtEtaPhiM(Lepton_pt[0], Lepton_eta[0], Lepton_phi[0], 0.);
    L2.SetPtEtaPhiM(Lepton_pt[1], Lepton_eta[1], Lepton_phi[1], 0.);
    MET.SetPtEtaPhiM(PuppiMET_pt, 0, PuppiMET_phi, 0.);


    ROOT::Math::PtEtaPhiEVector MET_vector;
    MET_vector.SetCoordinates(MET.Pt(), MET.Eta(), MET.Phi(), MET.E());

    Zp = L1 + L2 + MET;

    ROOT::Math::XYZVector METvector;
    METvector = MET_vector.BoostToCM();

    ROOT::Math::XYZVector L1_vector;
    ROOT::Math::XYZVector L2_vector;

    ROOT::Math::PtEtaPhiEVector l1;
    ROOT::Math::PtEtaPhiEVector l2;
    ROOT::Math::PtEtaPhiEVector met;

    l1.SetCoordinates(L1.Pt(), L1.Eta(), L1.Phi(), L1.E());
    l2.SetCoordinates(L2.Pt(), L2.Eta(), L2.Phi(), L2.E());
    met.SetCoordinates(MET.Pt(), MET.Eta(), MET.Phi(), MET.E());

    L1_vector = ROOT::Math::VectorUtil::boost(l1, METvector);
    L2_vector = ROOT::Math::VectorUtil::boost(l2, METvector);

    float metpt = MET.Pt();
    float metphi_val = MET.Phi();

    ROOT::Math::PtEtaPhiEVector Zp_vector;
    Zp_vector = l1+l2+met;

    ROOT::Math::XYZVector Zpvector;
    Zpvector = Zp_vector.BoostToCM();

    ROOT::Math::XYZVector L1_vector_toZp;
        ROOT::Math::XYZVector L2_vector_toZp;
    ROOT::Math::XYZVector MET_vector_toZp;

    L1_vector_toZp = ROOT::Math::VectorUtil::boost(l1, Zpvector);
    L2_vector_toZp = ROOT::Math::VectorUtil::boost(l2, Zpvector);

    MET_vector_toZp = ROOT::Math::VectorUtil::boost(met, Zpvector);

    float theta_ll = ROOT::Math::VectorUtil::Angle(L1_vector_toZp+L2_vector_toZp, Zp);
    
    return theta_ll;
}
"""
)


ROOT.gInterpreter.Declare(
"""
    double dPhill_MET(
                RVecF Lepton_pt,
                RVecF Lepton_eta,
                RVecF Lepton_phi,
                double PuppiMET_pt,
                double PuppiMET_phi) {
    TLorentzVector L1, L2, MET, Zp;

    L1.SetPtEtaPhiM(Lepton_pt[0], Lepton_eta[0], Lepton_phi[0], 0.);
    L2.SetPtEtaPhiM(Lepton_pt[1], Lepton_eta[1], Lepton_phi[1], 0.);
    MET.SetPtEtaPhiM(PuppiMET_pt, 0, PuppiMET_phi, 0.);


    ROOT::Math::PtEtaPhiEVector MET_vector;
    MET_vector.SetCoordinates(MET.Pt(), MET.Eta(), MET.Phi(), MET.E());

    Zp = L1 + L2 + MET;

    ROOT::Math::XYZVector METvector;
    METvector = MET_vector.BoostToCM();

    ROOT::Math::XYZVector L1_vector;
    ROOT::Math::XYZVector L2_vector;

    ROOT::Math::PtEtaPhiEVector l1;
    ROOT::Math::PtEtaPhiEVector l2;
    ROOT::Math::PtEtaPhiEVector met;

    l1.SetCoordinates(L1.Pt(), L1.Eta(), L1.Phi(), L1.E());
    l2.SetCoordinates(L2.Pt(), L2.Eta(), L2.Phi(), L2.E());
    met.SetCoordinates(MET.Pt(), MET.Eta(), MET.Phi(), MET.E());

    L1_vector = ROOT::Math::VectorUtil::boost(l1, METvector);
    L2_vector = ROOT::Math::VectorUtil::boost(l2, METvector);

    float metpt = MET.Pt();
    float metphi_val = MET.Phi();

    ROOT::Math::PtEtaPhiEVector Zp_vector;
    Zp_vector = l1+l2+met;

    ROOT::Math::XYZVector Zpvector;
    Zpvector = Zp_vector.BoostToCM();

    ROOT::Math::XYZVector L1_vector_toZp;
        ROOT::Math::XYZVector L2_vector_toZp;
    ROOT::Math::XYZVector MET_vector_toZp;

    L1_vector_toZp = ROOT::Math::VectorUtil::boost(l1, Zpvector);
    L2_vector_toZp = ROOT::Math::VectorUtil::boost(l2, Zpvector);

    MET_vector_toZp = ROOT::Math::VectorUtil::boost(met, Zpvector);

    float dphill_MET = ROOT::Math::VectorUtil::Angle(L1_vector_toZp+L2_vector_toZp, MET_vector);

    return dphill_MET;
}
"""
)

print("")
print("New variables being declared for the use within the dataframes-----------------------------------------------------------------------------")
print("")

bWP = '0.1208'
bAlgo = 'DeepB'

for key, df in dataframes.items():
    dataframes[key] = dataframes[key].Define("dPhillStar", "dPhillStar(Lepton_pt, Lepton_eta, Lepton_phi, PuppiMET_pt, PuppiMET_phi)")
    dataframes[key] = dataframes[key].Define("dPhill_Zp", "dPhill_Zp(Lepton_pt, Lepton_eta, Lepton_phi, PuppiMET_pt, PuppiMET_phi)")
    dataframes[key] = dataframes[key].Define("Cos_theta", "Cos_theta(Lepton_pt, Lepton_eta, Lepton_phi, PuppiMET_pt, PuppiMET_phi)")
    dataframes[key] = dataframes[key].Define("Theta_ll", "Theta_ll(Lepton_pt, Lepton_eta, Lepton_phi, PuppiMET_pt, PuppiMET_phi)")
    dataframes[key] = dataframes[key].Define("dPhill_MET", "dPhill_MET(Lepton_pt, Lepton_eta, Lepton_phi, PuppiMET_pt, PuppiMET_phi)")
    dataframes[key] = dataframes[key].Define("first_btag_ID", 'Alt(Jet_btag{}, Alt(CleanJet_jetIdx, 0, -1), -1)'.format(bAlgo))
    dataframes[key] = dataframes[key].Define("second_btag_ID", 'Alt(Jet_btag{}, Alt(CleanJet_jetIdx, 1, -1), -1)'.format(bAlgo))
    dataframes[key] = dataframes[key].Define("lep_eta1", "Lepton_eta[0]")
    dataframes[key] = dataframes[key].Define("lep_eta2", "Lepton_eta[1]")
    dataframes[key] = dataframes[key].Define("lep_pt1", "Lepton_pt[0]")
    dataframes[key] = dataframes[key].Define("lep_pt2", "Lepton_pt[1]")
    dataframes[key] = dataframes[key].Define("bVeto", 'Sum(CleanJet_pt > 20. && abs(CleanJet_eta) < 2.5 && Take(Jet_btag{}, CleanJet_jetIdx) > {}) == 0'.format(bAlgo, bWP))    

df_bkg = df_bkg.Define("dPhillStar", "dPhillStar(Lepton_pt, Lepton_eta, Lepton_phi, PuppiMET_pt, PuppiMET_phi)")
df_bkg = df_bkg.Define("dPhill_Zp", "dPhill_Zp(Lepton_pt, Lepton_eta, Lepton_phi, PuppiMET_pt, PuppiMET_phi)")
df_bkg = df_bkg.Define("Cos_theta", "Cos_theta(Lepton_pt, Lepton_eta, Lepton_phi, PuppiMET_pt, PuppiMET_phi)")
df_bkg = df_bkg.Define("Theta_ll", "Theta_ll(Lepton_pt, Lepton_eta, Lepton_phi, PuppiMET_pt, PuppiMET_phi)")
df_bkg = df_bkg.Define("dPhill_MET", "dPhill_MET(Lepton_pt, Lepton_eta, Lepton_phi, PuppiMET_pt, PuppiMET_phi)")
df_bkg = df_bkg.Define("first_btag_ID", 'Alt(Jet_btag{}, Alt(CleanJet_jetIdx, 0, -1), -1)'.format(bAlgo))
df_bkg = df_bkg.Define("second_btag_ID", 'Alt(Jet_btag{}, Alt(CleanJet_jetIdx, 1, -1), -1)'.format(bAlgo))
df_bkg = df_bkg.Define("lep_eta1", "Lepton_eta[0]")
df_bkg = df_bkg.Define("lep_eta2", "Lepton_eta[1]")
df_bkg = df_bkg.Define("lep_pt1", "Lepton_pt[0]")
df_bkg = df_bkg.Define("lep_pt2", "Lepton_pt[1]")
df_bkg = df_bkg.Define("bVeto", 'Sum(CleanJet_pt > 20. && abs(CleanJet_eta) < 2.5 && Take(Jet_btag{}, CleanJet_jetIdx) > {}) == 0'.format(bAlgo, bWP))

print("")
print("New variables loaded within the dataframes-------------------------------------------------------------------------------------------")
print("")

#### 0 Jet
for key, df in dataframes.items(): #same
	dataframes[key] = dataframes[key].Filter("Lepton_pdgId[0]*Lepton_pdgId[1] == -11*13 && Lepton_pt[0]>25 && Lepton_pt[1]>20 && Alt(Lepton_pt,2, 0)<10 && PuppiMET_pt>20 && mpmet > 20. && ptll>30 && mth>50 && mll > 12 && drll<2.5 && bVeto")
df_bkg = df_bkg.Filter("Lepton_pdgId[0]*Lepton_pdgId[1] == -11*13 && Lepton_pt[0]>25 && Lepton_pt[1]>20 && Alt(Lepton_pt,2, 0)<10 && PuppiMET_pt>20 && mpmet > 20. && ptll>30 && mth>50 && mll > 12 && drll<2.5 && bVeto")
print("")
print("Events filtering---------------------------------------------------------------------------------------------------------------------")

columns = [
  'lep_pt1','lep_pt2','lep_eta1','mll','mth','mtw2','ptll','drll','PuppiMET_pt','detall','mpmet','recoil','mR','mTi','upara',
  'dphillmet','mcollWW','dPhill_Zp','Theta_ll','dPhill_MET','first_btag_ID','second_btag_ID', 'dphilmet', 'mcoll', 'mtw1', 'mTe', 'dphill', 'Cos_theta', 'dPhillStar', 'uperp', 'mT2', 'lep_eta2']#, 'mhs', 'mDM', 'mZp']
print("DONE!")
print("")
print("Creating pandas dataframes!----------------------------------------------------------------------------------------------------------")


numpy_dataframes = {}
for key, df in dataframes.items():
    numpy_dataframes[key] = df.AsNumpy(var)
dfBkg = df_bkg.AsNumpy(var)

pd_dataframes = {}
for key, df_sig in numpy_dataframes.items():
    pd_dataframes[key] = pd.DataFrame(df_sig)
    if parametric:
        pd_dataframes[key]['mhs'] = int(files_DH[key]["mhs"])
        pd_dataframes[key]['mDM'] = int(files_DH[key]["mDM"])
        pd_dataframes[key]['mZp'] = int(files_DH[key]["mZp"])
# For the background, assign differnt values to the masses for each event from the list of masses
Bkg = pd.DataFrame(dfBkg)
if parametric:
    Bkg['mhs'] = np.random.choice([eval(i) for i in mhs], len(Bkg))
    Bkg['mDM'] = np.random.choice([eval(i) for i in mDM], len(Bkg))
    Bkg['mZp'] = np.random.choice([eval(i) for i in mZp], len(Bkg))

    # Add the masses to the variables:
    var.append('mhs')
    var.append('mDM')
    var.append('mZp')

print("DONE!")
print("")
print("Mass points specified inside the pandas dataframes for their use inside the parametrized neural network------------------------------")
print("")

#### Categories!
for key, df_sig in pd_dataframes.items():
    pd_dataframes[key]['isSignal'] = np.ones(len(df_sig))
    pd_dataframes[key]['isBkg'] = np.zeros(len(df_sig))
    
Bkg['isSignal'] = np.zeros(len(Bkg))
Bkg['isBkg'] = np.ones(len(Bkg))

print("")
print("Categories inserted in the dataframes for isSignal and isBkg-------------------------------------------------------------------------")
print("")

# Make all the signals together the same size as the background
# Find the dataframe with the lowest number of events
min_length = min([len(df_sig) for key, df_sig in pd_dataframes.items()])
print("The min lenght for the signal dataframes is:", min_length)

# Sample the rest of the dataframes to the length of the dataframe with the lowest number of events
for key, df_sig in pd_dataframes.items():
    if len(df_sig) > min_length:
        pd_dataframes[key] = df_sig.sample(min_length)
# Concatenate the dataframes
Sig = pd.concat([df_sig for key, df_sig in pd_dataframes.items()])
print("")
print("Seeing how each of the dataframes for Signal and Background look:")
print(Sig)
print(Bkg)


print("Initial lenght of Sig:", len(Sig))
print("Initial lenght of Bkg:", len(Bkg))


# Sample the background to the length of the concatenated signal dataframe
if len(Sig) > len(Bkg):
    Sig = Sig.sample(len(Bkg))
else:
    Bkg = Bkg.sample(len(Sig))

print("")
print("Statistics after sampling")  
print("Statistics for Sig: " + str(len(Sig)))
print("Statistics for Bkg: " + str(len(Bkg)))
print("")

# # Concatenate each DataFrame in pd_dataframes with the 'Bkg' DataFrame
# concatenated_dfs = []
# for key, df in pd_dataframes.items():
#     concatenated_df = pd.concat([df, Bkg])
#     concatenated_dfs.append(concatenated_df)

# Concatenate all DataFrames in concatenated_dfs with the 'Bkg' DataFrame
df_all = pd.concat([Sig, Bkg])



print("Length of dataset: " + str(len(df_all)))
# for df in concatenated_dfs:
#      df.dropna(inplace=True)
df_all.dropna(inplace=True)
print("After removing NANs: " + str(len(df_all)))

X_train, X_test, Y_train, Y_test = train_test_split(df_all[var], df_all[['isSignal']], test_size=0.2, random_state=6)

print("")
print("Variables to study for the analysis:")
print(X_train)
print(Y_train)
print("")
print("DONE!")

if choose_RFC:
    ANALYSIS_NAME += "_RFC_"
else:
    ANALYSIS_NAME += "_DNN_"

if loaded_model:
    if choose_RFC:
        rfc = joblib.load(MODEL_NAME)
    else:
        model= load_model(MODEL_NAME)
else:
    if choose_RFC:       
        print("Start training the Random Forest!----------------------------------------------------------------------------------------------------")
        print("")

        rfc = xgb.XGBClassifier(max_depth=6)
        #rfc = xgb.XGBClassifier(max_depth=15)
        #rfc = xgb.XGBClassifier(max_depth=20)
        rfc.fit(X_train, Y_train)
    else:
        print("Start training the Deep Neural Network!----------------------------------------------------------------------------------------------------")
        print("")
        model = Sequential()

        model.add(Dense(128, activation='relu', input_dim=len(var)))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(8, activation='relu'))

        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(0.00015), metrics=['accuracy'])
        n_epochs = 300
        n_batch = 512
        training = model.fit(X_train[var].values, Y_train, epochs=n_epochs, validation_split=0.15, batch_size=n_batch,
                                     callbacks = [callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=1)],
                                     verbose=2, shuffle= True)


print("DONE!")
print("Save and plot test distributions-----------------------------------------------------------------------------------------------------")

if save_model and loaded_model == False:
    print("")
    print("The current used Machine Learning model is being saved:")
    print("")
    if choose_RFC:
        joblib.dump(rfc, './Models/random_forest_'+ANALYSIS_NAME+'.pkl')
    else:
        model.save('./Models/model_'+ANALYSIS_NAME+'.h5')



#### --------------------------------------------------------------------------------------------------
if choose_RFC:
    ######################### FOR THE RANDOM FOREST
    y_pred = rfc.predict_proba(X_test)
    y_pred_t = rfc.predict_proba(X_train)

    y_pred_L = []
    y_pred_t_L = []
    for i in range(len(y_pred)):
      y_pred_L.append(y_pred[i][1])

    for i in range(len(y_pred_t)):
      y_pred_t_L.append(y_pred_t[i][1])
    ########################
else:
    ######################### FOR THE DEEP NEURAL NETWORK
    y_pred = model.predict(X_test)
    y_pred_t = model.predict(X_train)

    y_pred_L = y_pred
    y_pred_t_L = y_pred_t


########################

#### SIGNAL category

discriminant = np.squeeze(np.asarray(y_pred_L))
true_labels = np.squeeze(np.asarray(Y_test['isSignal']))

discriminant0 = discriminant[np.array(true_labels == 0)]
discriminant1 = discriminant[np.array(true_labels == 1)]

discriminant_t = np.squeeze(np.asarray(y_pred_t_L))
true_labels_t = np.squeeze(np.asarray(Y_train['isSignal']))

discriminant0_t = discriminant_t[np.array(true_labels_t == 0)]
discriminant1_t = discriminant_t[np.array(true_labels_t == 1)]

if parametric:
    #### FOR THE DIFFERENT MASSES OF THE HIGGS
    # mhs=160
    filtered_indices_160 = np.where(X_train['mhs'] == 160)
    filtered_indices_160_test = np.where(X_test['mhs'] == 160)
    discriminant_t_160 = discriminant_t[filtered_indices_160[0]]
    print(filtered_indices_160[0])
    true_labels_t_160 = np.squeeze(np.asarray(Y_train['isSignal'].iloc[filtered_indices_160[0]]))

    discriminant0_t_160 = discriminant_t_160[np.array(true_labels_t_160 == 0)]
    discriminant1_t_160 = discriminant_t_160[np.array(true_labels_t_160 == 1)]
    # mhs=180
    filtered_indices_180 = np.where(X_train['mhs'] == 180)
    filtered_indices_180_test = np.where(X_test['mhs'] == 180)
    discriminant_t_180 = discriminant_t[filtered_indices_180[0]]
    true_labels_t_180 = np.squeeze(np.asarray(Y_train['isSignal'].iloc[filtered_indices_180[0]]))
    discriminant0_t_180 = discriminant_t_180[np.array(true_labels_t_180 == 0)]
    discriminant1_t_180 = discriminant_t_180[np.array(true_labels_t_180 == 1)]
    # mhs=200
    filtered_indices_200 = np.where(X_train['mhs'] == 200)
    filtered_indices_200_test = np.where(X_test['mhs'] == 200)
    discriminant_t_200 = discriminant_t[filtered_indices_200[0]]
    true_labels_t_200 = np.squeeze(np.asarray(Y_train['isSignal'].iloc[filtered_indices_200[0]]))
    discriminant0_t_200 = discriminant_t_200[np.array(true_labels_t_200 == 0)]
    discriminant1_t_200 = discriminant_t_200[np.array(true_labels_t_200 == 1)]
    # mhs=300
    filtered_indices_300 = np.where(X_train['mhs'] == 300)
    filtered_indices_300_test = np.where(X_test['mhs'] == 300)
    discriminant_t_300 = discriminant_t[filtered_indices_300[0]]
    true_labels_t_300 = np.squeeze(np.asarray(Y_train['isSignal'].iloc[filtered_indices_300[0]]))
    discriminant0_t_300 = discriminant_t_300[np.array(true_labels_t_300 == 0)]
    discriminant1_t_300 = discriminant_t_300[np.array(true_labels_t_300 == 1)]


    #### FOR THE DIFFERENT MASSES OF THE Z
    # mZp=200
    filtered_indices_200z = np.where(X_train['mZp'] == 200)
    filtered_indices_200z_test = np.where(X_test['mZp'] == 200)
    discriminant_t_200z = discriminant_t[filtered_indices_200z[0]]
    print(filtered_indices_200z[0])
    true_labels_t_200z = np.squeeze(np.asarray(Y_train['isSignal'].iloc[filtered_indices_200z[0]]))

    discriminant0_t_200z = discriminant_t_200z[np.array(true_labels_t_200z == 0)]
    discriminant1_t_200z = discriminant_t_200z[np.array(true_labels_t_200z == 1)]
    # mZp=500
    filtered_indices_500z = np.where(X_train['mZp'] == 500)
    filtered_indices_500z_test = np.where(X_test['mZp'] == 500)
    discriminant_t_500z = discriminant_t[filtered_indices_500z[0]]
    true_labels_t_500z = np.squeeze(np.asarray(Y_train['isSignal'].iloc[filtered_indices_500z[0]]))
    discriminant0_t_500z = discriminant_t_500z[np.array(true_labels_t_500z == 0)]
    discriminant1_t_500z = discriminant_t_500z[np.array(true_labels_t_500z == 1)]
    # mZp=1200
    filtered_indices_1200z = np.where(X_train['mZp'] == 1200)
    filtered_indices_1200z_test = np.where(X_test['mZp'] == 1200)
    discriminant_t_1200z = discriminant_t[filtered_indices_1200z[0]]
    true_labels_t_1200z = np.squeeze(np.asarray(Y_train['isSignal'].iloc[filtered_indices_1200z[0]]))
    discriminant0_t_1200z = discriminant_t_1200z[np.array(true_labels_t_1200z == 0)]
    discriminant1_t_1200z = discriminant_t_1200z[np.array(true_labels_t_1200z == 1)]
    # mZp=1500
    filtered_indices_1500z = np.where(X_train['mZp'] == 1500)
    filtered_indices_1500z_test = np.where(X_test['mZp'] == 1500)
    discriminant_t_1500z = discriminant_t[filtered_indices_1500z[0]]
    true_labels_t_1500z = np.squeeze(np.asarray(Y_train['isSignal'].iloc[filtered_indices_1500z[0]]))
    discriminant0_t_1500z = discriminant_t_1500z[np.array(true_labels_t_1500z == 0)]
    discriminant1_t_1500z = discriminant_t_1500z[np.array(true_labels_t_1500z == 1)]
    # mZp=2000
    filtered_indices_2000z = np.where(X_train['mZp'] == 2000)
    filtered_indices_2000z_test = np.where(X_test['mZp'] == 2000)
    discriminant_t_2000z = discriminant_t[filtered_indices_2000z[0]]
    true_labels_t_2000z = np.squeeze(np.asarray(Y_train['isSignal'].iloc[filtered_indices_2000z[0]]))
    discriminant0_t_2000z = discriminant_t_2000z[np.array(true_labels_t_2000z == 0)]
    discriminant1_t_2000z = discriminant_t_2000z[np.array(true_labels_t_2000z == 1)]
    # mZp=2500
    filtered_indices_2500z = np.where(X_train['mZp'] == 2500)
    filtered_indices_2500z_test = np.where(X_test['mZp'] == 2500)
    discriminant_t_2500z = discriminant_t[filtered_indices_2500z[0]]
    true_labels_t_2500z = np.squeeze(np.asarray(Y_train['isSignal'].iloc[filtered_indices_2500z[0]]))
    discriminant0_t_2500z = discriminant_t_2500z[np.array(true_labels_t_2500z == 0)]
    discriminant1_t_2500z = discriminant_t_2500z[np.array(true_labels_t_2500z == 1)]


    #### FOR THE DIFFERENT MASSES OF THE DARK MATTER
    # mDM=100
    filtered_indices_100x = np.where(X_train['mDM'] == 100)
    filtered_indices_100x_test = np.where(X_test['mDM'] == 100)
    discriminant_t_100x = discriminant_t[filtered_indices_100x[0]]
    true_labels_t_100x = np.squeeze(np.asarray(Y_train['isSignal'].iloc[filtered_indices_100x[0]]))

    discriminant0_t_100x = discriminant_t_100x[np.array(true_labels_t_100x == 0)]
    discriminant1_t_100x = discriminant_t_100x[np.array(true_labels_t_100x == 1)]
    # mDM=150
    filtered_indices_150x = np.where(X_train['mDM'] == 150)
    filtered_indices_150x_test = np.where(X_test['mDM'] == 150)
    discriminant_t_150x = discriminant_t[filtered_indices_150x[0]]
    true_labels_t_150x = np.squeeze(np.asarray(Y_train['isSignal'].iloc[filtered_indices_150x[0]]))
    discriminant0_t_150x = discriminant_t_150x[np.array(true_labels_t_150x == 0)]
    discriminant1_t_150x = discriminant_t_150x[np.array(true_labels_t_150x == 1)]
    # mDM=200
    filtered_indices_200x = np.where(X_train['mDM'] == 200)
    filtered_indices_200x_test = np.where(X_test['mDM'] == 200)
    discriminant_t_200x = discriminant_t[filtered_indices_200x[0]]
    true_labels_t_200x = np.squeeze(np.asarray(Y_train['isSignal'].iloc[filtered_indices_200x[0]]))
    discriminant0_t_200x = discriminant_t_200x[np.array(true_labels_t_200x == 0)]
    discriminant1_t_200x = discriminant_t_200x[np.array(true_labels_t_200x == 1)]
    # mDM=300
    filtered_indices_300x = np.where(X_train['mDM'] == 300)
    filtered_indices_300x_test = np.where(X_test['mDM'] == 300)
    discriminant_t_300x = discriminant_t[filtered_indices_300x[0]]
    true_labels_t_300x = np.squeeze(np.asarray(Y_train['isSignal'].iloc[filtered_indices_300x[0]]))
    discriminant0_t_300x = discriminant_t_300x[np.array(true_labels_t_300x == 0)]
    discriminant1_t_300x = discriminant_t_300x[np.array(true_labels_t_300x == 1)]

binning = np.linspace(0, 1, 51)


# Plot the discriminant distributions ----------------------------

print("")
print("Plotinng discriminant distribution---------------------------------------------------------------------------------------------------")
plt.clf()
plt.figure(num=None, figsize=(6, 6))
plt.subplot(111)
pdf0, bins0, patches0 = plt.hist(discriminant0, bins = binning, color = 'm', alpha = 0.0, histtype = 'stepfilled', linewidth = 1, edgecolor='r', density=True)
pdf1, bins1, patches1 = plt.hist(discriminant1, bins = binning, color = 'y', alpha = 0.0, histtype = 'stepfilled', linewidth = 1, edgecolor='b', density=True)

pdf0_t, bins0_t, patches0_t = plt.hist(discriminant0_t, bins = binning, color = 'r', alpha = 0.3, histtype = 'stepfilled', linewidth = 2, edgecolor='r', label = 'Backgrounds (train)', density=True)
pdf1_t, bins1_t, patches1_t = plt.hist(discriminant1_t, bins = binning, color = 'b', alpha = 0.3, histtype = 'stepfilled', linewidth = 2, edgecolor='b', label = 'Dark Higgs signal (train)', density=True)

plt.scatter(bins0[:-1]+ 0.5*(bins0[1:] - bins0[:-1]), pdf0, marker='.', c='m', s=30, alpha=0.8, label = 'Backgrounds')
plt.scatter(bins1[:-1]+ 0.5*(bins1[1:] - bins1[:-1]), pdf1, marker='.', c='y', s=30, alpha=0.8, label = 'Dark Higgs signal')

plt.legend(loc = 'upper center')
plt.ylabel('Density', fontsize = 12)
plt.xlabel('Random Forest discriminant', fontsize = 12)
plt.savefig('Discriminant_distribution'+ANALYSIS_NAME+'.png', dpi = 600)

plt.clf()
plt.figure(num=None, figsize=(6, 6))
plt.subplot(111)

if parametric:
    # mhs=160
    pdf0_t_160, bins0_t_160, patches0_t_160 = plt.hist(discriminant0_t_160, bins = binning, color = 'r', alpha = 0.3, histtype = 'stepfilled', linewidth = 2, edgecolor='r', density=True)
    plt.scatter(bins0_t_160[:-1]+ 0.5*(bins0_t_160[1:] - bins0_t_160[:-1]), pdf0_t_160, marker='.', c='r', s=30, alpha=0.8, label = f'Backgrounds $m_s=160$')

    # mhs=180
    pdf0_t_180, bins0_t_180, patches0_t_180 = plt.hist(discriminant0_t_180, bins = binning, color = 'g', alpha = 0.3, histtype = 'stepfilled', linewidth = 2, edgecolor='g', density=True)
    plt.scatter(bins0_t_180[:-1]+ 0.5*(bins0_t_180[1:] - bins0_t_180[:-1]), pdf0_t_180, marker='.', c='g', s=30, alpha=0.8, label = f'Backgrounds $m_s=180$')

    # mhs=200
    pdf0_t_200, bins0_t_200, patches0_t_200 = plt.hist(discriminant0_t_200, bins = binning, color = 'm', alpha = 0.3, histtype = 'stepfilled', linewidth = 2, edgecolor='m', density=True)
    plt.scatter(bins0_t_200[:-1]+ 0.5*(bins0_t_200[1:] - bins0_t_200[:-1]), pdf0_t_200, marker='.', c='m', s=30, alpha=0.8, label = f'Backgrounds $m_s=200$')

    # mhs=300
    pdf0_t_300, bins0_t_300, patches0_t_300 = plt.hist(discriminant0_t_300, bins = binning, color = 'indigo', alpha = 0.3, histtype = 'stepfilled', linewidth = 2, edgecolor='m', density=True)
    plt.scatter(bins0_t_300[:-1]+ 0.5*(bins0_t_300[1:] - bins0_t_300[:-1]), pdf0_t_300, marker='.', c='indigo', s=30, alpha=0.8, label = f'Backgrounds $m_s=300$')


    plt.legend(loc = 'upper center')
    plt.ylabel('Density', fontsize = 12)
    plt.yscale('log')
    plt.xlabel('Random Forest discriminant', fontsize = 12)
    plt.savefig('Discriminant'+ANALYSIS_NAME+'Backgrounds_mhs.png', dpi = 600)

    plt.clf()
    plt.figure(num=None, figsize=(6, 6))
    plt.subplot(111)
    # mZp=200
    pdf0_t_200, bins0_t_200, patches0_t_200 = plt.hist(discriminant0_t_200z, bins = binning, color = 'r', alpha = 0.3, histtype = 'stepfilled', linewidth = 2, edgecolor='r', density=True)
    plt.scatter(bins0_t_200[:-1]+ 0.5*(bins0_t_200[1:] - bins0_t_200[:-1]), pdf0_t_200, marker='.', c='r', s=30, alpha=0.8, label = f'Backgrounds $m_Z=200$')

    # mZp=500
    pdf0_t_500, bins0_t_500, patches0_t_500 = plt.hist(discriminant0_t_500z, bins = binning, color = 'g', alpha = 0.3, histtype = 'stepfilled', linewidth = 2, edgecolor='g', density=True)
    plt.scatter(bins0_t_500[:-1]+ 0.5*(bins0_t_500[1:] - bins0_t_500[:-1]), pdf0_t_500, marker='.', c='g', s=30, alpha=0.8, label = f'Backgrounds $m_Z=500$')

    # mZp=1200
    pdf0_t_1200, bins0_t_1200, patches0_t_1200 = plt.hist(discriminant0_t_1200z, bins = binning, color = 'm', alpha = 0.3, histtype = 'stepfilled', linewidth = 2, edgecolor='m', density=True)
    plt.scatter(bins0_t_1200[:-1]+ 0.5*(bins0_t_1200[1:] - bins0_t_1200[:-1]), pdf0_t_1200, marker='.', c='m', s=30, alpha=0.8, label = f'Backgrounds $m_Z=1200$')

    # mZp=1500
    pdf0_t_1500, bins0_t_1500, patches0_t_1500 = plt.hist(discriminant0_t_1500z, bins = binning, color = 'indigo', alpha = 0.3, histtype = 'stepfilled', linewidth = 2, edgecolor='orange', density=True)
    plt.scatter(bins0_t_1500[:-1]+ 0.5*(bins0_t_1500[1:] - bins0_t_1500[:-1]), pdf0_t_1500, marker='.', c='indigo', s=30, alpha=0.8, label = f'Backgrounds $m_Z=1500$')

    # mZp=2000
    pdf0_t_2000, bins0_t_2000, patches0_t_2000 = plt.hist(discriminant0_t_2000z, bins = binning, color = 'r', alpha = 0.3, histtype = 'stepfilled', linewidth = 2, edgecolor='cyan', density=True)
    plt.scatter(bins0_t_2000[:-1]+ 0.5*(bins0_t_2000[1:] - bins0_t_2000[:-1]), pdf0_t_2000, marker='.', c='r', s=30, alpha=0.8, label = f'Backgrounds $m_Z=2000$')

    # mZp=2500
    pdf0_t_2500, bins0_t_2500, patches0_t_2500 = plt.hist(discriminant0_t_2500z, bins = binning, color = 'indigo', alpha = 0.3, histtype = 'stepfilled', linewidth = 2, edgecolor='m', density=True)
    plt.scatter(bins0_t_2500[:-1]+ 0.5*(bins0_t_2500[1:] - bins0_t_2500[:-1]), pdf0_t_2500, marker='.', c='indigo', s=30, alpha=0.8, label = f'Backgrounds $m_Z=2500$')


    plt.legend(loc = 'upper center')
    plt.ylabel('Density', fontsize = 12)
    plt.yscale('log')
    plt.xlabel('Random Forest discriminant', fontsize = 12)
    plt.savefig('Discriminant'+ANALYSIS_NAME+'Backgrounds_Zp.png', dpi = 600)

    plt.clf()
    plt.figure(num=None, figsize=(6, 6))
    plt.subplot(111)
    # mDM=100
    pdf0_t_100, bins0_t_100, patches0_t_100 = plt.hist(discriminant0_t_100x, bins = binning, color = 'r', alpha = 0.3, histtype = 'stepfilled', linewidth = 2, edgecolor='r', density=True)
    plt.scatter(bins0_t_100[:-1]+ 0.5*(bins0_t_100[1:] - bins0_t_100[:-1]), pdf0_t_100, marker='.', c='r', s=30, alpha=0.8, label = f'Backgrounds $m_\chi=100$')

    # mDM=150
    pdf0_t_150, bins0_t_150, patches0_t_150 = plt.hist(discriminant0_t_150x, bins = binning, color = 'g', alpha = 0.3, histtype = 'stepfilled', linewidth = 2, edgecolor='g', density=True)
    plt.scatter(bins0_t_150[:-1]+ 0.5*(bins0_t_150[1:] - bins0_t_150[:-1]), pdf0_t_150, marker='.', c='g', s=30, alpha=0.8, label = f'Backgrounds $m_\chi=150$')

    # mDM=200
    pdf0_t_200, bins0_t_200, patches0_t_200 = plt.hist(discriminant0_t_200x, bins = binning, color = 'm', alpha = 0.3, histtype = 'stepfilled', linewidth = 2, edgecolor='m', density=True)
    plt.scatter(bins0_t_200[:-1]+ 0.5*(bins0_t_200[1:] - bins0_t_200[:-1]), pdf0_t_200, marker='.', c='m', s=30, alpha=0.8, label = f'Backgrounds $m_\chi=200$')

    # mDM=300
    pdf0_t_300, bins0_t_300, patches0_t_300 = plt.hist(discriminant0_t_300x, bins = binning, color = 'indigo', alpha = 0.3, histtype = 'stepfilled', linewidth = 2, edgecolor='m', density=True)
    plt.scatter(bins0_t_300[:-1]+ 0.5*(bins0_t_300[1:] - bins0_t_300[:-1]), pdf0_t_300, marker='.', c='indigo', s=30, alpha=0.8, label = f'Backgrounds $m_\chi=300$')


    plt.legend(loc = 'upper center')
    plt.ylabel('Density', fontsize = 12)
    plt.yscale('log')
    plt.xlabel('Random Forest discriminant', fontsize = 12)
    plt.savefig('Discriminant'+ANALYSIS_NAME+'Backgrounds_mx.png', dpi = 600)






    plt.clf()
    plt.figure(num=None, figsize=(6, 6))
    plt.subplot(111)

    pdf1_t_160, bins1_t_160, patches1_t_160 = plt.hist(discriminant1_t_160, bins = binning, color = 'b', alpha = 0.3, histtype = 'stepfilled', linewidth = 2, edgecolor='b', density=True)
    plt.scatter(bins1_t_160[:-1]+ 0.5*(bins1_t_160[1:] - bins1_t_160[:-1]), pdf1_t_160, marker='.', c='b', s=30, alpha=0.8, label = f'$m_s=160$')
    pdf1_t_180, bins1_t_180, patches1_t_180 = plt.hist(discriminant1_t_180, bins = binning, color = 'c', alpha = 0.3, histtype = 'stepfilled', linewidth = 2, edgecolor='c', density=True)
    plt.scatter(bins1_t_180[:-1]+ 0.5*(bins1_t_180[1:] - bins1_t_180[:-1]), pdf1_t_180, marker='.', c='c', s=30, alpha=0.8, label = f'$m_s=180$')
    pdf1_t_200, bins1_t_200, patches1_t_200 = plt.hist(discriminant1_t_200, bins = binning, color = 'y', alpha = 0.3, histtype = 'stepfilled', linewidth = 2, edgecolor='y', density=True)
    plt.scatter(bins1_t_200[:-1]+ 0.5*(bins1_t_200[1:] - bins1_t_200[:-1]), pdf1_t_200, marker='.', c='y', s=30, alpha=0.8, label = f'$m_s=200$')
    pdf1_t_300, bins1_t_300, patches1_t_300 = plt.hist(discriminant1_t_300, bins = binning, color = 'lime', alpha = 0.3, histtype = 'stepfilled', linewidth = 2, edgecolor='y', density=True)
    plt.scatter(bins1_t_300[:-1]+ 0.5*(bins1_t_300[1:] - bins1_t_300[:-1]), pdf1_t_300, marker='.', c='lime', s=30, alpha=0.8, label = f'$m_s=300$')

    plt.legend(loc = 'upper center')
    plt.ylabel('Density', fontsize = 12)
    plt.yscale('log')
    plt.xlabel('Random Forest discriminant', fontsize = 12)
    plt.savefig('Discriminant'+ANALYSIS_NAME+'Signals_mhs.png', dpi = 600)

    plt.clf()
    plt.figure(num=None, figsize=(6, 6))
    plt.subplot(111)

    pdf1_t_200, bins1_t_200, patches1_t_200 = plt.hist(discriminant1_t_200z, bins = binning, color = 'b', alpha = 0.3, histtype = 'stepfilled', linewidth = 2, edgecolor='b', density=True)
    plt.scatter(bins1_t_200[:-1]+ 0.5*(bins1_t_200[1:] - bins1_t_200[:-1]), pdf1_t_200, marker='.', c='b', s=30, alpha=0.8, label = f'$m_Z=200$')
    pdf1_t_500, bins1_t_500, patches1_t_500 = plt.hist(discriminant1_t_500z, bins = binning, color = 'c', alpha = 0.3, histtype = 'stepfilled', linewidth = 2, edgecolor='c', density=True)
    plt.scatter(bins1_t_500[:-1]+ 0.5*(bins1_t_500[1:] - bins1_t_500[:-1]), pdf1_t_500, marker='.', c='c', s=30, alpha=0.8, label = f'$m_Z=500$')
    pdf1_t_1200, bins1_t_1200, patches1_t_1200 = plt.hist(discriminant1_t_1200z, bins = binning, color = 'y', alpha = 0.3, histtype = 'stepfilled', linewidth = 2, edgecolor='y', density=True)
    plt.scatter(bins1_t_1200[:-1]+ 0.5*(bins1_t_1200[1:] - bins1_t_1200[:-1]), pdf1_t_1200, marker='.', c='y', s=30, alpha=0.8, label = f'$m_Z=1200$')
    pdf1_t_1500, bins1_t_1500, patches1_t_1500 = plt.hist(discriminant1_t_1500z, bins = binning, color = 'orange', alpha = 0.3, histtype = 'stepfilled', linewidth = 2, edgecolor='y', density=True)
    plt.scatter(bins1_t_1500[:-1]+ 0.5*(bins1_t_1500[1:] - bins1_t_1500[:-1]), pdf1_t_1500, marker='.', c='orange', s=30, alpha=0.8, label = f'$m_Z=1500$')
    pdf1_t_2500, bins1_t_2500, patches1_t_2500 = plt.hist(discriminant1_t_2500z, bins = binning, color = 'lime', alpha = 0.3, histtype = 'stepfilled', linewidth = 2, edgecolor='y', density=True)
    plt.scatter(bins1_t_2500[:-1]+ 0.5*(bins1_t_2500[1:] - bins1_t_2500[:-1]), pdf1_t_2500, marker='.', c='lime', s=30, alpha=0.8, label = f'$m_Z=2500$')

    plt.legend(loc = 'upper center')
    plt.ylabel('Density', fontsize = 12)
    plt.yscale('log')
    plt.xlabel('Random Forest discriminant', fontsize = 12)
    plt.savefig('Discriminant'+ANALYSIS_NAME+'Signals_Zp.png', dpi = 600)

    plt.clf()
    plt.figure(num=None, figsize=(6, 6))
    plt.subplot(111)

    pdf1_t_100, bins1_t_100, patches1_t_100 = plt.hist(discriminant1_t_100x, bins = binning, color = 'b', alpha = 0.3, histtype = 'stepfilled', linewidth = 2, edgecolor='b', density=True)
    plt.scatter(bins1_t_100[:-1]+ 0.5*(bins1_t_100[1:] - bins1_t_100[:-1]), pdf1_t_100, marker='.', c='b', s=30, alpha=0.8, label = f'$m_\chi=100$')
    pdf1_t_150, bins1_t_150, patches1_t_150 = plt.hist(discriminant1_t_150x, bins = binning, color = 'c', alpha = 0.3, histtype = 'stepfilled', linewidth = 2, edgecolor='c', density=True)
    plt.scatter(bins1_t_150[:-1]+ 0.5*(bins1_t_150[1:] - bins1_t_150[:-1]), pdf1_t_150, marker='.', c='c', s=30, alpha=0.8, label = f'$m_\chi=150$')
    pdf1_t_200, bins1_t_200, patches1_t_200 = plt.hist(discriminant1_t_200x, bins = binning, color = 'y', alpha = 0.3, histtype = 'stepfilled', linewidth = 2, edgecolor='y', density=True)
    plt.scatter(bins1_t_200[:-1]+ 0.5*(bins1_t_200[1:] - bins1_t_200[:-1]), pdf1_t_200, marker='.', c='y', s=30, alpha=0.8, label = f'$m_\chi=200$')
    pdf1_t_300, bins1_t_300, patches1_t_300 = plt.hist(discriminant1_t_300x, bins = binning, color = 'lime', alpha = 0.3, histtype = 'stepfilled', linewidth = 2, edgecolor='y', density=True)
    plt.scatter(bins1_t_300[:-1]+ 0.5*(bins1_t_300[1:] - bins1_t_300[:-1]), pdf1_t_300, marker='.', c='lime', s=30, alpha=0.8, label = f'$m_\chi=300$')

    plt.legend(loc = 'upper center')
    plt.ylabel('Density', fontsize = 12)
    plt.yscale('log')
    plt.xlabel('Random Forest discriminant', fontsize = 12)
    plt.savefig('Discriminant'+ANALYSIS_NAME+'Signals_mx.png', dpi = 600)

plt.clf()
plt.figure(num=None, figsize=(6, 6))
plt.subplot(111)
pdf0, bins0, patches0 = plt.hist(discriminant0, bins = binning, color = 'm', alpha = 0.0, histtype = 'stepfilled', linewidth = 1, edgecolor='r', density=True)
pdf1, bins1, patches1 = plt.hist(discriminant1, bins = binning, color = 'y', alpha = 0.0, histtype = 'stepfilled', linewidth = 1, edgecolor='b', density=True)

pdf0_t, bins0_t, patches0_t = plt.hist(discriminant0_t, bins = binning, color = 'r', alpha = 0.3, histtype = 'stepfilled', linewidth = 2, edgecolor='r', label = 'Backgrounds (train)', density=True)
pdf1_t, bins1_t, patches1_t = plt.hist(discriminant1_t, bins = binning, color = 'b', alpha = 0.3, histtype = 'stepfilled', linewidth = 2, edgecolor='b', label = 'Dark Higgs signal (train)', density=True)

plt.scatter(bins0[:-1]+ 0.5*(bins0[1:] - bins0[:-1]), pdf0, marker='.', c='m', s=30, alpha=0.8, label = 'Backgrounds')
plt.scatter(bins1[:-1]+ 0.5*(bins1[1:] - bins1[:-1]), pdf1, marker='.', c='y', s=30, alpha=0.8, label = 'Dark Higgs signal')
plt.legend(loc = 'upper center')
plt.yscale('log')
plt.ylabel('Density', fontsize = 12)
plt.xlabel('Random Forest discriminant (Signal)', fontsize = 12)
plt.savefig('Log_Discriminant_distribution'+ANALYSIS_NAME+'.png', dpi = 600)
print("DONE!")
##### --------------------------------
print("")
print("Plotting ROC-------------------------------------------------------------------------------------------------------------------------")
fpr, tpr, thresholds = metrics.roc_curve(Y_test["isSignal"], y_pred_L)
auc = metrics.auc(fpr, tpr)

plt.clf()
plt.figure(num=None, figsize=(6, 6))
plt.subplot(111)
plt.plot(fpr, tpr, color = 'r', label = "ROC curve")
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label = "Random guess")
plt.legend(loc = "lower right")
plt.xlabel('False Positive rate', fontsize = 12)
plt.ylabel('True Positive rate', fontsize = 12)
plt.text(0.55, 0.3, f'AUC = {auc:.3f}', fontsize=12, color='r')
plt.axvline(x=0, color = 'black', linestyle = '--', linewidth = 0.5)
plt.axhline(y=1, color = 'black', linestyle = '--', linewidth = 0.5)
plt.savefig('ROC'+ANALYSIS_NAME+'.png', dpi = 600)
print("")
print("The AUC of the model is: ", auc)

if parametric:
    print("")
    print("Plotting individual ROCs for the different mass points")
    # Higgs mass ms
    fpr_160, tpr_160, thresholds_160 = metrics.roc_curve(Y_test["isSignal"].iloc[filtered_indices_160_test[0]], [y_pred_L[i] for i in filtered_indices_160_test[0]])
    fpr_180, tpr_180, thresholds_180 = metrics.roc_curve(Y_test["isSignal"].iloc[filtered_indices_180_test[0]], [y_pred_L[i] for i in filtered_indices_180_test[0]])
    fpr_200, tpr_200, thresholds_200 = metrics.roc_curve(Y_test["isSignal"].iloc[filtered_indices_200_test[0]], [y_pred_L[i] for i in filtered_indices_200_test[0]])
    fpr_300, tpr_300, thresholds_300 = metrics.roc_curve(Y_test["isSignal"].iloc[filtered_indices_300_test[0]], [y_pred_L[i] for i in filtered_indices_300_test[0]])
    auc_160 = metrics.auc(fpr_160, tpr_160)
    auc_180 = metrics.auc(fpr_180, tpr_180)
    auc_200 = metrics.auc(fpr_200, tpr_200)
    auc_300 = metrics.auc(fpr_300, tpr_300)

    print('The accuracy for each case is:')
    print('For ms=160', auc_160)
    print('For ms=180', auc_180)
    print('For ms=200', auc_200)
    print('For ms=300', auc_300)

    plt.clf()
    plt.figure(num=None, figsize=(6, 6))
    plt.subplot(111)
    plt.plot(fpr_160, tpr_160, color = 'r', label = f"ROC curve $m_s=160$")
    plt.plot(fpr_180, tpr_180, color = 'b', label = f"ROC curve $m_s=180$")
    plt.plot(fpr_200, tpr_200, color = 'g', label = f"ROC curve $m_s=200$")
    plt.plot(fpr_300, tpr_300, color = 'm', label = f"ROC curve $m_s=300$")
    # Show accuracy on the plot
    plt.text(0.55, 0.3, f'AUC $m_s=160$ = {auc_160:.3f}', fontsize=12, color='r')
    plt.text(0.55, 0.2, f'AUC $m_s=180$ = {auc_180:.3f}', fontsize=12, color='b')
    plt.text(0.55, 0.1, f'AUC $m_s=200$ = {auc_200:.3f}', fontsize=12, color='g')
    plt.text(0.55, 0.0, f'AUC $m_s=300$ = {auc_300:.3f}', fontsize=12, color='m')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label = "Random guess")
    #plt.legend(loc = "lower right")
    plt.xlabel('False Positive rate', fontsize = 12)
    plt.ylabel('True Positive rate', fontsize = 12)
    plt.axvline(x=0, color = 'black', linestyle = '--', linewidth = 0.5)
    plt.axhline(y=1, color = 'black', linestyle = '--', linewidth = 0.5)
    plt.savefig('ROC'+ANALYSIS_NAME+'mhs.png', dpi = 600)

    # Mediator Zp mass
    fpr_200, tpr_200, thresholds_200 = metrics.roc_curve(Y_test["isSignal"].iloc[filtered_indices_200z_test[0]], [y_pred_L[i] for i in filtered_indices_200z_test[0]])
    fpr_500, tpr_500, thresholds_500 = metrics.roc_curve(Y_test["isSignal"].iloc[filtered_indices_500z_test[0]], [y_pred_L[i] for i in filtered_indices_500z_test[0]])
    fpr_1200, tpr_1200, thresholds_1200 = metrics.roc_curve(Y_test["isSignal"].iloc[filtered_indices_1200z_test[0]], [y_pred_L[i] for i in filtered_indices_1200z_test[0]])
    fpr_1500, tpr_1500, thresholds_1500 = metrics.roc_curve(Y_test["isSignal"].iloc[filtered_indices_1500z_test[0]], [y_pred_L[i] for i in filtered_indices_1500z_test[0]])
    fpr_2000, tpr_2000, thresholds_2000 = metrics.roc_curve(Y_test["isSignal"].iloc[filtered_indices_2000z_test[0]], [y_pred_L[i] for i in filtered_indices_2000z_test[0]])
    fpr_2500, tpr_2500, thresholds_2500 = metrics.roc_curve(Y_test["isSignal"].iloc[filtered_indices_2500z_test[0]], [y_pred_L[i] for i in filtered_indices_2500z_test[0]])
    auc_200 = metrics.auc(fpr_200, tpr_200)
    auc_500 = metrics.auc(fpr_500, tpr_500)
    auc_1200 = metrics.auc(fpr_1200, tpr_1200)
    auc_1500 = metrics.auc(fpr_1500, tpr_1500)
    auc_2000 = metrics.auc(fpr_2000, tpr_2000)
    auc_2500 = metrics.auc(fpr_2500, tpr_2500)

    print('The accuracy for each case is:')
    print('For mZ=200', auc_200)
    print('For mZ=500', auc_500)
    print('For mZ=1200', auc_1200)
    print('For mZ=1500', auc_1500)
    print('For mZ=2000', auc_2000)
    print('For mZ=2500', auc_2500)

    plt.clf()
    plt.figure(num=None, figsize=(6, 6))
    plt.subplot(111)
    plt.plot(fpr_200, tpr_200, color = 'r', label = f"ROC curve $m_Z=200$")
    plt.plot(fpr_500, tpr_500, color = 'b', label = f"ROC curve $m_Z=500$")
    plt.plot(fpr_1200, tpr_1200, color = 'g', label = f"ROC curve $m_Z=1200$")
    plt.plot(fpr_1500, tpr_1500, color = 'orange', label = f"ROC curve $m_Z=1500$")
    plt.plot(fpr_2000, tpr_2000, color = 'c', label = f"ROC curve $m_Z=2000$")
    plt.plot(fpr_2500, tpr_2500, color = 'm', label = f"ROC curve $m_Z=2500$")
    # Show AUC on the plot
    plt.text(0.5, 0.5, f'AUC $m_Z=200$ = {auc_200:.3f}', fontsize=12, color='r')
    plt.text(0.5, 0.4, f'AUC $m_Z=500$ = {auc_500:.3f}', fontsize=12, color='b')
    plt.text(0.5, 0.3, f'AUC $m_Z=1200$ = {auc_1200:.3f}', fontsize=12, color='g')
    plt.text(0.5, 0.2, f'AUC $m_Z=1500$ = {auc_1500:.3f}', fontsize=12, color='orange')
    plt.text(0.5, 0.1, f'AUC $m_Z=2000$ = {auc_2000:.3f}', fontsize=12, color='c')
    plt.text(0.5, 0.0, f'AUC $m_Z=2500$ = {auc_2500:.3f}', fontsize=12, color='m')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label = "Random guess")
    #plt.legend(loc = "lower right")
    plt.xlabel('False Positive rate', fontsize = 12)
    plt.ylabel('True Positive rate', fontsize = 12)
    plt.axvline(x=0, color = 'black', linestyle = '--', linewidth = 0.5)
    plt.axhline(y=1, color = 'black', linestyle = '--', linewidth = 0.5)
    plt.savefig('ROC'+ANALYSIS_NAME+'mZp.png', dpi = 600)


    # Dark matter mass
    fpr_100, tpr_100, thresholds_100 = metrics.roc_curve(Y_test["isSignal"].iloc[filtered_indices_100x_test[0]], [y_pred_L[i] for i in filtered_indices_100x_test[0]])
    fpr_150, tpr_150, thresholds_150 = metrics.roc_curve(Y_test["isSignal"].iloc[filtered_indices_150x_test[0]], [y_pred_L[i] for i in filtered_indices_150x_test[0]])
    fpr_200, tpr_200, thresholds_200 = metrics.roc_curve(Y_test["isSignal"].iloc[filtered_indices_200x_test[0]], [y_pred_L[i] for i in filtered_indices_200x_test[0]])
    fpr_300, tpr_300, thresholds_300 = metrics.roc_curve(Y_test["isSignal"].iloc[filtered_indices_300x_test[0]], [y_pred_L[i] for i in filtered_indices_300x_test[0]])
    auc_100 = metrics.auc(fpr_100, tpr_100)
    auc_150 = metrics.auc(fpr_150, tpr_150)
    auc_200 = metrics.auc(fpr_200, tpr_200)
    auc_300 = metrics.auc(fpr_300, tpr_300)

    print('The accuracy for each case is:')
    print('For mx=100', auc_100)
    print('For mx=150', auc_150)
    print('For mx=200', auc_200)
    print('For mx=300', auc_300)

    plt.clf()
    plt.figure(num=None, figsize=(6, 6))
    plt.subplot(111)
    plt.plot(fpr_100, tpr_100, color = 'r', label = f"ROC curve $m_x=100$")
    plt.plot(fpr_150, tpr_150, color = 'b', label = f"ROC curve $m_x=150$")
    plt.plot(fpr_200, tpr_200, color = 'g', label = f"ROC curve $m_x=200$")
    plt.plot(fpr_300, tpr_300, color = 'm', label = f"ROC curve $m_x=300$")
    # Show AUC on the plot
    plt.text(0.55, 0.3, f'AUC $m_x=100$ = {auc_100:.3f}', fontsize=12, color='r')
    plt.text(0.55, 0.2, f'AUC $m_x=150$ = {auc_150:.3f}', fontsize=12, color='b')
    plt.text(0.55, 0.1, f'AUC $m_x=200$ = {auc_200:.3f}', fontsize=12, color='g')
    plt.text(0.55, 0.0, f'AUC $m_x=300$ = {auc_300:.3f}', fontsize=12, color='m')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label = "Random guess")
    #plt.legend(loc = "lower right")
    plt.xlabel('False Positive rate', fontsize = 12)
    plt.ylabel('True Positive rate', fontsize = 12)
    plt.axvline(x=0, color = 'black', linestyle = '--', linewidth = 0.5)
    plt.axhline(y=1, color = 'black', linestyle = '--', linewidth = 0.5)
    plt.savefig('ROC'+ANALYSIS_NAME+'mx.png', dpi = 600)

print("DONE!")
##### Feature importance
print("")
print("Plotting feature importance-----------------------------------------------------------------------------------------------------------")

if choose_RFC:
    
    ######################### FOR THE RANDOM FOREST
    plt.clf()
    plt.figure(num=None, figsize=(8, 6))  # Adjusted figure size
    plt.subplot(111)
    plt.barh(range(len(rfc.feature_importances_)), rfc.feature_importances_)  # Use plt.barh() for horizontal bars
    plt.ylabel('Input Features', fontsize=14)  # Increased font size
    plt.xlabel('Variable Importance', fontsize=14)  # Increased font size
    plt.title('Feature Importance', fontsize=16)  # Increased font size for title
    plt.savefig('Feature_importance' + ANALYSIS_NAME + '.png', dpi=600)

    tickets = var
    plt.clf()
    fig, ax = plt.subplots(figsize=(12, 8))  # Adjusted figure size
    plt.barh(range(len(rfc.feature_importances_)), rfc.feature_importances_)  # Use plt.barh() for horizontal bars
    ax.set_yticks(np.arange(len(tickets)))  # Set ticks on Y-axis
    ax.set_yticklabels(tickets, fontsize=12)  # Increased font size for y-axis labels
    plt.xlabel('Feature Importance', fontsize=14)  # Increased font size for x-axis label
    plt.title('Log Feature Importance', fontsize=16)  # Increased font size for title
    fig.tight_layout(pad=3.0)  # Increased padding for better layout
    plt.xscale('log')  # If you want to use log scale for X-axis
    plt.savefig('Log_Feature_importance' + ANALYSIS_NAME + '.png', dpi=1200)
    """ 
    plt.clf()
    import shap
    tickets = var
    X_training = np.array(X_train)
    # Calculate SHAP values for train data
    explainer_train = shap.Explainer(rfc, X_training)
    shap_values_train = explainer_train.shap_values(X_training)

    # Summarize the SHAP values for train data
    shap.summary_plot(shap_values_train, X_training, feature_names=tickets, show = False)
    plt.title('Feature Importance - SHAP Values', fontsize=16)
    plt.xlabel('SHAP Value', fontsize=14)
    plt.ylabel('Input Features', fontsize=14)
    plt.tight_layout()
    plt.savefig('SHAP_Feature_importance_Train.png', dpi=600)

    # Create a horizontal bar plot for log feature importance for train data
    plt.clf()
    shap.summary_plot(shap_values_train, X_training, plot_type="bar", feature_names=tickets, show = False)
    plt.xlabel('Log Feature Importance', fontsize=14)  # Increased font size for x-axis label
    plt.title('Log Feature Importance - SHAP Values (Train Data)', fontsize=16)  # Increased font size for title
    plt.xscale('log')  # If you want to use log scale for X-axis
    plt.tight_layout()
    plt.savefig('Log_SHAP_Feature_importance_Train_bar.png', dpi=1200)

    # Summarize the SHAP values for train data, but in another way
    plt.clf()
    shap.summary_plot(shap_values_train, X_training, feature_names=tickets, plot_type="bar", show = False)
    plt.title('Feature Importance - SHAP Values (Train Data)', fontsize=16)
    plt.xlabel('SHAP Value', fontsize=14)
    plt.ylabel('Input Features', fontsize=14)
    plt.tight_layout()
    plt.savefig('SHAP_Feature_importance_Train_bar.png', dpi=600)
    """
else:   
    """
    plt.clf()
    import shap

    # Assuming you have a list of feature names named 'var'
    tickets = var

    X_training = np.array(X_train)
    # Calculate SHAP values for train data
    explainer_train = shap.Explainer(model, X_training)
    shap_values_train = explainer_train.shap_values(X_training)

    # Summarize the SHAP values for train data
    shap.summary_plot(shap_values_train, X_training, feature_names=tickets, show = False)
    plt.title('Feature Importance - SHAP Values', fontsize=16)
    plt.xlabel('SHAP Value', fontsize=14)
    plt.ylabel('Input Features', fontsize=14)
    plt.tight_layout()
    plt.savefig('SHAP_Feature_importance_Train.png', dpi=600)

    # Create a horizontal bar plot for log feature importance for train data
    plt.clf()
    shap.summary_plot(shap_values_train, X_training, plot_type="bar", feature_names=tickets, show = False)
    plt.xlabel('Log Feature Importance', fontsize=14)  # Increased font size for x-axis label
    plt.title('Log Feature Importance - SHAP Values (Train Data)', fontsize=16)  # Increased font size for title
    plt.xscale('log')  # If you want to use log scale for X-axis
    plt.tight_layout()
    plt.savefig('Log_SHAP_Feature_importance_Train_bar.png', dpi=1200)

    # Summarize the SHAP values for train data, but in another way
    plt.clf()
    shap.summary_plot(shap_values_train, X_training, feature_names=tickets, plot_type="bar", show = False)
    plt.title('Feature Importance - SHAP Values (Train Data)', fontsize=16)
    plt.xlabel('SHAP Value', fontsize=14)
    plt.ylabel('Input Features', fontsize=14)
    plt.tight_layout()
    plt.savefig('SHAP_Feature_importance_Train_bar.png', dpi=600)
    ########################
    """
print("DONE!")

print("")
print("Finding and ploting correlation matrix-----------------------------------------------------------------------------------------------")
import matplotlib.cm as cm
m = np.corrcoef(X_train, rowvar=False) # Correlation matrix with numpy

# Name of variable in order
tickets = var
## PLOTING 
fig, ax = plt.subplots(figsize=(20, 20))
im = ax.matshow(abs(m))
plt.colorbar(im)
for (i, j), z in np.ndenumerate(m):
    ax.text(j, i, '{:0.1f}'.format(abs(z)), ha='center', va='center')

ax.set_xticks(np.arange(len(tickets)))
ax.set_yticks(np.arange(len(tickets)))    
ax.set_xticklabels(tickets)
ax.set_yticklabels(tickets)
plt.setp(ax.get_xticklabels(), rotation=-45, ha="right", rotation_mode="anchor")    
ax.set_title("Correlation matrix")
fig.tight_layout()
plt.savefig('CorrelationMatrix'+ANALYSIS_NAME+'.png', dpi = 600)

print("DONE!")

print("")
print("Finding and ploting correlation matrix for the signal events-------------------------------------------------------------------------")
import matplotlib.cm as cm
m = np.corrcoef(Sig[var], rowvar=False) # Correlation matrix with numpy

# Name of variable in order
tickets = var
## PLOTING 
fig, ax = plt.subplots(figsize=(20, 20))
im = ax.matshow(abs(m))
plt.colorbar(im)
for (i, j), z in np.ndenumerate(m):
    ax.text(j, i, '{:0.1f}'.format(abs(z)), ha='center', va='center')

ax.set_xticks(np.arange(len(tickets)))
ax.set_yticks(np.arange(len(tickets)))
ax.set_xticklabels(tickets)
ax.set_yticklabels(tickets)
plt.setp(ax.get_xticklabels(), rotation=-45, ha="right", rotation_mode="anchor")
ax.set_title("Correlation matrix")
fig.tight_layout()
plt.savefig('CorrelationMatrix'+ANALYSIS_NAME+'Sig.png', dpi = 600)

print("DONE!")

print("")
print("Finding and ploting correlation matrix for the background events---------------------------------------------------------------------")
import matplotlib.cm as cm
m = np.corrcoef(Bkg[var], rowvar=False) # Correlation matrix with numpy

# Name of variable in order
tickets = var
## PLOTING
fig, ax = plt.subplots(figsize=(20, 20))
im = ax.matshow(abs(m))
plt.colorbar(im)
for (i, j), z in np.ndenumerate(m):
    ax.text(j, i, '{:0.1f}'.format(abs(z)), ha='center', va='center')

ax.set_xticks(np.arange(len(tickets)))
ax.set_yticks(np.arange(len(tickets)))
ax.set_xticklabels(tickets)
ax.set_yticklabels(tickets)
plt.setp(ax.get_xticklabels(), rotation=-45, ha="right", rotation_mode="anchor")
ax.set_title("Correlation matrix")
fig.tight_layout()
plt.savefig('CorrelationMatrix'+ANALYSIS_NAME+'Bkg.png', dpi = 600)

print("DONE!")
print("")

covariance = np.cov(Bkg['mpmet'].values, Bkg['ptll'].values)
print("COVARIANCE MATRIX - ptll - mpmet")
print(covariance / (np.std(Bkg['mpmet'])*np.std(Bkg['ptll'])))

print("")
print("END OF THE ANALYSIS------------------------------------------------------------------------------------------------------------------")
