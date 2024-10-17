import os
import copy
import inspect

configurations = os.path.realpath(inspect.getfile(inspect.currentframe())) # this file

aliases = {}
aliases = OrderedDict()


mc = [skey for skey in samples if skey not in ('Fake', 'DATA')]



eleWP='mvaFall17V1Iso_WP90'
muWP= 'cut_Tight_HWWW'

aliases['LepWPCut'] = {
    'expr': 'LepCut2l__ele_'+eleWP+'__mu_'+muWP,
    'samples': mc + ['DATA']
}

aliases['LepWPSF'] = {
    'expr': 'LepSF2l__ele_'+eleWP+'__mu_'+muWP,
    'samples': mc
}


aliases['gstarLow'] = {
    'expr': 'Gen_ZGstar_mass >0 && Gen_ZGstar_mass < 4',
    'samples': 'VgS'
}

aliases['gstarHigh'] = {
    'expr': 'Gen_ZGstar_mass <0 || Gen_ZGstar_mass > 4',
    'samples': 'VgS'
}

aliases['zeroJet'] = {
    'expr': 'Alt(CleanJet_pt,0, 0) < 30.'
}

aliases['oneJet'] = {
    'expr': 'Alt(CleanJet_pt, 0, 0) > 30.'
}

aliases['multiJet'] = {
    'expr': 'Alt(CleanJet_pt, 1, 0) > 30.'
}

aliases['njets'] = {
    'expr': 'Sum(Alt(CleanJet_pt, 1, 0) > 30. && abs(CleanJet_eta) < 2.5)'
}

#aliases['bVeto'] = {
#    'expr': 'Sum(CleanJet_pt > 20. && abs(CleanJet_eta) < 2.5 && Jet_btagDeepB[CleanJet_jetIdx] > 0.1241) == 0'
#}

#aliases['bReq'] = {
#    'expr': 'Sum(CleanJet_pt > 30. && abs(CleanJet_eta) < 2.5 && Jet_btagDeepB[CleanJet_jetIdx] > 0.1241) >= 1'
#}

#aliases['topcr'] = {
#    'expr': 'mth>50 && mll<80 && drll<2.5 && ((zeroJet && !bVeto) || bReq)'
#}







# DeepB = DeepCSV
bWP_loose_deepB  = '0.1208'
bWP_medium_deepB = '0.4168'
bWP_tight_deepB  = '0.7665'

# DeepFlavB = DeepJet
bWP_loose_deepFlavB  = '0.0490'
bWP_medium_deepFlavB = '0.2783'
bWP_tight_deepFlavB  = '0.7100'
"""
# Actual algo and WP definition. BE CONSISTENT!!
bAlgo = 'DeepFlavB' # ['DeepB','DeepFlavB']
bWP   = bWP_loose_deepFlavB
bSF   = 'deepjet' # ['deepcsv','deepjet']  ## deepflav is new b-tag SF
"""
# Actual algo and WP definition. BE CONSISTENT!!
bAlgo = 'DeepB' # ['DeepB','DeepFlavB']
bWP   = bWP_loose_deepB
bSF   = 'deepcsv' # ['deepcsv','deepjet']  ## deepflav is new b-tag SF

aliases['btag_ID'] = {
        'expr': 'Jet_btag{}'.format(bAlgo)
}

aliases['first_btag_ID'] = {
    'expr': 'Alt(Jet_btag{}, Alt(CleanJet_jetIdx, 0, -1), -1)'.format(bAlgo)
}

aliases['second_btag_ID'] = {
    'expr': 'Alt(Jet_btag{}, Alt(CleanJet_jetIdx, 1, -1), -1)'.format(bAlgo)
}



aliases['nbjets'] = {
    'expr': 'Sum(CleanJet_pt > 30. && abs(CleanJet_eta) < 2.5 && Take(Jet_btag{}, CleanJet_jetIdx) > {})'.format(bAlgo, bWP)
}

aliases['bVeto'] = {
    'expr': 'Sum(CleanJet_pt > 20. && abs(CleanJet_eta) < 2.5 && Take(Jet_btag{}, CleanJet_jetIdx) > {}) == 0'.format(bAlgo, bWP)
}

# At least one b-tagged jet  
aliases['bReq'] = { 
    'expr': 'Sum(CleanJet_pt > 30. && abs(CleanJet_eta) < 2.5 && Take(Jet_btag{}, CleanJet_jetIdx) > {}) >= 1'.format(bAlgo, bWP)
}

aliases['topcr'] = {
    'expr': 'mth>40 && PuppiMET_pt>20 && mll > 12 && ((zeroJet && !bVeto) || bReq)'
}

aliases['bVetoSF'] = {
    'expr': 'TMath::Exp(Sum(LogVec((CleanJet_pt>20 && abs(CleanJet_eta)<2.5)*Take(Jet_btagSF_{}_shape, CleanJet_jetIdx)+1*(CleanJet_pt<20 || abs(CleanJet_eta)>2.5))))'.format(bSF),
    'samples': mc
}

aliases['bReqSF'] = {
    'expr': 'TMath::Exp(Sum(LogVec((CleanJet_pt>30 && abs(CleanJet_eta)<2.5)*Take(Jet_btagSF_{}_shape, CleanJet_jetIdx)+1*(CleanJet_pt<30 || abs(CleanJet_eta)>2.5))))'.format(bSF),
    'samples': mc
}

aliases['btagSF'] = {
    'expr': '(bVeto || (topcr && zeroJet))*bVetoSF + (topcr && !zeroJet)*bReqSF',
    'samples': mc
}

aliases['JetPUID_SF'] = {
  'expr' : 'TMath::Exp(Sum((Jet_jetId>=2)*LogVec(Jet_PUIDSF_loose)))',
  'samples': mc
}

#aliases['JetPUID_SF'] = {
#    'expr': '( 1 * !(topcr) + (topcr)*Jet_PUIDSF_loose)',
#    'samples': mc
#}


for shift in ['jes', 'lf', 'hf', 'lfstats1', 'lfstats2', 'hfstats1', 'hfstats2', 'cferr1', 'cferr2']:
    for targ in ['bVeto', 'bReq']:
        alias = aliases['%sSF%sup' % (targ, shift)] = copy.deepcopy(aliases['%sSF' % targ])
        alias['expr'] = alias['expr'].replace('btagSF_deepcsv_shape', 'btagSF_deepcsv_shape_up_%s' % shift)

        alias = aliases['%sSF%sdown' % (targ, shift)] = copy.deepcopy(aliases['%sSF' % targ])
        alias['expr'] = alias['expr'].replace('btagSF_deepcsv_shape', 'btagSF_deepcsv_shape_down_%s' % shift)

    aliases['btagSF%sup' % shift] = {
        'expr': aliases['btagSF']['expr'].replace('SF', 'SF' + shift + 'up'),
        'samples': mc
    }

    aliases['btagSF%sdown' % shift] = {
        'expr': aliases['btagSF']['expr'].replace('SF', 'SF' + shift + 'down'),
        'samples': mc
    }


# Fake leptons transfer factor
aliases['fakeW'] = {
    'expr': 'fakeW2l_ele_'+eleWP+'_mu_'+muWP,
    'samples': ['Fake']
}
# And variations - already divided by central values in formulas !
aliases['fakeWEleUp'] = {
    'expr': 'fakeW2l_ele_'+eleWP+'_mu_'+muWP+'_EleUp',
    'samples': ['Fake']
}
aliases['fakeWEleDown'] = {
    'expr': 'fakeW2l_ele_'+eleWP+'_mu_'+muWP+'_EleDown',
    'samples': ['Fake']
}
aliases['fakeWMuUp'] = {
    'expr': 'fakeW2l_ele_'+eleWP+'_mu_'+muWP+'_MuUp',
    'samples': ['Fake']
}
aliases['fakeWMuDown'] = {
    'expr': 'fakeW2l_ele_'+eleWP+'_mu_'+muWP+'_MuDown',
    'samples': ['Fake']
}
aliases['fakeWStatEleUp'] = {
    'expr': 'fakeW2l_ele_'+eleWP+'_mu_'+muWP+'_statEleUp',
    'samples': ['Fake']
}
aliases['fakeWStatEleDown'] = {
    'expr': 'fakeW2l_ele_'+eleWP+'_mu_'+muWP+'_statEleDown',
    'samples': ['Fake']
}
aliases['fakeWStatMuUp'] = {
    'expr': 'fakeW2l_ele_'+eleWP+'_mu_'+muWP+'_statMuUp',
    'samples': ['Fake']
}
aliases['fakeWStatMuDown'] = {
    'expr': 'fakeW2l_ele_'+eleWP+'_mu_'+muWP+'_statMuDown',
    'samples': ['Fake']
}

# gen-matching to prompt only (GenLepMatch2l matches to *any* gen lepton)
aliases['PromptGenLepMatch2l'] = {
    'expr': 'Alt(Lepton_promptgenmatched,0,0)*Alt(Lepton_promptgenmatched,1,0)',
    'samples': mc
}

aliases['Top_pTrw'] = {
    'expr': '(topGenPt * antitopGenPt > 0.) * (TMath::Sqrt((0.103*TMath::Exp(-0.0118*topGenPt) - 0.000134*topGenPt + 0.973) * (0.103*TMath::Exp(-0.0118*antitopGenPt) - 0.000134*antitopGenPt + 0.973))) + (topGenPt * antitopGenPt <= 0.)',
    'samples': ['top']
}


# ##### DY Z pT reweighting
aliases['nCleanGenJet'] = {
    'linesToAdd': ['.L /afs/cern.ch/user/v/victorr/private/DarkHiggs/Full2018_v7/ngenjet.cc+'],
    'class': 'CountGenJet',
    'args': 'nLeptonGen, LeptonGen_isPrompt, LeptonGen_pdgId, LeptonGen_pt, LeptonGen_eta, LeptonGen_phi, LeptonGen_mass, nPhotonGen, PhotonGen_pt, PhotonGen_eta,PhotonGen_phi, PhotonGen_mass, nGenJet, GenJet_pt, GenJet_eta, GenJet_phi',
    'samples': mc
}


aliases['getGenZpt_OTF'] = {
    #'linesToAdd': ['/afs/cern.ch/work/s/sblancof/private/Run2Analysis/mkShapesRDF/examples/Full2017_v9/getGenZpt.cc'],
    'linesToAdd': ['.L /afs/cern.ch/user/v/victorr/private/DarkHiggs/Full2018_v7/getGenZpt.cc+'],
    'class': 'GetGenZpt',
    'args': 'nGenPart, GenPart_pt, GenPart_pdgId, GenPart_genPartIdxMother, GenPart_statusFlags, gen_ptll',
    'samples': ['DY']
}

exec(open('DYrew30.py', "r").read())
#handle = open('./DYrew30.py','r')
#eval(handle)
#handle.close()

aliases['DY_NLO_pTllrw'] = {
    'expr': '('+DYrew['2018']['NLO'].replace('x', 'getGenZpt_OTF')+')*(nCleanGenJet == 0)+1.0*(nCleanGenJet > 0)',
    'samples': ['DY']
}
aliases['DY_LO_pTllrw'] = {
    'expr': '('+DYrew['2018']['LO'].replace('x', 'getGenZpt_OTF')+')*(nCleanGenJet == 0)+1.0*(nCleanGenJet > 0)',
    'samples': ['DY']
}
# Jet bins
# using Alt$(CleanJet_pt[n], 0) instead of Sum$(CleanJet_pt >= 30) because jet pt ordering is not strictly followed in JES-varied samples


# ### Dark Higgs search variables

aliases['dphill_DH'] = {
        'linesToAdd': [".L /afs/cern.ch/user/v/victorr/private/DarkHiggs/Full2018_v7/newvar.cc+"],
        'class' : 'dphill_DH',
        'args': 'Lepton_pt,Lepton_eta,Lepton_phi,PuppiMET_pt,PuppiMET_phi',
}

aliases['dPhillStar']={
    'expr': 'dphill_DH[0]',
}
aliases['dPhill_Zp']={
    'expr': 'dphill_DH[1]',
}
aliases['Cos_theta']={
    'expr': 'dphill_DH[2]',
}
aliases['Theta_ll']={
    'expr': 'dphill_DH[3]',
}
aliases['dPhill_MET']={
    'expr': 'dphill_DH[4]',
}


# DISCRMINANT EVALUATION FOR DNN AND RANDOM FOREST
aliases['lep_pt1']={
    'expr': 'Lepton_pt[0]',
}
aliases['lep_pt2']={
    'expr': 'Lepton_pt[1]',
}
aliases['lep_eta1']={
    'expr': 'Lepton_eta[0]',
}
aliases['lep_eta2']={
    'expr': 'Lepton_eta[1]',
}




# data/MC scale factors
aliases['SFweight'] = {
    'expr': ' * '.join(['SFweight2l','LepWPCut','LepWPSF','btagSF','JetPUID_SF']),
    'samples': mc
}


# variations
aliases['SFweightEleUp'] = {
    'expr': 'LepSF2l__ele_'+eleWP+'__Up',
    'samples': mc
}
aliases['SFweightEleDown'] = {
    'expr': 'LepSF2l__ele_'+eleWP+'__Do',
    'samples': mc
}
aliases['SFweightMuUp'] = {
    'expr': 'LepSF2l__mu_'+muWP+'__Up',
    'samples': mc
}
aliases['SFweightMuDown'] = {
    'expr': 'LepSF2l__mu_'+muWP+'__Do',
    'samples': mc
}
