# nuisances

nuisances = {}

# name of samples here must match keys in samples.py 

# imported from samples.py:
# samples, treeBaseDir, mcProduction, mcSteps
# imported from cuts.py
# cuts

mcProduction = 'Autumn18_102X_nAODv7_Full2018v7'
dataReco = 'Run2018_102X_nAODv7_Full2018v7'
fakeReco = 'Run2018_102X_nAODv7_Full2018v7'
mcSteps = 'MCl1loose2018v7__MCCorr2018v7__l2loose__l2tightOR2018v7{var}'
fakeSteps = 'DATAl1loose2018v7__l2loose__fakeW'
dataSteps = 'DATAl1loose2018v7__l2loose__l2tightOR2018v7'
# signalDirectory = '/eos/user/r/rocio/MonoH/Autumn18_102X_nAODv7_Full2018v7/MCl1loose2018v7__MCCorr2018v7__l2loose__l2tightOR2018v7__{}'

treeBaseDir = '/eos/cms/store/group/phys_higgs/cmshww/amassiro/HWWNano'
limitFiles = -1


redirector = ""

def makeMCDirectory(var=''):
    if var:
        return os.path.join(treeBaseDir, mcProduction, mcSteps.format(var='__'+var))
    else:
        return os.path.join(treeBaseDir, mcProduction, mcSteps.format(var=""))

mcDirectory = makeMCDirectory()
fakeDirectory = os.path.join(treeBaseDir, dataReco, fakeSteps)
dataDirectory = os.path.join(treeBaseDir, dataReco, dataSteps)

def nanoGetSampleFiles(path, name):
    _files = searchFiles.searchFiles(path, name, redirector=redirector)
    if limitFiles != -1 and len(_files) > limitFiles:
        return [(name, _files[:limitFiles])]
    else:
                return [(name, _files)]

try:
    mc = [skey for skey in samples if skey != 'DATA' and not skey.startswith('Fake')]
except NameError:
    mc = []
    cuts = {}
    nuisances = {}
    def makeMCDirectory(x=''):
        return ''

################################ EXPERIMENTAL UNCERTAINTIES  #################################

#### Luminosity

#nuisances['lumi'] = {
#    'name': 'lumi_13TeV_2018',
#    'type': 'lnN',
#    'samples': dict((skey, '1.025') for skey in mc if skey not in ['WW', 'top', 'DY'])
#}

nuisances['lumi_Uncorrelated'] = {
    'name': 'lumi_13TeV_2018',
    'type': 'lnN',
    'samples': dict((skey, '1.015') for skey in mc if skey not in ['WW', 'top','DY'])
}

nuisances['lumi_XYFact'] = {
    'name': 'lumi_13TeV_XYFact',
    'type': 'lnN',
    'samples': dict((skey, '1.02') for skey in mc if skey not in ['WW', 'top','DY'])
}

nuisances['lumi_LScale'] = {
    'name': 'lumi_13TeV_LSCale',
    'type': 'lnN',
    'samples': dict((skey, '1.002') for skey in mc if skey not in ['WW', 'top','DY'])
}

nuisances['lumi_CurrCalib'] = {
    'name': 'lumi_13TeV_CurrCalib',
    'type': 'lnN',
    'samples': dict((skey, '1.002') for skey in mc if skey not in ['WW', 'top','DY'])
}

#### FAKES

nuisances['fake_syst'] = {
    'name': 'CMS_fake_syst',
    'type': 'lnN',
    'samples': {
        'Fake': '1.3'
    },
}

nuisances['fake_ele'] = {
    'name': 'CMS_fake_e_2018',
    'kind': 'weight',
    'type': 'shape',
    'samples': {
        'Fake': ['fakeWEleUp', 'fakeWEleDown'],
    }
}

nuisances['fake_ele_stat'] = {
    'name': 'CMS_fake_stat_e_2018',
    'kind': 'weight',
    'type': 'shape',
    'samples': {
        'Fake': ['fakeWStatEleUp', 'fakeWStatEleDown']
    }
}

nuisances['fake_mu'] = {
    'name': 'CMS_fake_m_2018',
    'kind': 'weight',
    'type': 'shape',
    'samples': {
        'Fake': ['fakeWMuUp', 'fakeWMuDown'],
    }
}

nuisances['fake_mu_stat'] = {
    'name': 'CMS_fake_stat_m_2018',
    'kind': 'weight',
    'type': 'shape',
    'samples': {
        'Fake': ['fakeWStatMuUp', 'fakeWStatMuDown'],
    }
}

##### B-tagger

for shift in ['jes', 'lf', 'hf', 'hfstats1', 'hfstats2', 'lfstats1', 'lfstats2', 'cferr1', 'cferr2']:
    btag_syst = ['(btagSF%sup)/(btagSF)' % shift, '(btagSF%sdown)/(btagSF)' % shift]

    name = 'CMS_btag_%s' % shift
    if 'stats' in shift:
        name += '_2018'

    nuisances['btag_shape_%s' % shift] = {
        'name': name,
        'kind': 'weight',
        'type': 'shape',
        'samples': dict((skey, btag_syst) for skey in mc),
    }

##### Trigger Efficiency

trig_syst = ['((TriggerEffWeight_2l_u)/(TriggerEffWeight_2l))*(TriggerEffWeight_2l>0.02) + (TriggerEffWeight_2l<=0.02)', '(TriggerEffWeight_2l_d)/(TriggerEffWeight_2l)']

nuisances['trigg'] = {
    'name': 'CMS_eff_hwwtrigger_2018',
    'kind': 'weight',
    'type': 'shape',
    'samples': dict((skey, trig_syst) for skey in mc)
}

##### Electron Efficiency and energy scale

nuisances['eff_e'] = {
    'name': 'CMS_eff_e_2018',
    'kind': 'weight',
    'type': 'shape',
    'samples': dict((skey, ['SFweightEleUp', 'SFweightEleDown']) for skey in mc)
}

nuisances['electronpt'] = {
    'name': 'CMS_scale_e_2018',
    'kind': 'suffix',
    'type': 'shape',
    'mapUp': 'ElepTup',
    'mapDown': 'ElepTdo',
    'samples': dict((skey, ['1', '1']) for skey in mc),
    'folderUp': makeMCDirectory('ElepTup_suffix'),
    'folderDown': makeMCDirectory('ElepTdo_suffix'),
#    'AsLnN': '1'
}

##### Muon Efficiency and energy scale

nuisances['eff_m'] = {
    'name': 'CMS_eff_m_2018',
    'kind': 'weight',
    'type': 'shape',
    'samples': dict((skey, ['SFweightMuUp', 'SFweightMuDown']) for skey in mc)
}

nuisances['muonpt'] = {
    'name': 'CMS_scale_m_2018',
    'kind': 'suffix',
    'type': 'shape',
    'mapUp': 'MupTup',
    'mapDown': 'MupTdo',
    'samples': dict((skey, ['1', '1']) for skey in mc),
    'folderUp': makeMCDirectory('MupTup_suffix'),
    'folderDown': makeMCDirectory('MupTdo_suffix'),
#    'AsLnN': '1'
}


##### Jet energy scale
jes_systs = ['JESAbsolute','JESAbsolute_2018','JESBBEC1','JESBBEC1_2018','JESEC2','JESEC2_2018','JESFlavorQCD','JESHF','JESHF_2018','JESRelativeBal','JESRelativeSample_2018']
folderup = ""
folderdo = ""

for js in jes_systs:
  if 'Absolute' in js: 
    folderup = makeMCDirectory('JESAbsoluteup_suffix')
    folderdo = makeMCDirectory('JESAbsolutedo_suffix')
  elif 'BBEC1' in js:
    folderup = makeMCDirectory('JESBBEC1up_suffix')
    folderdo = makeMCDirectory('JESBBEC1do_suffix')
  elif 'EC2' in js:
    folderup = makeMCDirectory('JESEC2up_suffix')
    folderdo = makeMCDirectory('JESEC2do_suffix')
  elif 'HF' in js:
    folderup = makeMCDirectory('JESHFup_suffix')
    folderdo = makeMCDirectory('JESHFdo_suffix')
  elif 'Relative' in js:
    folderup = makeMCDirectory('JESRelativeup_suffix')
    folderdo = makeMCDirectory('JESRelativedo_suffix')
  elif 'FlavorQCD' in js:
    folderup = makeMCDirectory('JESFlavorQCDup_suffix')
    folderdo = makeMCDirectory('JESFlavorQCDdo_suffix')

  nuisances[js] = {
      'name': 'CMS_scale_'+js,
      'kind': 'suffix',
      'type': 'shape',
      'mapUp': js+'up',
      'mapDown': js+'do',
      'samples': dict((skey, ['1', '1']) for skey in mc),
      'folderUp': folderup,
      'folderDown': folderdo,
#      'AsLnN': '1'
  }


##### Jet energy resolution
nuisances['JER'] = {
    'name': 'CMS_res_j_2018',
    'kind': 'suffix',
    'type': 'shape',
    'mapUp': 'JERup',
    'mapDown': 'JERdo',
    'samples': dict((skey, ['1', '1']) for skey in mc),
    'folderUp': makeMCDirectory('JERup_suffix'),
    'folderDown': makeMCDirectory('JERdo_suffix'),
#    'AsLnN': '1'
}


# MET energy scale

nuisances['met'] = {
    'name': 'CMS_scale_met_2018',
    'kind': 'suffix',
    'type': 'shape',
    'mapUp': 'METup',
    'mapDown': 'METdo',
    'samples': dict((skey, ['1', '1']) for skey in mc),
    'folderUp': makeMCDirectory('METup_suffix'),
    'folderDown': makeMCDirectory('METdo_suffix'),
#    'AsLnN': '1'
}

# ##### Pileup

pu_syst = '(puWeightUp/puWeight)', '(puWeightDown/puWeight)'

nuisances['PU'] = {
    'name': 'CMS_PU_2018',
    'kind': 'weight',
    'type': 'shape',
    'samples': dict((skey, pu_syst) for skey in mc),
#    'AsLnN': '1',
}

##### PS
nuisances['PS_ISR']  = {
    'name': 'PS_ISR',
    'kind': 'weight',
    'type': 'shape',
    'samples': dict((skey, ['PSWeight[2]', 'PSWeight[0]']) for skey in mc if skey not in ['Vg','VgS','WWewk']), #PSWeights are buggy for some samples, we add them back by hand below
}

nuisances['PS_FSR']  = {
    'name': 'PS_FSR',
    'kind': 'weight',
    'type': 'shape',
    'samples': dict((skey, ['PSWeight[3]', 'PSWeight[1]']) for skey in mc if skey not in ['Vg','VgS','WWewk']), #PSWeights are buggy for some samples, we add them back by hand below
}


## PS nuisances computed by hand as a function of nCleanGenJets using alternative samples (when available). Needed if nominal samples have buggy PSWeights
nuisances['PS_ISR_ForBuggySamples']  = {
    'name': 'PS_ISR',
    'kind': 'weight',
    'type': 'shape',
    'samples': {
        'Vg'     : ['1.00227428567253*(nCleanGenJet==0) + 1.00572014989997*(nCleanGenJet==1) + 0.970824885256465*(nCleanGenJet==2) + 0.927346068071086*(nCleanGenJet>=3)', '0.996488506572636*(nCleanGenJet==0) + 0.993582795375765*(nCleanGenJet==1) + 1.03643678934568*(nCleanGenJet==2) + 1.09735277266955*(nCleanGenJet>=3)'],
        'VgS'    : ['1.0000536116408023*(nCleanGenJet==0) + 1.0100100693580492*(nCleanGenJet==1) + 0.959068359375*(nCleanGenJet==2) + 0.9117049260469496*(nCleanGenJet>=3)', '0.9999367833485968*(nCleanGenJet==0) + 0.9873682892005163*(nCleanGenJet==1) + 1.0492717737268518*(nCleanGenJet==2) + 1.1176958835210322*(nCleanGenJet>=3)'],
    },
}

nuisances['PS_FSR_ForBuggySamples']  = {
    'name': 'PS_FSR',
    'kind': 'weight',
    'type': 'shape',
    'samples': {
        'Vg'     : ['0.999935529935028*(nCleanGenJet==0) + 0.997948255568351*(nCleanGenJet==1) + 1.00561645493085*(nCleanGenJet==2) + 1.0212896960035*(nCleanGenJet>=3)', '1.00757702771109*(nCleanGenJet==0) + 1.00256681166083*(nCleanGenJet==1) + 0.93676371569867*(nCleanGenJet==2) + 0.956448336052435*(nCleanGenJet>=3)'],
        'VgS'    : ['0.9976593177227735*(nCleanGenJet==0) + 1.0016125187585532*(nCleanGenJet==1) + 1.0049344618055556*(nCleanGenJet==2) + 1.0195631514301164*(nCleanGenJet>=3)', '1.0026951855766457*(nCleanGenJet==0) + 1.0008132148661049*(nCleanGenJet==1) + 1.003949291087963*(nCleanGenJet==2) + 0.9708160910230832*(nCleanGenJet>=3)'],
    },
}

# An overall 1.5% UE uncertainty will cover all the UEup/UEdo variations
# And we don't observe any dependency of UE variations on njet
nuisances['UE']  = {
                'name'  : 'UE_CP5',
                'skipCMS' : 1,
                'type': 'lnN',
                'samples': dict((skey, '1.015') for skey in mc), 
}

# nuisances['UE']  = {
#                'name'  : 'UE_CP5',
#                'skipCMS' : 1,
#                'kind'  : 'tree',
#                'type'  : 'shape',
#                'samples'  : {
#                  'WW'      : ['1.0017139', '0.99350287'],
#                  'ggH_hww' : ['1.0272226', '1.0123689'],
#                  'qqH_hww' : ['1.0000192', '0.98367442']
#                },
#                'folderUp': makeMCDirectory('UEup'),
#                'folderDown': makeMCDirectory('UEdo'),
#                'AsLnN'      : '1',
# }

# ####### Generic "cross section uncertainties"

# apply_on = {
#     'top': [
#         'isSingleTop * 1.0816 + isTTbar',
#         'isSingleTop * 0.9184 + isTTbar'
#     ]
# }

# nuisances['singleTopToTTbar'] = {
#     'name': 'singleTopToTTbar',
#     'skipCMS': 1,
#     'kind': 'weight',
#     'type': 'shape',
#     'samples': apply_on
# }

# ## Top pT reweighting uncertainty

nuisances['TopPtRew'] = {
    'name': 'CMS_topPtRew',   # Theory uncertainty
    'kind': 'weight',
    'type': 'shape',
    'samples': {'top': ["1.", "1./Top_pTrw"]},
    'symmetrize': True
}

nuisances['VgStar'] = {
    'name': 'CMS_hww_VgStarScale',
    'type': 'lnN',
    'samples': {
        'VgS_L': '1.25'
    }
}

nuisances['VZ'] = {
    'name': 'CMS_hww_VZScale',
    'type': 'lnN',
    'samples': {
        'VgS_H': '1.16'
    }
}

nuisances['pdf']  = {
               'name'  : 'pdf',
               'type'  : 'lnN',
               'samples'  : {
                   'ggWW'    : '1.05',
                   'WW'      : '1.04',
                   'Vg'      : '1.04',
                   'VZ'      : '1.04',
                   'VgS'     : '1.04',
                   'Higgs'     : '1.04',
                   'DY'      : '1.002', # For HM category, no DY CR
                   },
              }


# ##### Renormalization & factorization scales

# ## Shape nuisance due to QCD scale variations for DY
# # LHE scale variation weights (w_var / w_nominal)

# ## This should work for samples with either 8 or 9 LHE scale weights (Length$(LHEScaleWeight) == 8 or 9)
# variations = ['LHEScaleWeight[0]', 'LHEScaleWeight[1]', 'LHEScaleWeight[3]', 'LHEScaleWeight[Length$(LHEScaleWeight)-4]', 'LHEScaleWeight[Length$(LHEScaleWeight)-2]', 'LHEScaleWeight[Length$(LHEScaleWeight)-1]']

# nuisances['QCDscale_V'] = {
#     'name': 'QCDscale_V',
#     'skipCMS': 1,
#     'kind': 'weight_envelope',
#     'type': 'shape',
#     'samples': {'DY': variations},
#     'AsLnN': '1'
# }

# nuisances['QCDscale_VV'] = {
#     'name': 'QCDscale_VV',
#     'kind': 'weight_envelope',
#     'type': 'shape',
#     'samples': {
#         'Vg': variations,
#         'VZ': variations,
#         'VgS': variations
#     }
# }

# nuisances['QCDscale_ggVV'] = {
#     'name': 'QCDscale_ggVV',
#     'type': 'lnN',
#     'samples': {
#         'ggWW': '1.15',
#     },
# }

##### Renormalization & factorization scales
nuisances['WWresum']  = {
  'name'  : 'CMS_hww_WWresum',
  'skipCMS' : 1,
  'kind'  : 'weight',
  'type'  : 'shape',
  'samples'  : {
     'WW'   : ['nllW_Rup/nllW', 'nllW_Rdown/nllW'],
   },
}

nuisances['WWqscale']  = {
   'name'  : 'CMS_hww_WWqscale',
   'skipCMS' : 1,
   'kind'  : 'weight',
   'type'  : 'shape',
   'samples'  : {
      'WW'   : ['nllW_Qup/nllW', 'nllW_Qdown/nllW'],
    },
}

# # # Uncertainty on SR/CR ratio
# # nuisances['CRSR_accept_DY'] = {
# #     'name': 'CMS_hww_CRSR_accept_DY',
# #     'type': 'lnN',
# #     'samples': {'DY': '1.02'},
# #     'cuts': [cut for cut in cuts if '_CR_' in cut],
# #     'cutspost': (lambda self, cuts: [cut for cut in cuts if '_DY_' in cut]),
# # }

# # Uncertainty on SR/CR ratio
# nuisances['CRSR_accept_top'] = {
#     'name': 'CMS_hww_CRSR_accept_top',
#     'type': 'lnN',
#     'samples': {'top': '1.01'},
#     'cuts': [cut for cut in cuts if 'TopCR' in cut],
#     'cutspost': (lambda self, cuts: [cut for cut in cuts if 'TopCR' in cut]),
# }

# # Theory uncertainty for ggH
# #
# #
# #   THU_ggH_Mu, THU_ggH_Res, THU_ggH_Mig01, THU_ggH_Mig12, THU_ggH_VBF2j, THU_ggH_VBF3j, THU_ggH_PT60, THU_ggH_PT120, THU_ggH_qmtop
# #
# #   see https://twiki.cern.ch/twiki/bin/viewauth/CMS/HiggsWG/SignalModelingTools

# thus = [
#     ('THU_ggH_Mu', 'ggH_mu'),
#     ('THU_ggH_Res', 'ggH_res'),
#     ('THU_ggH_Mig01', 'ggH_mig01'),
#     ('THU_ggH_Mig12', 'ggH_mig12'),
#     ('THU_ggH_VBF2j', 'ggH_VBF2j'),
#     ('THU_ggH_VBF3j', 'ggH_VBF3j'),
#     ('THU_ggH_PT60', 'ggH_pT60'),
#     ('THU_ggH_PT120', 'ggH_pT120'),
#     ('THU_ggH_qmtop', 'ggH_qmtop')
# ]

# for name, vname in thus:
#     updown = [vname, '2.-%s' % vname]
    
#     nuisances[name] = {
#         'name': name,
#         'skipCMS': 1,
#         'kind': 'weight',
#         'type': 'shape',
#         'samples': {
#           'ggH_hww': updown,
#           #'ggH_htt': updown
#         }
#     }

# #### QCD scale uncertainties for Higgs signals other than ggH

# values = HiggsXS.GetHiggsProdXSNP('YR4','13TeV','vbfH','125.09','scale','sm')

# nuisances['QCDscale_qqH'] = {
#     'name': 'QCDscale_qqH', 
#     'samples': {
#         'qqH_hww': values,
#         'qqH_htt': values
#     },
#     'type': 'lnN'
# }

# valueswh = HiggsXS.GetHiggsProdXSNP('YR4','13TeV','WH','125.09','scale','sm')
# valueszh = HiggsXS.GetHiggsProdXSNP('YR4','13TeV','ZH','125.09','scale','sm')

# nuisances['QCDscale_VH'] = {
#     'name': 'QCDscale_VH', 
#     'samples': {
#         'WH_hww': valueswh,
#         'WH_htt': valueswh,
#         'ZH_hww': valueszh,
#         'ZH_htt': valueszh
#     },
#     'type': 'lnN',
# }

# values = HiggsXS.GetHiggsProdXSNP('YR4','13TeV','ggZH','125.09','scale','sm')

# nuisances['QCDscale_ggZH'] = {
#     'name': 'QCDscale_ggZH', 
#     'samples': {
#         'ggZH_hww': values
#     },
#     'type': 'lnN',
# }

# values = HiggsXS.GetHiggsProdXSNP('YR4','13TeV','ttH','125.09','scale','sm')

# nuisances['QCDscale_ttH'] = {
#     'name': 'QCDscale_ttH',
#     'samples': {
#         'ttH_hww': values
#     },
#     'type': 'lnN',
# }

# nuisances['QCDscale_WWewk'] = {
#     'name': 'QCDscale_WWewk',
#     'samples': {
#         'WWewk': '1.11',
#     },
#     'type': 'lnN'
# }

# nuisances['QCDscale_qqbar_ACCEPT'] = {
#     'name': 'QCDscale_qqbar_ACCEPT',
#     'type': 'lnN',
#     'samples': {
#         'qqH_hww': '1.003',
#         'qqH_htt': '1.003',
#         'WH_hww': '1.010',
#         'WH_htt': '1.010',
#         'ZH_hww': '1.015',
#         'ZH_htt': '1.015',
#     }
# }

# #FIXME: these come from HIG-16-042, maybe should be recomputed?
# nuisances['QCDscale_gg_ACCEPT'] = {
#     'name': 'QCDscale_gg_ACCEPT',
#     'samples': {
#         'ggH_htt': '1.012',
#         'ggH_hww': '1.012',
#         'ggZH_hww': '1.012',
#         'ggWW': '1.012',
#     },
#     'type': 'lnN',
# }

## Use the following if you want to apply the automatic combine MC stat nuisances.
nuisances['stat'] = {
    'type': 'auto',
    'maxPoiss': '10',
    'includeSignal': '0',
    #  nuisance ['maxPoiss'] =  Number of threshold events for Poisson modelling
    #  nuisance ['includeSignal'] =  Include MC stat nuisances on signal processes (1=True, 0=False)
    'samples': {}
}

# ##rate parameters
nuisances['Topnorm']  = {
               'name'  : 'Topnorm_2018',
               'samples'  : {
                   'top' : '1.00',
                   },
               'type'  : 'rateParam',
               'cuts'  : [
                   'higgs_sr_',
                   'tt_cr_',
                   'WW_cr_',
                   'DY_cr_',
                   ]
              }


nuisances['WWnorm']  = {
               'name'  : 'WWnorm_2018',
               'samples'  : {
                   'WW' : '1.00',
                   },
               'type'  : 'rateParam',
               'cuts'  : [
                   'higgs_sr_',
                   'tt_cr_',
                   'WW_cr_',
                   'DY_cr_',
                   ]
              }




nuisances['DYnorm']  = {
               'name'  : 'DYnorm_2018',
               'samples'  : {
                   'DY' : '1.00',
                   },
               'type'  : 'rateParam',
               'cuts'  : [
                   'higgs_sr_',
                   'tt_cr_',
                   'WW_cr_',
                   'DY_cr_',
                   ]
              }



for n in nuisances.values():
    n['skipCMS'] = 1

# print ' '.join(nuis['name'] for nname, nuis in nuisances.iteritems() if nname not in ('lumi', 'stat'))
