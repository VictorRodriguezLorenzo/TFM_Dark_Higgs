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

nuisances['lumi'] = {
   'name': 'lumi_13TeV_2018',
   'type': 'lnN',
   'samples': dict((skey, '1.025') for skey in mc if skey not in ['WW', 'top', 'DY'])
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
    'samples': dict((skey, ['1', '1']) for skey in mc), # if not skey.startswith('DH')),
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
    'samples': dict((skey, ['1', '1']) for skey in mc), # if not skey.startswith('DH')),
    'folderUp': makeMCDirectory('MupTup_suffix'),
    'folderDown': makeMCDirectory('MupTdo_suffix'),
#    'AsLnN': '1'
}

# ##### Jet energy scale

nuisances['jes'] = {
    'name': 'CMS_scale_JES_2018',
    'type': 'lnN',
    'samples': dict((skey, '1.01') for skey in mc), 
}

nuisances['jer'] = {
    'name': 'CMS_res_j_2018',
    'type': 'lnN',
    'samples': dict((skey, '1.005') for skey in mc), 
}

# MET energy scale

nuisances['met'] = {
    'name': 'CMS_scale_met_2018',
    'type'  : 'lnN',
    'samples'  : {
        'DH_mhs_180_mx_150_mZp_1200' : '1.01',
        'DY'      : '1.08',
        'ggWW'    : '1.02',
        'WW'      : '1.02',
        'top'      : '1.04',
        'WWewk'      : '1.02',
        'Vg'      : '1.06',
        'VZ'      : '1.06',
        'VgS'     : '1.06',
        'Higgs'     : '1.04',
        'VVV'      : '1.06'
    },
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
        'VgS': '1.20'
    }
}



###### pdf uncertainties
"""
# PDF eigenvariations for WW and top
for i in range(1,33):
  # LHEPdfWeight are PDF4LHC variations, while nominal is NNPDF.
  # LHEPdfWeight[i] reweights from NNPDF nominal to PDF4LHC member i
  # LHEPdfWeight[0] in particular reweights from NNPDF nominal to PDF4LHC nominal
  pdf_variations = ["LHEPdfWeight[%d]/LHEPdfWeight[0]" %i, "2. - LHEPdfWeight[%d]/LHEPdfWeight[0]" %i ]

  nuisances['pdf_WW_eigen'+str(i)]  = {
    'name'  : 'CMS_hww_pdf_WW_eigen'+str(i),
    'skipCMS' : 1,
    'kind'  : 'weight',
    'type'  : 'shape',
    'samples'  : {
      'WW'   : pdf_variations,
    },
  }
  nuisances['pdf_top_eigen'+str(i)]  = {
    'name'  : 'CMS_hww_pdf_top_eigen'+str(i),
    'skipCMS' : 1,
    'kind'  : 'weight',
    'type'  : 'shape',
    'samples'  : {
      'top'   : pdf_variations,
    },
  }
"""

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
                    'DY'      : '1.002', 
                    },
               }


# ##### Renormalization & factorization scales
"""
variations = [
    'LHEScaleWeight[0]', 
    'LHEScaleWeight[1]', 
    'LHEScaleWeight[3]', 
    'LHEScaleWeight[LHEScaleWeight.size()-4]', 
    'LHEScaleWeight[LHEScaleWeight.size()-2]', 
    'LHEScaleWeight[LHEScaleWeight.size()-1]'
]

variations_dy = [
    'LHEScaleWeight[0]/(656.3/637.2)', 
    'LHEScaleWeight[1]/(688.3/637.2)', 
    'LHEScaleWeight[3]/(624.9/637.2)', 
    'LHEScaleWeight[LHEScaleWeight.size()-4]/(644.9/637.2)', 
    'LHEScaleWeight[LHEScaleWeight.size()-2]/(610.3/637.2)', 
    'LHEScaleWeight[LHEScaleWeight.size()-1]/(617.7/637.2)'
]

variations_top = [
    'LHEScaleWeight[0]/(15711.2/14288.2)', 
    'LHEScaleWeight[1]/(15614.4/14288.2)', 
    'LHEScaleWeight[3]/(14412.3/14288.2)', 
    'LHEScaleWeight[LHEScaleWeight.size()-4]/(14217.4/14288.2)', 
    'LHEScaleWeight[LHEScaleWeight.size()-2]/(13038.6/14288.2)', 
    'LHEScaleWeight[LHEScaleWeight.size()-1]/(12940.7/14288.2)'
]




# NOT WORKING
variations = ['LHEScaleWeight[0]', 'LHEScaleWeight[1]', 'LHEScaleWeight[3]', 'LHEScaleWeight[Length$(LHEScaleWeight)-4]', 'LHEScaleWeight[Length$(LHEScaleWeight)-2]', 'LHEScaleWeight[Length$(LHEScaleWeight)-1]']
variations_dy = ['LHEScaleWeight[0]/(656.3/637.2)', 'LHEScaleWeight[1]/(688.3/637.2)', 'LHEScaleWeight[3]/(624.9/637.2)', 'LHEScaleWeight[Length$(LHEScaleWeight)-4]/(644.9/637.2)', 'LHEScaleWeight[Length$(LHEScaleWeight)-2]/(610.3/637.2)', 'LHEScaleWeight[Length$(LHEScaleWeight)-1]/(617.7/637.2)']
variations_top = ['LHEScaleWeight[0]/(15711.2/14288.2)', 'LHEScaleWeight[1]/(15614.4/14288.2)', 'LHEScaleWeight[3]/(14412.3/14288.2)', 'LHEScaleWeight[Length$(LHEScaleWeight)-4]/(14217.4/14288.2)', 'LHEScaleWeight[Length$(LHEScaleWeight)-2]/(13038.6/14288.2)', 'LHEScaleWeight[Length$(LHEScaleWeight)-1]/(12940.7/14288.2)']



nuisances['QCDscale_V'] = {
    'name': 'QCDscale_V',
    'kind': 'weight',
    'type': 'shape',
    'samples': {'DY': variations_dy},
}


nuisances['QCDscale_VV'] = {
    'name': 'QCDscale_VV',
    'kind': 'weight',
    'type': 'shape',
    'samples': {
        'Vg': variations,
        'VZ': variations,
        'VgS': variations
    },
    'symmetrize': True
}

nuisances['QCDscale_top']  = {
               'name'  : 'QCDscale_top', 
                'kind'  : 'weight',
                'type'  : 'shape',
                'samples'  : {
                   'top' : variations_top,
                   }
}
"""
nuisances['QCDscale_WWewk']  = {
    'name'  : 'QCDscale_WWewk',
    'type'  : 'lnN',
    'samples'  : {
        'WWewk' : '1.11',
    },
}

nuisances['QCDscale_ggVV'] = {
    'name': 'QCDscale_ggVV',
    'type': 'lnN',
    'samples': {
        'ggWW': '1.15',
    },
}


##### Renormalization & factorization scales
nuisances['WWresum']  = {
  'name'  : 'CMS_hww_WWresum',
  'skipCMS' : 1,
  'kind'  : 'weight',
  'type'  : 'shape',
  'samples'  : {
      'WW'   : ['(nllW_Rup/nllW)/(13695.2/13523.2)', '(nllW_Rdown/nllW)/(12894.6/13523.2)'],
   },
}

nuisances['WWqscale']  = {
   'name'  : 'CMS_hww_WWqscale',
   'skipCMS' : 1,
   'kind'  : 'weight',
   'type'  : 'shape',
   'samples'  : {
       'WW'   : ['(nllW_Qup/nllW)/(13428.4/13523.2)', '(nllW_Qdown/nllW)/(13604.7/13523.2)'],
    },
}


# WW EWK NLO correction uncertainty
nuisances['EWKcorr_WW'] = {
    'name': 'CMS_hww_EWKcorr_WW',
    'skipCMS': 1,
    'kind': 'weight',
    'type': 'shape',
    'samples': {
        'WW': ['1.', '1./ewknloW']
    },
    'symmetrize' : True,
}

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
