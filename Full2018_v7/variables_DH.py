variables = {}

variables['events']  = {   
    'name': '1',
    'range' : (1,0,2),
    'xaxis' : 'events',
    'fold' : 3
}
"""

variables['mll']  = {   'name': 'mll',
                        'range' : (40, 0, 320),
                        'xaxis' : 'm_{ll} [GeV]',
                        'fold' : 3
                        }



variables['drll']  = {   'name': 'drll',
                         'range' : (30, 0, 2.5),
                        'xaxis' : '#Deltar_{ll} [GeV]',
                        'fold' : 3
                        }

"""
variables['mth']  = {   'name': 'mth',
                        'range' : (30, 50, 500),
                        'xaxis' : 'm_{T}^{ll+MET} [GeV]',
                        'fold' : 3
                        }

variables['mtw2']  = {   'name': 'mtw2',
                        'range' : (30, 20, 500),
                         'xaxis' : 'm_{T}^{l2+MET} [GeV]',
                        'fold' : 3
                        }
"""
variables['ptll']  = {   'name': 'ptll',
                        'range' : (30, 30,250),
                        'xaxis' : 'p_{T}^{ll} [GeV]',
                        'fold' : 3
                        }

variables['pt1']  = {   'name': 'Lepton_pt[0]',
                        'range' : (30,25,250),
                        'xaxis' : 'p_{T} 1st lep',
                        'fold'  : 3
                        }

variables['pt2']  = {   'name': 'Lepton_pt[1]',
                        'range' : (30,0,140),
                        'xaxis' : 'p_{T} 2nd lep',
                        'fold'  : 3
                        }


variables['eta1']  = {   'name': 'Lepton_eta[0]',
                        'range' : (30,-3,3),
                        'xaxis' : '#eta 1st lep',
                        'fold'  : 3
                        }

variables['eta2']  = {  'name': 'Lepton_eta[1]',
                        'range' : (30,-3,3),
                        'xaxis' : '#eta 2nd lep',
                        'fold'  : 3
                        }

variables['puppimet']  = {'name': 'PuppiMET_pt',
                        'range' : (30,20,250),
                        'xaxis' : 'puppimet [GeV]',
                        'fold'  : 3
                        }

variables['mpmet']  = { 'name': 'mpmet',
                        'range' : (30,20,400),
                        'xaxis' : 'mpmet [GeV]',
                        'fold'  : 3
                        }

variables['dphill']  = { 'name': 'abs(dphill)',
                        'range' : (30,0,3.14),
                        'xaxis' : '#Delta#phi_{ll}',
                        'fold' : 3
                        }
variables['njets']  = { 'name': 'njets',
                        'range' : (7,0,6),
                        'xaxis' : 'N jets',
                        'fold' : 3
                        }
variables['nbjets']  = { 'name': 'nbjets',
                        'range' : (7,0,6),
                        'xaxis' : 'N b jets',
                        'fold' : 3
                        }

variables['btag_ID']  = { 'name': 'btag_ID',
                        'range' : (30,0,1),
                        'xaxis' : 'b tag',
                        'fold' : 3
                        }
variables['first_btag_ID']  = { 'name': 'first_btag_ID',
                        'range' : (30,-1,1),
                        'xaxis' : '1st b tag',
                        'fold' : 3
                        }
variables['second_btag_ID']  = { 'name': 'second_btag_ID',
                        'range' : (30,-1,1),
                        'xaxis' : '2nd b tag',
                        'fold' : 3
                        }
"""
## VARIABLE FOR THE DARK HIGGS ANALYSIS
"""
variables['mT2']  = { 'name': 'mT2',
                        'range' : (30,0,500),
                        'xaxis' : 'mT2',
                        'fold' : 3
                        }
variables['mTe']  = { 'name': 'mTe',
                        'range' : (30,160,900),
                        'xaxis' : 'mTe',
                        'fold' : 3
                        }
variables['mTi']  = { 'name': 'mTi',
                        'range' : (30,60,900),
                        'xaxis' : 'mTi',
                        'fold' : 3
                        }
variables['detall']  = { 'name': 'detall',
                        'range' : (30,0,3),
                        'xaxis' : 'detall',
                        'fold' : 3
                        }
variables['dphillmet']  = { 'name': 'dphillmet',
                        'range' : (30,0,3.14),
                        'xaxis' : 'dphillmet',
                        'fold' : 3
                        }
variables['dphilmet']  = { 'name': 'dphilmet',
                        'range' : (30,0,3.14),
                        'xaxis' : 'dphilmet',
                        'fold' : 3
                        }
variables['projpfmet']  = { 'name': 'projpfmet',
                        'range' : (30,0,400),
                        'xaxis' : 'projpfmet',
                        'fold' : 3
                        }
variables['projtkmet']  = { 'name': 'projtkmet',
                        'range' : (30,0,400),
                        'xaxis' : 'projtkmet',
                        'fold' : 3
                        }
variables['mcoll']  = { 'name': 'mcoll',
                        'range' : (30,0,225),
                        'xaxis' : 'mcoll',
                        'fold' : 3
                        }
variables['mcollWW']  = { 'name': 'mcollWW',
                        'range' : (30,140,350),
                        'xaxis' : 'mcollww',
                        'fold' : 3
                        }
variables['mR']  = { 'name': 'mR',
                        'range' : (30,25,350),
                        'xaxis' : 'mR',
                        'fold' : 3
                        }
variables['upara']  = { 'name': 'upara',
                        'range' : (30,0,30),
                        'xaxis' : 'upara',
                        'fold' : 3
                        }
variables['uperp']  = { 'name': 'uperp',
                        'range' : (30,0,30),
                        'xaxis' : 'uperp',
                        'fold' : 3
                        }
variables['recoil']  = { 'name': 'recoil',
                        'range' : (30,0,30),
                        'xaxis' : 'recoil',
                        'fold' : 3
                        }
variables['dPhillStar']  = { 'name': 'dPhillStar',
                        'range' : (30,-0.5,0.5),
                        'xaxis' : '#Delta#phi_{ll}*',
                        'fold' : 3
                        }
"""
variables['dPhill_Zp']  = { 'name': 'dPhill_Zp',
                        'range' : (30,0,3.14),
                        'xaxis' : '#Delta#phi_{ll,Zp}*',
                        'fold' : 3
                        }
"""
variables['Cos_theta']  = { 'name': 'Cos_theta',
                        'range' : (30,-1,1),
                        'xaxis' : 'cos(#theta)',
                        'fold' : 3
                        }
variables['Theta_ll']  = { 'name': 'Theta_ll',
                        'range' : (30,0,3.14),
                        'xaxis' : '#theta_{ll}*',
                        'fold' : 3
                        }

variables['dPhill_MET']  = { 'name': 'dPhill_MET',
                        'range' : (30,0,3.14),
                        'xaxis' : '#Delta #phi_{ll,MET}*',
                        'fold' : 3
                        }


# ms=160
variables['evaluate_rf_mhs_160_mx_150_mZp_800']  = { 'name': 'evaluate_rf_mhs_160_mx_150_mZp_800',
                        'range' : (30,0,1),
                        'xaxis' : 'Discriminant m_{s}=160, m_{#chi}=150, m_{Z}=800',
                        'fold' : 3
                        }
variables['evaluate_rf_mhs_180_mx_150_mZp_800']  = { 'name': 'evaluate_rf_mhs_180_mx_150_mZp_800',
                        'range' : (30,0,1),
                        'xaxis' : 'Discriminant m_{s}=180, m_{#chi}=150, m_{Z}=800',
                        'fold' : 3
                        }
variables['evaluate_rf_mhs_200_mx_150_mZp_800']  = { 'name': 'evaluate_rf_mhs_200_mx_150_mZp_800',
                        'range' : (30,0,1),
                        'xaxis' : 'Discriminant m_{s}=200, m_{#chi}=150, m_{Z}=800',
                        'fold' : 3
                        }
variables['evaluate_rf_mhs_300_mx_150_mZp_800']  = { 'name': 'evaluate_rf_mhs_300_mx_150_mZp_800',
                        'range' : (30,0,1),
                        'xaxis' : 'Discriminant m_{s}=300, m_{#chi}=150, m_{Z}=800',
                        'fold' : 3
}

# DNN discriminator
variables['evaluate_dnn_mhs_160_mx_150_mZp_800']  = { 'name': 'evaluate_dnn_mhs_160_mx_150_mZp_800',
                        'range' : (30,0,1),
                        'xaxis' : 'Discriminant m_{s}=160, m_{#chi}=150, m_{Z}=800',
                        'fold' : 3
                        }
variables['evaluate_dnn_mhs_180_mx_150_mZp_800']  = { 'name': 'evaluate_dnn_mhs_180_mx_150_mZp_800',
                        'range' : (30,0,1),
                        'xaxis' : 'Discriminant m_{s}=180, m_{#chi}=150, m_{Z}=800',
                        'fold' : 3
                        }
variables['evaluate_dnn_mhs_200_mx_150_mZp_800']  = { 'name': 'evaluate_dnn_mhs_200_mx_150_mZp_800',
                        'range' : (30,0,1),
                        'xaxis' : 'Discriminant m_{s}=200, m_{#chi}=150, m_{Z}=800',
                        'fold' : 3
                        }
variables['evaluate_dnn_mhs_300_mx_150_mZp_800']  = { 'name': 'evaluate_dnn_mhs_300_mx_150_mZp_800',
                        'range' : (30,0,1),
                        'xaxis' : 'Discriminant m_{s}=300, m_{#chi}=150, m_{Z}=800',
                        'fold' : 3
                        }
"""
