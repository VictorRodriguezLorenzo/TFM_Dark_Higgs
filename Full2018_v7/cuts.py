# cuts

cuts = {}


_tmp = [
    'Lepton_pdgId[0]*Lepton_pdgId[1] == -11*13',
    'Lepton_pt[0] > 25.',
    'Lepton_pt[1] > 20.',
    'PuppiMET_pt > 20',
    'mpmet > 20.',
    'Alt(Lepton_pt,2, 0) < 10.',
     ]

preselections = ' && '.join(_tmp)

#cuts['main_cuts_'] = '1'

cuts['WW_sr_'] = {
    'expr': 'mth>50 && ptll>30 && mll>20 && bVeto',
    'categories' : {
        '0jet' : 'zeroJet',
        '1jet' : 'oneJet ',
    }
}

#cuts['tt_cr_'] = 'mth > 50 && ptll>30 && mll>20 && bReq'

#cuts['DY_cr_'] = {
#    'expr': 'mth < 50 && ptll<30 && mll<80 && bVeto',
#    'categories' : {
#        '0jet' : 'zeroJet',
#        '1jet' : 'oneJet',
#    }
#}






