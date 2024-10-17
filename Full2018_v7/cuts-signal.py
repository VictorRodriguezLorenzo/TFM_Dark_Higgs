# cuts

cuts = {}


_tmp = [
    'Lepton_pdgId[0]*Lepton_pdgId[1] == -11*13',
    'Lepton_pt[0] > 25.',
    'Lepton_pt[1] > 20.',
    'PuppiMET_pt >20',
    'mpmet > 20.',
    'ptll>30',
    'mll>12',
    'Alt(Lepton_pt,2, 0) < 10.',
    'mth > 50',
    'bVeto',
     ]

preselections = ' && '.join(_tmp)

cuts['higgs_sr_'] = 'drll < 2.5'





