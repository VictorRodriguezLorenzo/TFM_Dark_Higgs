# cuts

cuts = {}


_tmp = [
    'Lepton_pt[0] > 25.',
    'Lepton_pt[0] > 20.',
    'PuppiMET_pt >20',
    'mpmet > 20.',
    'mll>12',
    'Alt(Lepton_pt,2, 0) < 10.',
     ]

preselections = ' && '.join(_tmp)

cuts['main_cuts_'] = 'Lepton_pdgId[0]*Lepton_pdgId[1] == -11*13'

cuts['WW_cr_'] = 'drll>2.5 && mth>50 && bVeto && Lepton_pdgId[0]*Lepton_pdgId[1] == -11*13'

cuts['tt_cr_'] = 'drll<2.5 && mth>50 && bReq && Lepton_pdgId[0]*Lepton_pdgId[1] == -11*13'

cuts['DY_cr_'] = 'drll<2.5 && mth<50 && bVeto && Lepton_pdgId[0]*Lepton_pdgId[1] == -11*13'

cuts['non_prompt_cr_'] = 'drll<2.5 && mth<50 && bVeto && Lepton_pdgId[0]*Lepton_pdgId[1] == 11*13'




