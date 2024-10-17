

groupPlot = {}
plot = {}

# groupPlot = {}
# 
# Groups of samples to improve the plots.
# If not defined, normal plots is used
#


groupPlot['Fake']  = {
                  'nameHR' : 'nonprompt',
                  'isSignal' : 0,
                  'color': 921,    # kGray + 1
                  # 'samples'  : ['Fake_me', 'Fake_em']
                  'samples'  : ['Fake']
}

groupPlot['VZ']  = {  
                  'nameHR' : "VZ",
                  'isSignal' : 0,
                  'color'    : 617,   # kViolet + 1  
                  'samples'  : ['VZ']
              }


groupPlot['WW']  = {  
                  'nameHR' : 'WW',
                  'isSignal' : 0,
                  'color': 851, # kAzure -9 
                  'samples'  : ['WW', 'ggWW', 'WWewk']
              }



groupPlot['VgS'] = {
		  'nameHR' : "V#gamma*",
                  'isSignal' : 0,
                  'color': 810,   # kYellow
                  'samples'  : ['VgS']
              }

groupPlot['Vg'] = {
          'nameHR' : "V#gamma",
                  'isSignal' : 0,
                  'color': 857,   # kAzure-3
                  'samples'  : ['Vg']
              }


groupPlot['top']  = {  
                  'nameHR' : 't#bar{t}',
                  'isSignal' : 0,
                  'color': 400,   # kOrange+10
                  'samples'  : ['top']
              }


groupPlot['DY']  = {  
                  'nameHR' : "DY",
                  'isSignal' : 0,
                  'color': 418,    # kGreen+2
                  'samples'  : ['DY']
              }



groupPlot['Higgs']  = {
                  'nameHR' : 'Higgs',
                  'isSignal' : 0,
                  'color': 632, # kRed
                  'samples'  : ['Higgs' ]
              }


groupPlot['VVV']  = {
                  'nameHR' : 'VVV',
                  'isSignal' : 0,
                  'color': 800, 
                  'samples'  : ['VVV']
              }


'''
mhs = ['160']
mDM = ['100']
mZp = ['300','400','500','800','1000','1200','1500']

j=0
for hs in mhs:
    for DM in mDM:
        for Zp in mZp:
            j+=100
            groupPlot['DH_' + hs  +  '_'   + DM + '_' + Zp]  = {
                'nameHR' : 'DH_' + hs  + '_' + DM + '_' + Zp ,
                'isSignal' : 2,
                'color': 1 + j, # kRed
                'samples'  : ['DH_mhs_' + hs + '_mx_' + DM +  '_mZp_' + Zp]
            }
'''


#plot = {}

# keys here must match keys in samples.py    
#  


plot['DY']  = {
                  'color': 418,    # kGreen+2
                  'isSignal' : 0,
                  'isData'   : 0,
                  'scale'    : 1.0
}

'''
    'cuts'  : {
                       'hww2l2v_13TeV_top_0j'  : 0.76 ,
                       'hww2l2v_13TeV_dytt_0j' : 0.76 ,
                       'hww2l2v_13TeV_top_1j'  : 0.79 ,
                       'hww2l2v_13TeV_dytt_1j' : 0.79 ,
                       'hww2l2v_13TeV_WW_1j'     : 0.79 ,
                       'hww2l2v_13TeV_WW_noVeto_1j'     : 0.79 ,
                       'hww2l2v_13TeV_WP65_sr_1j' : 0.76,
                       'hww2l2v_13TeV_top_2j'  : 0.76 ,
                       'hww2l2v_13TeV_dytt_2j' : 0.76 ,
                       'hww2l2v_13TeV_WW_2j'     : 0.76 ,
                       'hww2l2v_13TeV_WW_noVeto_2j'     : 0.76 ,
                       'hww2l2v_13TeV_WP75_sr_2j' : 0.76,
                       'hww2l2v_13TeV_top_Inclusive'  : 0.77 ,
                       'hww2l2v_13TeV_dytt_Inclusive' : 0.77 ,
                       'hww2l2v_13TeV_WW_Inclusive'     : 0.77 ,
                        },
'''

plot['Fake']  = {
                  'color': 921,    # kGray + 1
                  'isSignal' : 0,
                  'isData'   : 0,
                  'scale'    : 1.0
              }



plot['top'] = {   
                  'nameHR' : 't#bar{t}',
                  'color': 400,   # kYellow
                  'isSignal' : 0,
                  'isData'   : 0, 
                  'scale'    : 1.0
                  #'cuts'  : {
                       #'hww2l2v_13TeV_of0j'      : 0.94 ,
                       #'hww2l2v_13TeV_top_of0j'  : 0.94 , 
                       #'hww2l2v_13TeV_dytt_of0j' : 0.94 ,
                       #'hww2l2v_13TeV_em_0j'     : 0.94 , 
                       #'hww2l2v_13TeV_me_0j'     : 0.94 , 
                       ##
                       #'hww2l2v_13TeV_of1j'      : 0.86 ,
                       #'hww2l2v_13TeV_top_of1j'  : 0.86 , 
                       #'hww2l2v_13TeV_dytt_of1j' : 0.86 ,
                       #'hww2l2v_13TeV_em_1j'     : 0.86 , 
                       #'hww2l2v_13TeV_me_1j'     : 0.86 , 
                        #},
                  }


plot['WW']  = {
                  'color': 851, # kAzure -9 
                  'isSignal' : 0,
                  'isData'   : 0,    
                  'scale'    : 1.0   # ele/mu trigger efficiency   datadriven
                  }

plot['ggWW']  = {
                  'color': 850, # kAzure -10
                  'isSignal' : 0,
                  'isData'   : 0,    
                  'scale'    : 1.0
                  }

plot['WWewk']  = {
                  'color': 851, # kAzure -9 
                  'isSignal' : 0,
                  'isData'   : 0,
                  'scale'    : 1.0   # ele/mu trigger efficiency   datadriven
                  }


plot['Vg']  = {
                  'color': 857,
                  'isSignal' : 0,
                  'isData'   : 0,
                  'scale'    : 1.0
                  }


plot['VgS']  = { 
    'color'    : 810,   
    'isSignal' : 0,
    'isData'   : 0,
    'scale'    : 1.0
}


plot['VZ']  = { 
                  'color': 858,
                  'isSignal' : 0,
                  'isData'   : 0,
                  'scale'    : 1.0
                  }


plot['Higgs']  = { 
                  'color': 632, # kAzure -3  
                  'isSignal' : 0,
                  'isData'   : 0,
                  'scale'    : 1.0
}


plot['VVV']  = { 
                  'color': 800, 
                  'isSignal' : 0,
                  'isData'   : 0,
                  'scale'    : 1.0
}

'''
j=0 
for hs in mhs:
    for DM in mDM:
        for Zp in mZp:
            j+=100
            plot['DH_mhs_' + hs + '_mx_' + DM  + '_mZp_' + Zp]  = {
                'color': 1 + j, # kRed
                'isSignal' : 2,
                'isData'   : 0,
                'scale'    : 1.0
            }
'''



# data

plot['DATA']  = { 
                  'nameHR' : 'Data',
                  'color': 1 ,  
                  'isSignal' : 0,
                  'isData'   : 1 ,
                  'isBlind'  : 0
              }

# additional options

legend = {}

legend['lumi'] = 'L = 59.74/fb'

legend['sqrt'] = '#sqrt{s} = 13 TeV'
