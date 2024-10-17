variables = {}

variables['events']  = {
    'name': '1',
    'range' : (1,0,2),
    'xaxis' : 'events',
    'fold' : 3
}

#mhs = ['160','180','200','300']
mhs = ['160']
#mDM = ['100','150','200','300']
mDM = ['100']
mZp = ['200','300','400','500','800','1000','1200','1500','2000','2500']

variables['evaluate_normal_dnn']  = { 'name': 'evaluate_normal_dnn',
                        'range' : (30,0,1),
                        #'range' : ([0.5, 0.53, 0.56, 0.59, 0.62, 0.65, 0.68, 0.71, 0.74, 0.77, 0.80, 0.83, 0.86, 0.89, 0.92, 1],),
                        'xaxis' : 'Discriminant normal DNN',
                        'fold' : 3
                        }

# Parametric
"""
for hs in mhs:
    for DM in mDM:
        for Zp in mZp:
            if DM == '300' and Zp in ['200', '300', '400', '500']:
                continue
            elif hs == '300' and DM == '300' and Zp in ['1500', '2000', '2500']:
                continue
            elif hs == '300' and DM == '200' and Zp in ['200', '300', '400', '2000', '2500']:
                continue
            elif hs == '300' and DM == '150' and Zp in ['200', '300', '2000', '2500']:
                continue
            elif hs == '300' and DM == '100':
                continue
            variables['evaluate_rf_mhs_' + hs + '_mx_' + DM + '_mZp_' + Zp]  = { 'name': 'evaluate_rf_mhs_' + hs + '_mx_' + DM + '_mZp_' + Zp,
                        'range' : (30,0,1),
                        #'range' : ([0.5, 0.53, 0.56, 0.59, 0.62, 0.65, 0.68, 0.71, 0.74, 0.77, 0.80, 0.83, 0.86, 0.89, 0.92, 1],),
                        'xaxis' : 'Discriminant RF m_{s}='+ hs + ', m_{#chi}='+ DM + ', m_{Z}=' + Zp,
                        'fold' : 3
                        }
for hs in mhs:
    for DM in mDM:
        for Zp in mZp:
            if DM == '300' and Zp in ['200', '300', '400', '500']:
                continue
            elif hs == '300' and DM == '300' and Zp in ['1500', '2000', '2500']:
                continue
            elif hs == '300' and DM == '200' and Zp in ['200', '300', '400', '2000', '2500']:
                continue
            elif hs == '300' and DM == '150' and Zp in ['200', '300', '2000', '2500']:
                continue
            elif hs == '300' and DM == '100':
                continue
            variables['evaluate_dnn_mhs_' + hs + '_mx_' + DM + '_mZp_' + Zp]  = { 'name': 'evaluate_dnn_mhs_' + hs + '_mx_' + DM + '_mZp_' + Zp,
                        #'range' : (30,0,1),
                        'range' : ([0.5, 0.53, 0.56, 0.59, 0.62, 0.65, 0.68, 0.71, 0.74, 0.77, 0.80, 0.83, 0.86, 0.89, 0.92, 1],),
                        'xaxis' : 'Discriminant DNN m_{s}='+ hs + ', m_{#chi}='+ DM + ', m_{Z}=' + Zp,
                        'fold' : 3
                                                }
"""
"""
variables['dphill']  = { 'name': 'abs(dphill)',
                        'range' : (30,0,3.14),
                        'xaxis' : '#Delta#phi_{ll}',
                        'fold' : 3
                        }
"""
