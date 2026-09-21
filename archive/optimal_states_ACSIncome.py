


#ACSIncome
optimal_states_2 = [['CA', 'MD']]
optimal_states_3 = [ ['CA', 'MD', 'MA'] ]
optimal_states_5 = [ ['CA', 'MD', 'VA', 'MA', 'CT'] ]



#Without X_i/X_j correlations

#No differential privacy
optimal_states_2_inf = [['MD', 'PA'],
                        ['CA', 'MD']]

optimal_states_2_01 = [['CA', 'MD'],
                       ['CA', 'MD'],
                       ['CA', 'MD'],
                       ['CA', 'MD'],
                       ['CA', 'MD'],
                       ['CA', 'MD'],
                       ['CA', 'MD'],
                       ['CA', 'MD'],
                       ['CA', 'MD'],
                       ['CA', 'MD']]

optimal_states_2_05 = [['CA', 'MD'],
                       ['CA', 'MD'],
                       ['CA', 'MD'],
                       ['CA', 'MD'],
                       ['CA', 'MD'],
                       ['CA', 'MD'],
                       ['CA', 'MD'],
                       ['CA', 'MD'],
                       ['CA', 'MD'],
                       ['CA', 'MA']]

optimal_states_2_001 = [['CA', 'CT'],
                        ['CA', 'CT'],
                        ['CA', 'CT'],
                        ['CA', 'CT'],
                        ['CA', 'CT'],
                        ['CA', 'CT']]


optimal_states_3_inf = [['MA', 'MD', 'VA'],
                        ['CT', 'MD', 'NJ'],
                        ['CA', 'MD', 'MA']]


#DP --> epsilon = 1
optimal_states_3_1 = [['MA', 'MD', 'VA'],
                       ['CT', 'MD', 'NJ']]
#DP --> epsilon = 0.05
optimal_states_3_05 = [['CT', 'MD', 'NJ'],
                       ['MA', 'MD', 'VA'],
                       ['CT', 'MD', 'NJ'],
                       ['CT', 'MA', 'NJ']]

#DP --> epsilon = 0.1
optimal_states_3_01 = [['MA', 'MD', 'VA'],
                       ['MA', 'MD', 'NJ']]

#DP --> epsilon = 0.01
optimal_states_3_001 = [['MA', 'NJ', 'WA'],
                        ['MA', 'NJ', 'WA'],
                        ['MA', 'NJ', 'WA']]

#No differential privacy

optimal_states_5_inf_2  = [['CT', 'MA', 'MD', 'NH', 'RI'],
                           ['AK', 'CT', 'MA', 'MD', 'RI'],
                           ['AK', 'CT', 'MA', 'MD', 'RI'],
                           ['CT', 'DE', 'MD', 'RI', 'VA'],
                           ['CT', 'DE', 'MA', 'MD', 'RI'],
                           ['CT', 'MD', 'RI', 'VA', 'VT'],
                           ['AK', 'CT', 'MA', 'MD', 'RI']]


optimal_states_5_inf = [['CT', 'DE', 'MA', 'MD', 'RI'],
                        ['AK', 'CT', 'MA', 'MD', 'RI'],
                        ['CT', 'DE', 'MA', 'MD', 'RI'],
                        ['CT', 'DE', 'MA', 'MD', 'VT'],
                        ['CT', 'MA', 'MD', 'RI', 'VT']]

#DP --> epsilon = 1
optimal_states_5_1 = [['AK', 'CT', 'MA', 'MD', 'RI'],
                      ['CT', 'DE', 'MA', 'MD', 'WY'],
                      ['CT', 'HI', 'MD', 'NJ', 'RI'],
                      ['CT', 'MA', 'MD', 'NH', 'UT'],
                      ['CT', 'HI', 'MA', 'MD', 'RI']]


#DP --> epsilon = 0.05
optimal_states_5_05 = [['CT', 'HI', 'MA', 'MD', 'VA'],
                       ['CT', 'MA', 'MD', 'UT', 'VA'],
                       ['CT', 'HI', 'MD', 'UT', 'VA'],
                       ['CO', 'CT', 'MD', 'UT', 'VA'],
                       ['CT', 'HI', 'MA', 'MD', 'SD'],
                       ['CT', 'MA', 'MD', 'SD', 'WY'],
                       ['CT', 'DE', 'MA', 'MD', 'SD'],
                       ['CT', 'MA', 'MD', 'RI', 'WY'],
                       ['CT', 'MA', 'MD', 'NV', 'SD'],
                       ['CT', 'MA', 'MD', 'NJ', 'SD'],
                       ['CT', 'MA', 'MD', 'SD', 'WY']]


#DP ---> epsilon = 0.1
optimal_states_5_01 = [['AK', 'CT', 'MA', 'MD', 'RI'],
                       ['CT', 'MA', 'MD', 'RI', 'VT'],
                       ['CT', 'MD', 'NM', 'RI', 'VA'],
                       ['CT', 'DE', 'MD', 'RI', 'VA'],
                       ['CT', 'MD', 'RI', 'VA', 'WY']]



#DP ---> epsilon = inf
optimal_states_10_inf = [['CT', 'MT', 'ND', 'NE', 'NH', 'OR', 'RI', 'SC', 'VA', 'VT'],
                        ['CO', 'CT', 'DE', 'HI', 'ID', 'MD', 'ND', 'RI', 'UT', 'WA'],
                        ['AL', 'CA', 'HI', 'MD', 'NH', 'NJ', 'NV', 'OK', 'VA', 'VT'],
                        ['CA', 'CT', 'DE', 'MD', 'ME', 'NV', 'RI', 'SC', 'UT', 'VA'],
                        ['AR', 'KS', 'MD', 'ND', 'NM', 'RI', 'UT', 'VA', 'WA', 'WV'],
                        ['AK', 'CA', 'DE', 'HI', 'MD', 'NJ', 'OR', 'SD', 'VA', 'VT']]

#DP ---> epsilon = 0.1
optimal_states_10_01 = [['AL', 'CT', 'DE', 'KS', 'MD', 'NH', 'UT', 'VA', 'VT', 'WY'],
                        ['AK', 'AR', 'CT', 'MA', 'MD', 'NM', 'NV', 'OH', 'UT', 'VT'],
                        ['AK', 'AR', 'CT', 'MA', 'MD', 'NH', 'NJ', 'NM', 'TN', 'WY'],
                        ['AK', 'CA', 'CT', 'DE', 'MA', 'MD', 'NV', 'OK', 'SD', 'WY'],
                        ['CA', 'HI', 'LA', 'MA', 'MD', 'MN', 'NE', 'NJ', 'NM', 'SC'],
                        ['AK', 'CT', 'DE', 'MD', 'MN', 'ND', 'NJ', 'OK', 'PA', 'VA']]

#DP ---> epsilon = 0.05
optimal_states_10_05 = [['AK', 'CT', 'MA', 'MD', 'NJ', 'RI', 'SC', 'VA', 'VT', 'WY'],
                        ['AK', 'CT', 'MA', 'MD', 'NH', 'NJ', 'SC', 'VA', 'VT', 'WA'],
                        ['AK', 'CA', 'CO', 'DE', 'MD', 'NH', 'NJ', 'VA', 'VT', 'WY'],
                        ['AK', 'CT', 'HI', 'MA', 'MD', 'NJ', 'RI', 'VA', 'VT', 'WA'],
                        ['AK', 'CT', 'MA', 'MD', 'NH', 'NJ', 'NV', 'UT', 'VA', 'WA'],
                        ['AK', 'CT', 'MA', 'MD', 'NH', 'NJ', 'RI', 'VA', 'VT', 'WY'],
                        ['AK', 'CT', 'DE', 'MA', 'MD', 'NE', 'NJ', 'VA', 'VT', 'WA'],
                        ['AK', 'CT', 'DE', 'HI', 'MA', 'MD', 'NJ', 'UT', 'VA', 'WY'],
                        ['AK', 'CT', 'DE', 'HI', 'MA', 'MD', 'NJ', 'SC', 'VA', 'VT'],
                        ['AK', 'CT', 'HI', 'MA', 'MD', 'NH', 'NJ', 'NM', 'VA', 'VT']]




###################################These states are sufficiently representative of the 50##################################


#epsilon=0.05
minimal_optimal_states = [['WA', 'RI', 'FL', 'ME', 'NE', 'KS', 'IL', 'MA'],
                         ['MI', 'IL', 'AZ', 'WY', 'MO', 'IA', 'FL', 'MD', 'SD', 'TX', 'WA']]


vars_to_bin = ['AGEP']
binned_feature_name = [f'{var_name}_BINNED' for var_name in vars_to_bin]
MI_VARS_TO_USE = ['SEX', 'RAC1P'] + binned_feature_name
print(MI_VARS_TO_USE)
