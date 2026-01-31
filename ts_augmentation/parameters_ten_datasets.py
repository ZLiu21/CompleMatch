time_aug_001labeled = {'HAR': {'aug_time_method_index': 3, 'aug2_time_method_index': 6, 'dropout': 0.05, 'self_contra_tau': 100, 'lr': 0.005},
                      'Epilepsy': {'aug_time_method_index': 1, 'aug2_time_method_index': 4, 'dropout': 0.05, 'self_contra_tau': 5, 'lr': 0.0003},
                      'SleepEDF': {'aug_time_method_index': 3, 'aug2_time_method_index': 4, 'dropout': 0.20, 'self_contra_tau': 50, 'lr': 0.01},
                      'Wafer': {'aug_time_method_index': 1, 'aug2_time_method_index': 4, 'dropout': 0.20, 'self_contra_tau': 0.1, 'lr': 0.0003},
                      'FordA': {'aug_time_method_index': 5, 'aug2_time_method_index': 6, 'dropout': 0.25, 'self_contra_tau': 1.0, 'lr': 0.002},
                      'FordB': {'aug_time_method_index': 3, 'aug2_time_method_index': 4, 'dropout': 0.45, 'self_contra_tau': 1.0, 'lr': 0.005},
                      'PhalangesOutlinesCorrect': {'aug_time_method_index': 3, 'aug2_time_method_index': 3, 'dropout': 0.15, 'self_contra_tau': 5.0, 'lr': 0.0003},
                      'ProximalPhalanxOutlineCorrect': {'aug_time_method_index': 1, 'aug2_time_method_index': 3, 'dropout': 0.10, 'self_contra_tau': 5.0, 'lr': 0.0003},
                      'StarLightCurves': {'aug_time_method_index': 3, 'aug2_time_method_index': 4, 'dropout': 0.10, 'self_contra_tau': 500, 'lr': 0.0003},
                      'ElectricDevices': {'aug_time_method_index': 1, 'aug2_time_method_index': 2, 'dropout': 0.40, 'self_contra_tau': 0.005, 'lr': 0.0003}
} 

time_aug_005labeled = {'HAR': {'aug_time_method_index': 5, 'aug2_time_method_index': 5, 'dropout': 0.10, 'self_contra_tau': 50, 'lr': 0.005},
                      'Epilepsy': {'aug_time_method_index': 1, 'aug2_time_method_index': 3, 'dropout': 0.01, 'self_contra_tau': 500, 'lr': 0.0003},
                      'SleepEDF': {'aug_time_method_index': 4, 'aug2_time_method_index': 6, 'dropout': 0.35, 'self_contra_tau': 20, 'lr': 0.05},
                      
                      'Wafer': {'aug_time_method_index': 3, 'aug2_time_method_index': 6, 'dropout': 0.10, 'self_contra_tau': 20, 'lr': 0.0003},
                      'FordA': {'aug_time_method_index': 2, 'aug2_time_method_index': 3, 'dropout': 0.35, 'self_contra_tau': 0.5, 'lr': 0.0003},
                      'FordB': {'aug_time_method_index': 6, 'aug2_time_method_index': 6, 'dropout': 0.45, 'self_contra_tau': 0.1, 'lr': 0.002},
                      'PhalangesOutlinesCorrect': {'aug_time_method_index': 2, 'aug2_time_method_index': 2, 'dropout': 0.01, 'self_contra_tau': 20, 'lr': 0.0003},
                      'ProximalPhalanxOutlineCorrect': {'aug_time_method_index': 1, 'aug2_time_method_index': 3, 'dropout': 0.01, 'self_contra_tau': 200, 'lr': 0.0003},
                      'StarLightCurves': {'aug_time_method_index': 1, 'aug2_time_method_index': 2, 'dropout': 0.01, 'self_contra_tau': 0.1, 'lr': 0.0003},
                      'ElectricDevices': {'aug_time_method_index': 3, 'aug2_time_method_index': 4, 'dropout': 0.15, 'self_contra_tau': 0.1, 'lr': 0.0003}
} 

freq_aug_001labeled = {'HAR': {'aug_freq_method_index': 3, 'aug2_freq_method_index': 3},
                      'Epilepsy': {'aug_freq_method_index': 1, 'aug2_freq_method_index': 2},
                      'SleepEDF': {'aug_freq_method_index': 2, 'aug2_freq_method_index': 2},
                      
                     'Wafer': {'aug_freq_method_index': 2, 'aug2_freq_method_index': 4},
                      'FordA': {'aug_freq_method_index': 2, 'aug2_freq_method_index': 2},
                      'FordB': {'aug_freq_method_index': 3, 'aug2_freq_method_index': 4},
                      'PhalangesOutlinesCorrect': {'aug_freq_method_index': 1, 'aug2_freq_method_index': 4},
                      'ProximalPhalanxOutlineCorrect': {'aug_freq_method_index': 3, 'aug2_freq_method_index': 4},
                      'StarLightCurves': {'aug_freq_method_index': 1, 'aug2_freq_method_index': 4},
                      'ElectricDevices': {'aug_freq_method_index': 1, 'aug2_freq_method_index': 1}
}  

freq_aug_005labeled = {'HAR': {'aug_freq_method_index': 2, 'aug2_freq_method_index': 2},
                      'Epilepsy': {'aug_freq_method_index': 3, 'aug2_freq_method_index': 3},
                      'SleepEDF': {'aug_freq_method_index': 2, 'aug2_freq_method_index': 3},
                      
                      'Wafer': {'aug_freq_method_index': 1, 'aug2_freq_method_index': 1},
                      'FordA': {'aug_freq_method_index': 1, 'aug2_freq_method_index': 4},
                      'FordB': {'aug_freq_method_index': 1, 'aug2_freq_method_index': 3},
                      'PhalangesOutlinesCorrect': {'aug_freq_method_index': 1, 'aug2_freq_method_index': 1},
                      'ProximalPhalanxOutlineCorrect': {'aug_freq_method_index': 1, 'aug2_freq_method_index': 2},
                      'StarLightCurves': {'aug_freq_method_index': 2, 'aug2_freq_method_index': 3},
                      'ElectricDevices': {'aug_freq_method_index': 2, 'aug2_freq_method_index': 3}
} 