import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import statsmodels.tsa.stattools as smt

import statsmodels.graphics.tsaplots as smp


def adf_test(df):
    
    #Creation of dataframe to store results
    lcol = ['ADF', 'pval', 'lags', 'num_obs', 'critical_values', 'IC']
    adf_df = pd.DataFrame(index = df.columns, columns = lcol)
    
    
    
    for i in df.columns:
    
        x = df[i]
        
        l = smt.adfuller(x, regression='n', autolag='BIC')
        print(l)
        
        adf_df.loc[i,:] = list(l)
    
    return adf_df