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

        
        adf_df.loc[i,:] = list(l)
    
    return adf_df

def graph(df,var,col,freq):
    
    for i in df.columns:
        
        plt.plot(df[i],color=f"{col}")
        plt.title(f"{i} {var} - {freq} frequency")
        plt.show()
            
        
def correlation_graphs(df,nlg,z):
    
    for i in df.columns:
        
        smp.plot_acf(df[i], lags = nlg)
        plt.title(f"{i} log prices - {z} (ACF)")
        plt.show()
        
        smp.plot_pacf(df[i], lags = nlg)
        plt.title(f"{i} log prices - {z} (PACF)")
        
        plt.show()
        

        
def hist_comparison(df_1, df_2, var1, var2, freq):
    
    
    for i in df_1.columns:
        
        #plt.figure()
        
        plt.hist(df_1[i], bins = 1000, color = "blue")
        plt.title(f"{i} {var1} - {freq} frequency")
        
        plt.show()
        
        
        #plt.figure()
        
        plt.hist(df_2[i], bins = 1000, color = "orange")
        plt.title(f"{i} {var2} - {freq} frequency")
        
        plt.show()
        
