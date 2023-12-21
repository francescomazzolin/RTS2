import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import statsmodels.tsa.stattools as smt

import statsmodels.graphics.tsaplots as smp

"""
POINT 1: DATA DOWNLOAD AND MANIPULATION
"""

"""
Monthly
"""

df_price_monthly = pd.read_excel('Monthly.xlsx')
time_m = df_price_monthly.iloc[:,0]

#df_price_monthly = df_price_monthly.set_index("Exchange Date")

"""
Getting the log-levels of the prices of the equities
"""
#df_price_monthly = pd.DataFrame(data = np.array(df_price_monthly.iloc[:,1:]),
                                           #columns = df_price_monthly.columns[1:])
df_logprice_monthly = pd.DataFrame(data = np.array(np.log(df_price_monthly.iloc[:,1:])),
                                           columns = df_price_monthly.columns[1:])


"""
Creating the daily log returns for the equities and the equity index
"""

df_ret_monthly = pd.DataFrame(data = np.array(100*(df_logprice_monthly - 
                                            df_logprice_monthly . shift(1))),
                              columns = df_logprice_monthly.columns)


df_ret_monthly = df_ret_monthly.iloc[1:,:]
df_ret_monthly['Date'] = time_m[1:]
#df_ret_monthly = df_ret_monthly.set_index("Date")

#Adding time again

df_logprice_monthly['Date'] = time_m
#df_logprice_monthly = df_logprice_monthly.set_index("Date")

"""
Daily
"""

df_price_daily = pd.read_excel('Daily.xlsx')
time_d = df_price_daily.iloc[:,0]

#df_price_daily = df_price_daily.set_index("Exchange Date")

"""
Getting the log-levels of the prices of the equities
"""
#df_price_daily = pd.DataFrame(data = np.array(df_price_daily.iloc[:,1:]),
                                           #columns = df_price_monthly.columns[:])
df_logprice_daily = pd.DataFrame(data = np.array(np.log(df_price_daily.iloc[:,1:])),
                                           columns = df_price_monthly.columns[1:])


"""
Creating the daily log returns for the equities and the equity index
"""

df_ret_daily = pd.DataFrame(data = np.array(100*(df_logprice_daily - 
                                            df_logprice_daily . shift(1))),
                              columns = df_logprice_daily.columns)


df_ret_daily = df_ret_daily.iloc[1:,:]
df_ret_daily['Date'] = time_d[1:]

#df_ret_daily = df_ret_daily.set_index("Date")


#Adding time again

df_logprice_daily['Date'] = time_d
#df_logprice_daily = df_logprice_daily.set_index("Date")


"""
ECONOMIC INDICATORS
"""

df_eco = pd.read_excel("Eco.xlsx", sheet_name = "Sheet2")



#Setting the index and reordering from oldest to newest date

df_eco = df_eco.set_index("Exchange Date")

#df_eco = df_eco.sort_index()

"""
-------------------------------------------------------------------------------
POINT 2
------------------------------------------------------------------------------
"""

"""
PLOTS
"""
def graph(x,y,var,col,freq):
    plt.plot(x,y,color=f"{col}")
    plt.title(f"{i} {var} - {freq} frequency")
    plt.show()
"""
Plots of the daily log prices of the equities
"""
for i in df_logprice_daily.columns[:-1]:
    #graph(time_d, df_price_daily[i],"prices","blue","Daily")
    graph(time_d, df_logprice_daily[i],"log prices","blue","Daily")
    graph(time_d[1:], df_ret_daily[i],"log returns","blue","Daily")
    
    
"""
Plots of the monthly log prices of the equities
"""
for i in df_logprice_monthly.columns[:-1]:
    #graph(time_m, df_price_monthly[i],"prices","red","Monthly")
    graph(time_m, df_logprice_monthly[i],"log prices","red","Monthly")
    graph(time_m[1:], df_ret_monthly[i],"log returns","red","Monthly")
    
"""
Plots for the economic indicators
"""

for i in df_eco.columns:
    
    graph(time_m[1:], df_eco[i],"","green","Monthly")
    
"""
-------------------------------------------------------------------------------
POINT 3
------------------------------------------------------------------------------
"""

"""
Removing the last two years from monthly series
"""

df_logprice_monthly = df_logprice_monthly.iloc[24:,:]

df_eco = df_eco.iloc[24:,:]

"""
Removing the last six months from daily series
"""
"""
Since the number of days in which the market is open varies from year to year
because of festivities and other factors and we have ten years of data and we 
have to remove the last six months, we take the last 5% out
"""

n = int(np.round(df_logprice_daily.shape[0] * 0.05) )

df_logprice_daily = df_logprice_daily.iloc[n:,:]

"""
ADF TEST
"""
    
#TEST

#Creation of dataframe to store results
lcol = ['ADF', 'pval', 'lags', 'num_obs', 'critical_values', 'IC']
adf_df = pd.DataFrame(index = df_logprice_daily.columns[:-1], columns = lcol)

from statsmodels.tsa.stattools import adfuller

for i in df_logprice_daily.columns[:-1]:
    
    adftest = adfuller(df_logprice_daily[i], maxlag =21)      
    adf_df.loc[i,:] = adftest

adf_df    



"""
CORRELATION FUNCTIONS
"""

def correlation_graphs(x,y,z):
    smp.plot_acf(x, lags = y)
    plt.title(f"{i} log prices - {z} (ACF)")
    plt.show()
    
    smp.plot_pacf(x, lags = y)
    plt.title(f"{i} log prices - {z} (PACF)")
    
    plt.show()
    
    

for i in df_logprice_daily.columns[:-1]:
    
    correlation_graphs(df_logprice_daily[i], 20, "Daily")
    
for i in df_logprice_monthly.columns[:-1]:
    
    correlation_graphs(df_logprice_monthly[i], 20, "Monthly")
 
    
#Problem with the series being of different lengths 
df_eco = df_eco.iloc[1:,:]
for i in df_eco.columns:
    
    correlation_graphs(df_eco[i],20,"Monthly")
    
    
