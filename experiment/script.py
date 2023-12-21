import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import statsmodels.tsa.stattools as smt

import statsmodels.graphics.tsaplots as smp

import function2 as rf


"""
POINT 1: DATA DOWNLOAD AND MANIPULATION
"""

"""
Monthly
"""

df_price_monthly = pd.read_excel('Monthly.xlsx')
time_m = df_price_monthly.iloc[:,0]

df_price_monthly = df_price_monthly.set_index("Exchange Date")

"""
Getting the log-levels of the prices of the equities
"""

df_logprice_monthly = pd.DataFrame(data = np.array(np.log(df_price_monthly)),
                                           columns = df_price_monthly.columns)


"""
Creating the daily log returns for the equities and the equity index
"""

df_ret_monthly = pd.DataFrame(data = np.array(100*(df_logprice_monthly - 
                                            df_logprice_monthly . shift(1))),
                              columns = df_logprice_monthly.columns)


df_ret_monthly = df_ret_monthly.iloc[1:,:]
df_ret_monthly['Date'] = time_m[1:]
df_ret_monthly = df_ret_monthly.set_index("Date")

#Adding time again

df_logprice_monthly['Date'] = time_m
df_logprice_monthly = df_logprice_monthly.set_index("Date")

"""
Daily
"""

df_price_daily = pd.read_excel('Daily.xlsx')
time_d = df_price_daily.iloc[:,0]

df_price_daily = df_price_daily.set_index("Exchange Date")

"""
Getting the log-levels of the prices of the equities
"""

df_logprice_daily = pd.DataFrame(data = np.array(np.log(df_price_daily)),
                                           columns = df_price_monthly.columns)


"""
Creating the daily log returns for the equities and the equity index
"""

df_ret_daily = pd.DataFrame(data = np.array(100*(df_logprice_daily - 
                                            df_logprice_daily . shift(1))),
                              columns = df_logprice_daily.columns)


df_ret_daily = df_ret_daily.iloc[1:,:]
df_ret_daily['Date'] = time_d[1:]

df_ret_daily = df_ret_daily.set_index("Date")


#Adding time again

df_logprice_daily['Date'] = time_d
df_logprice_daily = df_logprice_daily.set_index("Date")


"""
ECONOMIC INDICATORS
"""

df_eco = pd.read_excel("Eco.xlsx", sheet_name = "Sheet2")



#Setting the index and reordering from oldest to newest date

df_eco = df_eco.set_index("Exchange Date")
df_eco = df_eco.sort_index()


"""
-------------------------------------------------------------------------------
POINT 2: PRELIMINARY ANALYSIS OF THE DATA
------------------------------------------------------------------------------
"""

"""
PLOTS
"""

"""
Plots of the daily log prices of the equities
"""

    
rf.graph(df_logprice_daily,"log prices","blue","Daily")
    
"""
Plots of the monthly log prices of the equities
"""
    
rf.graph(df_logprice_monthly,"log prices","red","Monthly")    

"""
Plots for the economic indicators
"""

rf.graph(df_eco,"log prices","green","Monthly") 

"""
CORRELATION FUNCTIONS
"""

rf.correlation_graphs(df_logprice_daily, 20, "Daily")    

rf.correlation_graphs(df_logprice_monthly, 20, "Monthly")  

#Problem with the series being of different lengths 
df_eco = df_eco.iloc[:-1,:]
    
rf.correlation_graphs(df_eco, 20, "Monthly")  
    
"""
-------------------------------------------------------------------------------
POINT 3: ADF TEST
------------------------------------------------------------------------------
"""

"""
Removing the last two years from monthly series
"""

df_logprice_monthly = df_logprice_monthly.iloc[:-24,:]

df_eco = df_eco.iloc[:-24,:]

"""
Removing the last six months from daily series
"""
"""
Since the number of days in which the market is open varies from year to year
because of festivities and other factors and we have ten years of data and we 
have to remove the last six months, we take the last 5% out
"""

n = int(np.round(df_logprice_daily.shape[0] * 0.05) )

df_logprice_daily = df_logprice_daily.iloc[:-n,:]

"""
ADF TEST
"""

"""
Daily 
"""

adf_daily_df = rf.adf_test(df_logprice_daily)

"""
Monthly
"""

adf_monthly_df = rf.adf_test(df_logprice_monthly)

"""
Economic indicators
"""

adf_eco = rf.adf_test(df_eco)

"""
Taking the first difference
"""

"""
As we already computed the log returns for the equities' series, we remove agian 
the last two years for the monthly series and the last six months for the 
daily series
"""
df_ret_monthly = df_ret_monthly.iloc[:-24,:]

n = int(np.round(df_ret_daily.shape[0] * 0.05) )

df_ret_daily = df_ret_daily.iloc[:-n,:]

"""
We re-evaluate the ADF test for the first differences of these series
"""

adf_ret_monthly = rf.adf_test(df_ret_monthly)

adf_ret_daily = rf.adf_test(df_ret_daily)

"""
Taking the log-first difference of the economic indicators' series
"""

df_ret_eco = pd.DataFrame(data = np.array(100*(df_eco - 
                                            df_eco . shift(1))),
                              columns = df_eco.columns)


df_ret_eco = df_ret_eco.iloc[1:,:]
df_ret_eco['Date'] = time_m[1:]
df_ret_eco = df_ret_eco.set_index("Date")

"""
Evaluate the ADF test for the log-first difference of the economic indicators'
series
"""

adf_ret_eco = rf.adf_test(df_ret_eco)

    
"""
Befora and after comparison between the histograms of:
    1) log levels
    2) log returns
"""


rf.hist_comparison(df_logprice_daily, df_ret_daily, var1 = "log prices", var2 = "log returns", 
                   freq = "Daily", nbn = 1000)

rf.hist_comparison(df_logprice_monthly, df_ret_monthly, 
                   var1 = 'log prices', var2 = 'log returns',
                   freq = 'Monthly', nbn = 90)

rf.hist_comparison(df_eco, df_ret_eco, 
                   var1 = 'log prices', var2 = 'log returns',
                   freq = 'Monthly', nbn = 90)



"""
-------------------------------------------------------------------------------
POINT 4: ARMA MODELS
------------------------------------------------------------------------------
"""


