import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import statsmodels.tsa.stattools as smt

import statsmodels.tsa.arima.model as tsa

import statsmodels.graphics.tsaplots as smp

import statsmodels.stats.diagnostic as smd

import function5 as rf

import pdb


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
df_price_daily.columns = df_price_monthly.columns

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
Plots of the daily: 
    1) prices of the equities
    2) log-prices of the equities
    3) ACF
    4) PACF
"""

    
rf.graph(df_logprice_daily,df_price_daily,"log prices","blue","Daily", 20)
    
#rf.correlation_graphs(df_logprice_daily, 20, "Daily") 
    
"""
Plots of the monthly: 
    1) prices of the equities
    2) log-prices of the equities
    3) ACF
    4) PACF
"""
    
rf.graph(df_logprice_monthly,df_price_monthly,"log prices","red","Monthly", 20)    

"""
Plots for the economic indicators of the....
"""

df_eco = df_eco.iloc[:-1,:]
rf.graph_eco(df_eco,"log prices","green","Monthly", 20) 


"""
CORRELATION FUNCTIONS
"""
   

#rf.correlation_graphs(df_logprice_monthly, 20, "Monthly")  

#Problem with the series being of different lengths 

    
#rf.correlation_graphs(df_eco, 20, "Monthly")  

"""
STATISTICS TABLE
"""

table_monthly = rf.descriptive_table(df_logprice_monthly)

table_daily = rf.descriptive_table(df_logprice_daily)

table_eco = rf.descriptive_table(df_eco)



"""
-------------------------------------------------------------------------------
POINT 3: ADF TEST
------------------------------------------------------------------------------
"""

"""
Removing the last two years from monthly series
"""

df_logprice_monthly_cut = df_logprice_monthly.iloc[:-24,:]

df_eco_cut = df_eco.iloc[:-24,:]

"""
Removing the last six months from daily series
"""
"""
Since the number of days in which the market is open varies from year to year
because of festivities and other factors and we have ten years of data and we 
have to remove the last six months, we take the last 5% out
"""

n = int(np.round(df_logprice_daily.shape[0] * 0.05) )

df_logprice_daily_cut = df_logprice_daily.iloc[:-n,:]

"""
ADF TEST
"""

"""
Daily 
"""

adf_daily_df = rf.adf_test(df_logprice_daily_cut)

rf.plotbar(adf_daily_df['pval'],"H", obj = 'p-value of the ADF test\nDaily Log Prices')

non_stationary_daily = rf.non_stationarity_check(adf_daily_df)

stationary_daily = rf.stationary_check(df_logprice_daily_cut, non_stationary_daily)


"""
Monthly
"""

adf_monthly_df = rf.adf_test(df_logprice_monthly_cut)

rf.plotbar(adf_monthly_df['pval'],"H", obj = 'p-value of the ADF test\nMonthly Log Prices')

non_stationary_monthly = rf.non_stationarity_check(adf_monthly_df)

stationary_monthly = rf.stationary_check(df_logprice_monthly_cut, non_stationary_monthly)

"""
Economic indicators
"""

adf_eco_cut = rf.adf_test(df_eco_cut)

rf.plotbar(adf_eco_cut['pval'],"H", obj = 'p-value of the ADF test\nMonthly Log Prices')

non_stationarity_eco = rf.non_stationarity_check(adf_eco_cut)

stationary_eco = rf.stationary_check(df_eco_cut, non_stationarity_eco)


"""
Taking the first difference
"""

"""
As we already computed the log returns for the equities' series, we remove agian 
the last two years for the monthly series and the last six months for the 
daily series.
We then take only the series that we found to be non-stationary.
"""
df_ret_monthly_cut = df_ret_monthly.iloc[:-24,:]

df_ret_monthly_cut = df_ret_monthly_cut.loc[:,non_stationary_monthly]

for i in stationary_monthly:
    
    df_ret_monthly_cut[i] = df_logprice_monthly.iloc[:-24,:][i]




n_d = int(np.round(df_ret_daily.shape[0] * 0.05) )

df_ret_daily_cut = df_ret_daily.iloc[:-n_d,:]

df_ret_daily_cut = df_ret_daily_cut.loc[:,non_stationary_daily]

for i in stationary_daily:
    
    df_ret_daily_cut[i] = df_logprice_daily.iloc[:-n_d,:][i]

            


"""
We re-evaluate the ADF test for the first differences of these series
"""

adf_ret_monthly_cut = rf.adf_test(df_ret_monthly_cut)
rf.plotbar(adf_ret_monthly_cut ['pval'],"H", obj = 'p-value of the ADF test\nMonthly Log Returns')


adf_ret_daily_cut = rf.adf_test(df_ret_daily_cut)
rf.plotbar(adf_ret_daily_cut ['pval'],"H", obj = 'p-value of the ADF test\nDaily Log Returns')


"""
Taking the log-first difference of the economic indicators' series
"""

df_ret_eco = pd.DataFrame(data = np.array(100*(np.log(df_eco) - 
                                            np.log(df_eco . shift(1)) )),
                              columns = df_eco_cut.columns)


df_ret_eco = df_ret_eco.iloc[1:,:]
df_ret_eco['Date'] = time_m[1:]
df_ret_eco = df_ret_eco.set_index("Date")

df_ret_eco_cut = df_ret_eco.iloc[:-24,:]

"""
Takes only the returns for the series found to be non-stationary
"""

df_ret_eco_cut = df_ret_eco_cut.loc[:, non_stationarity_eco]

"""
Adds the series that have been found to be stationary
"""

for i in stationary_eco:
    
    df_ret_eco_cut[i] = df_eco_cut[i]

"""
Evaluate the ADF test for the log-first difference of the economic indicators'
series
"""

adf_ret_eco_cut = rf.adf_test(df_ret_eco_cut)

rf.plotbar(adf_ret_eco_cut ['pval'],"H", obj = 'p-value of the ADF test\nMonthly Log Returns')


    
"""
Before and after comparison between the histograms of:
    1) log levels
    2) log returns
"""


rf.hist_comparison(df_logprice_daily_cut, df_ret_daily_cut, var1 = "log prices", var2 = "log returns", 
                   freq = "Daily", nbn = 1000)

rf.hist_comparison(df_logprice_monthly_cut, df_ret_monthly_cut, 
                   var1 = 'log prices', var2 = 'log returns',
                   freq = 'Monthly', nbn = 90)

rf.hist_comparison(df_eco_cut, df_ret_eco_cut, 
                   var1 = 'log prices', var2 = 'log returns',
                   freq = 'Monthly', nbn = 90)



"""
-------------------------------------------------------------------------------
POINT 4: ARMA MODELS
------------------------------------------------------------------------------
"""

"""
Plotting the PACF and ACF of the series and computing the best ARMA model
"""

"""
Monthly series for the equities
"""

arma_monthly = rf.arma_sorter(df_ret_monthly_cut, "Monthly", 
                              maxlag = 5,
                              criterion = "BIC")

"""
Daily series for the equities
"""

arma_daily = rf.arma_sorter(df_ret_daily_cut, "Daily", 
                              maxlag = 5,
                              criterion = "BIC")

"""
Monthly series for the economic indicators
"""

arma_eco = rf.arma_sorter(df_ret_eco_cut, "Monthly", 
                              maxlag = 5,
                              criterion = "BIC")


"""
Plotting the residuals
"""

rf.resid_graph(arma_monthly, "Residuals" ,"Blue", "Monthly", nlg = 20)

rf.resid_graph(arma_daily, "Residuals" ,"Blue", "Daily", nlg = 20)

rf.resid_graph(arma_eco, "Residuals" ,"Blue", "Monthly", nlg = 20)

"""
Getting the parameters for the various models and their t-statistics
"""


param_monthly = rf.get_params(arma_monthly)

param_daily = rf.get_params(arma_daily)

param_eco = rf.get_params(arma_eco)

"""
In a single table form 
"""

param_try_monthly = rf.get_params2(arma_monthly)
param_try_daily = rf.get_params2(arma_daily)
param_try_eco = rf.get_params2(arma_eco)

"""
Plotting the pvalues of the parameters
"""

rf.plot_pval_params(param_monthly, "Monthly")
rf.plot_pval_params(param_daily, "Daily")      
rf.plot_pval_params(param_eco, "Monthly")  

"""
Ljung-box test on the residuals 
"""


lb_monthly = rf.ljung_box_test(arma_monthly)

lb_daily = rf.ljung_box_test(arma_daily)

lb_eco = rf.ljung_box_test(arma_eco)


"""
Plotting the p-values of the Ljung-Box test on the residuals
"""

rf.plot_lb(lb_monthly, "Monthly")
rf.plot_lb(lb_daily, "Daily")
rf.plot_lb(lb_eco, "Monthly")

"""
Handling the ones for which there was still structure in their residuals by 
using the AIC instead of the BIC as the information criterion to choose among
the possible combinations of the AR and MA parameters
"""

imp1_d_res, imp1_d_lb = rf.improvement_1(lb_daily, df_ret_daily_cut, "Daily")

imp1_m_res, imp1_m_lb = rf.improvement_1(lb_monthly, df_ret_monthly_cut, "Monthly")

imp1_e_res, imp1_e_lb = rf.improvement_1(lb_eco, df_ret_eco_cut, "Monthly")
#%%
check_d = rf.structure_check(imp1_d_lb)

lcol = ["AR", "MA","BIC","AIC", "Model"]
result_df = pd.DataFrame(index = check_d[:-1], columns = lcol)

df_2 = pd.DataFrame(columns = lcol)
freq_2 = None
df = df_ret_daily_cut
criterion = 'AIC'
maxlag = 3

exog1 = df["NORWAY OBX INDEX"].shift(1)
exog1 = exog1.dropna()
exog1.name = "NORWAY OBX INDEX.L1"

if check_d:
    for k in check_d[:-1]:
        for i in range(maxlag + 1):
            for j in range(maxlag + 1):
        
                
                    
                    mod = tsa.ARIMA(df[k].iloc[1:].to_frame(), order= (i,0,j),
                                    exog = exog1,
                                    freq = freq_2)
                    
                    res = mod.fit()
                    l = []
                    
                    spec = res.model_orders
                    
                    l.append(spec["ar"])
                    l.append(spec["ma"])
                    l.append(res.bic)
                    l.append(res.aic)
                    l.append(res)
                    
                    df_2.loc[len(df_2.index)] = l
            
    
        df_2 = df_2.sort_values(criterion)
    
    
        result_df.loc[k, :] = df_2.iloc[0,:]            
        
        
imp2_d_lb =  rf.ljung_box_test(result_df)
rf.plot_lb(imp2_d_lb , "Daily")

#%%

"""
Substituting as the new best ARMA model if the number of lags which have a
p-value smaller than 0.05 in the new model is itself smaller than the one 
obtained in the previous case.
"""
#pdb.set_trace()
for j in imp1_d_lb.index:
    
    new = [k for k in imp1_d_lb.loc[j,:] if k < 0.05]
    old = [u for u in lb_daily.loc[j,:] if u < 0.05]
    
    if len(old) > len(new):
        
        arma_daily.loc[j,:] = imp1_d_res.loc[j,:]
    

#Function 



#%%
"""
-------------------------------------------------------------------------------
POINT 5: FORECASTS
-------------------------------------------------------------------------------
"""
name = "AKER"

#foreca = arma_monthly.loc[name, 'Model']. forecast ( steps = 24)

df = df_ret_daily['Name']

ar = arma_daily.loc[name, "AR"]
ma = arma_daily.loc[name, "MA"]

len_param = len(df_ret_daily_cut.index)

forecasts = []

for i in range(24):
    

    
    mod = tsa.ARIMA(df.iloc[len_param +i].to_frame(), order= (i,0,j), freq = freq_2)
    
    res = mod.fit()
    
    f = res. forecast ( steps = 1)
    
    forecasts.append(f)
    

    
    


#%%
# example dynamic vs static forecast
# simulate AR (1)

import statsmodels . api as sm
phi1 = 0.8
ma= [1]
ar=[1 , - phi1 ]
x=sm . tsa . arma_generate_sample ( ar ,ma , nsample =230 , burnin = 100)
# estimate - leave last 30 for forecast
mod1 =sm . tsa . ARIMA ( x [0:199] , order =(1 ,0 ,0) , trend ='n')
out1 = mod1 . fit ()
# dynamic forecast
for1 = out1 . forecast ( steps = 30)
# static forecast
for2 =np . zeros ([30 ,1])
for2 [0 ,0]= out1 . forecast ( steps =1)
se1 =np . zeros ([30 ,1])
resvar = out1 . sse / np . size ( out1 . resid )
se1 [0 ,0]=np . sqrt ( resvar )
for ii in range (1 ,30) :
    m= 199+ ii
    mod1 =sm . tsa . ARIMA ( x [0: m], order =(1 ,0 ,0) , trend ='n')
    out1 = mod1 . fit ()
    for2 [ii ,0] = out1 . forecast ( steps =1)
    resvar = out1 . sse / np . size ( out1 . resid )
    se1 [ii ,0] =np . sqrt ( resvar )

