import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import statsmodels.tsa.stattools as smt
import statsmodels.tsa.arima.model as tsa
import statsmodels.stats.diagnostic as smd

import statsmodels.graphics.tsaplots as smp
from matplotlib import gridspec


def adf_test(df):
    
    #Creation of dataframe to store results
    lcol = ['ADF', 'pval', 'lags', 'num_obs', 'critical_values', 'IC']
    adf_df = pd.DataFrame(index = df.columns, columns = lcol)
    
    
    
    for i in df.columns:
    
        adftest = smt.adfuller(df[i], maxlag =21)      
        adf_df.loc[i,:] = adftest
    
    return adf_df


def graph(df,df2,var,col,freq,nlg):
    
    for i in df.columns:
        
        fig, ax = plt.subplots(2, 2, figsize=(12, 8))
        
        #plt.subplot(000)
        ax[0,0].plot(df2[i],color=f"{col}")
        #axs[0,0].set_title(f"{i} {var} - {freq} frequency")
        
        ax[0,1].plot(df[i],color=f"{col}")
        #axs[0,1].set_title(f"{i} {var} - {freq} frequency")



        smp.plot_acf(df[i], lags = nlg, ax = ax[1,0])
        #axs[1,0].set_title(f"{i} log prices - {z} (ACF)")
        
        smp.plot_pacf(df[i], lags = nlg, ax = ax[1,1])
        #axs[1,1].set_title(f"{i} log prices - {z} (PACF)")
        
        plt.suptitle(f"Analysis of {i}", fontsize=16)  
        plt.tight_layout()  
        
        plt.show()
        

def graph_eco(df, var, col, freq, nlg):
    
    for i in df.columns:
        
        fig = plt.figure(figsize=(12, 8))
        gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[2, 1])

        # Plot for the first row, first column (big plot)
        ax1 = plt.subplot(gs[0, 0:])
        ax1.plot(df[i], color=col)

        ax1.set_xlabel('Index')
        ax1.set_ylabel(f'{i} values')

        # ACF plot
        ax2 = plt.subplot(gs[1, 0])
        smp.plot_acf(df[i], lags=nlg, ax=ax2)
        ax2.set_title(f'Autocorrelation Function ({i})')
        ax2.set_xlabel('Lags')
        ax2.set_ylabel('ACF')

        # PACF plot
        ax3 = plt.subplot(gs[1, 1])
        smp.plot_pacf(df[i], lags=nlg, ax=ax3)
        ax3.set_title(f'Partial Autocorrelation Function ({i})')
        ax3.set_xlabel('Lags')
        ax3.set_ylabel('PACF')

        # Adjust layout for better spacing
        #plt.tight_layout()

        plt.suptitle(f"Analysis of {i}", fontsize=16)  
        plt.show()
        
def hist_comparison(df_1, df_2, var1, var2, freq, nbn):
    
    
    for i in df_1.columns:
        
        #plt.figure()
        
        plt.hist(df_1[i], bins = nbn, color = "blue")
        plt.title(f"{i} {var1} - {freq} frequency")
        
        plt.show()
        
        
        #plt.figure()
        
        plt.hist(df_2[i], bins = nbn, color = "orange")
        plt.title(f"{i} {var2} - {freq} frequency")
        
        plt.show()
        
"""
REMOVED MEAN
"""
def plotbar(P,SavePath, one_value = 0.01, five_value = 0.05, 
            ten_value = 0.1, obj = ''):
    """/3_p_value_plots/"""
    variable = P.name
    P = pd.DataFrame(data = P, columns = [variable])
    #mean = P.loc['Mean', variable]
    P['stock_names'] = P.index
    
    #removed mean
    def bar_highlight(value, one_value, 
                    five_value,
                    ten_value):
        
        if value <= one_value:
            return 'red'
        elif value <= five_value:
            return 'orange'
        elif value <= ten_value:
            return 'gold'
        
        else:
            return 'grey'
    fig, ax = plt.subplots()   
    
    #REMOVED MEAN FROM THE ARGUMENTS
    P['colors'] = P[variable].apply(bar_highlight, args = (one_value, 
                        five_value,
                        ten_value))
    

    bars = plt.bar(P['stock_names'], P[variable], color=P['colors'])
    x_pos = range(P['stock_names'].shape[0])
    plt.xticks(x_pos, P['stock_names'], rotation=90)
    plt.title(obj)
    
    variable = variable.replace(":","_")
    """
    plt.savefig(folder_definer(SavePath)+"/"+variable+".png")
    if allow_clean:
        plt.show()
    
    plt.close()
    """
    

def plotbar2(P,SavePath, one_value = 0.01, five_value = 0.05, 
            ten_value = 0.1, obj = ''):
    """/3_p_value_plots/"""
    variable = P.columns[0]
    #P = pd.DataFrame(data = P, columns = [variable])
    #mean = P.loc['Mean', variable]
    #P['parameters names'] = P.index
    
    #removed mean
    def bar_highlight(value, one_value, 
                    five_value,
                    ten_value):
        
        if value <= one_value:
            return 'red'
        elif value <= five_value:
            return 'orange'
        elif value <= ten_value:
            return 'gold'
        
        else:
            return 'grey'
    fig, ax = plt.subplots()   
    
    #REMOVED MEAN FROM THE ARGUMENTS
    P['colors'] = P[variable].apply(bar_highlight, args = (one_value, 
                        five_value,
                        ten_value))
    

    bars = plt.bar(P.index, P[variable], color=P['colors'])
    x_pos = range(P.index.shape[0])
    plt.xticks(x_pos, P.index, rotation=90)
    plt.title(obj)
    
    variable = variable.replace(":","_")
    """
    plt.savefig(folder_definer(SavePath)+"/"+variable+".png")
    if allow_clean:
        plt.show()
    
    plt.close()
    """


    

def descriptive_table(df):
    i = df.columns[0]
        
    d = df[i].describe()
    
    lrow = list(d.index)
    lrow.append('Skewness')
    lrow.append('Kurtosis')
        
    descr_df = pd.DataFrame(index = lrow, columns = df.columns)
    
    for i in df.columns:
        
        l = list(df[i].describe())
        l.append(df[i].skew())
        l.append(df[i].kurt())
        
        descr_df.loc[:,i] = l
        
    return descr_df
    


def non_stationarity_check(df):

    stationary_list = []
    
    for i,j in zip(df['pval'], df.index):
        
        if i > 0.05:
            
            stationary_list.append(j)
    
    return stationary_list

def stationary_check(df, stationary_list):
    
    non_stationary = []
    
    for i in df.columns:
               
        if i not in stationary_list:
            
            non_stationary.append(i)
            
    return non_stationary


def graph_4(df,var,col,freq,nlg):
    
    for i in df.columns:
        
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))

        smp.plot_acf(df[i], lags = nlg, ax = ax[0])
                
        smp.plot_pacf(df[i], lags = nlg, ax = ax[1])
        
        plt.suptitle(f"Correlation functions of {i} ({freq})".format(freq), fontsize=16)  
        plt.tight_layout()  
        
        plt.show()
     
"""
This function enables us to:
    1) Plot the ACF and PACF of the series of interest
    2) Find the best possible ARMA model.
    
This is done by estimating all the possible combinations of the AR and MA
components for the given series from 0 to maxlag.

Then the results of each model is stored in a dataframe, along with the 
order of the parameters and the information criterions' values.

We then sort the models using the information criterion choosed by the 
user and save for the given asset the model with the lowest value for the
selected information criterion.
"""
  
def arma_sorter(df, freq_1, maxlag = 2, criterion = "BIC"):
    """
    Creating a dataframe in which we will save the best model for each asset
    or economic indicator
    """
    
    if freq_1 == "Monthly":
        
        freq_2 = "M"
        
    elif freq_1 == "Daily":
        
        freq_2 = None
    
    lcol = ["AR", "MA","BIC","AIC", "Model"]
    
    result_df = pd.DataFrame(index = df.columns, columns = lcol)
     
    for name in df.columns:
        
        """
        Creating a dataframe in which we will store the order of the model,
        the BIC value and the object containing all the results for further use.
        """
        
        df_2 = pd.DataFrame(columns = lcol)
        
        """
        Plotting the correlation functions
        """
        
        graph_4(df[name].to_frame(), "log prices","blue",freq_1, 20)
        
        """
        Estimating each possible combination of the AR and MA parameters from zero to six
        """
        
        for i in range(maxlag + 1):
            for j in range(maxlag + 1):
                
                if i == 0 and j == 0:
                    
                    continue
                
                """
                Estimating the model
                """
                
                mod = tsa.ARIMA(df[name].to_frame(), order= (i,0,j), freq = freq_2)
                
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
        
        
        result_df.loc[name, :] = df_2.iloc[0,:]            
                        
    
    return result_df



def resid_graph(df,var,col,freq,nlg):
    
    for i in df.index:
        
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))

        smp.plot_acf(df.loc[i, "Model"].resid, lags = nlg, ax = ax[0])
                
        smp.plot_pacf(df.loc[i, "Model"].resid, lags = nlg, ax = ax[1])
        
        plt.suptitle(f"Correlation functions of the {var} of {i} ({freq})".format(freq), fontsize=16)  
        plt.tight_layout()  
        
        plt.show()
        
        
        

def get_params(df):
    dict_1 = {}
    
     
    # iterating through the elements of list
    for i in df.index:
        dict_1[i] = None
        
        
    for name in dict_1.keys():   
        print(name)
        results = df.loc[name, 'Model']
        
        results_summary = results.summary()
        
        #dataframe
        results_df = pd.DataFrame({
            'Parameter': results.params.index,
            'Estimate': results.params.values,
            'Standard Error': results.bse.values,
            't Value': results.tvalues.values,
            'P Value': results.pvalues.values})
        
        results_df.set_index("Parameter", inplace = True)
        dict_1[name] = results_df
    
    return dict_1




def get_params2(df):
    
    #Creating a dataframe in which we store the information
    result_df = pd.DataFrame(index = df.index)
    
    
    for name in df.index:
        
        
        #Getting the parameters, std errors and t-statistic values
        
        parameters = df.loc[name, 'Model'].params
        
        stderr = df.loc[name, 'Model'].bse
        
        tstat = df.loc[name, 'Model'].tvalues
        
        pvalue = df.loc[name, 'Model'].pvalues
        
        #Getting the name of the asset and parameters names in order to store
        #The information accordingly
        
        endog_variable = df.loc[name, 'Model'].model.endog_names
        
        param_names = df.loc[name, 'Model'].model.param_names
        
        l = []
        
        
        for i in range(parameters.shape[0]):
            
            #Finding if there are new items
            new_columns = [nwcol for nwcol in param_names if nwcol not in result_df.columns]
            
            #Creating the name for the new columns    
            for nwcol in new_columns:
                
                l.append(nwcol)
                l.append(f"Standard_Error_{nwcol}")
                l.append(f"T-statistic_{nwcol}")
                l.append(f"P_value_{nwcol}")
            
            #Adding the new columns to the dataframe
            
            for col_name in l:
                
                result_df[col_name] = None
                
            
            for m in param_names: 
                
                result_df.loc[name,m] = parameters[m]
                
                str1 = f"Standard_Error_{m}"
                
                result_df.loc[name, str1] = stderr[m]
                
                str2 = f"T-statistic_{m}"
                
                result_df.loc[name, str2] = tstat[m]
                
                str2 = f"P_value_{m}"
                
                result_df.loc[name, str2] = pvalue[m]
                
    return result_df            
       

def plot_pval_params(dic, freq):
    for i in list(dic.keys()):
        
        #to_plot = [j for j in param_monthly[i].columns if "P Value" in j]
        
        plotbar2(dic[i].loc[:,"P Value"].to_frame(),"H", 
                   obj = f'p-value of the parameters of the model for {i} ({freq})')         

def ljung_box_test(df, nlags = 10): 
    
    lcol = list(range(1,nlags+1))
    p_val=pd.DataFrame(index=df.index, columns = lcol)

    for name in df.index:

        x = df.loc[name, 'Model'].resid
        n = list(df.loc[name,:])
        n = n[0] + n[1] 
        result = smd.acorr_ljungbox(x, lags = nlags, model_df= n )
        p_val.loc[name,:]=result['lb_pvalue']
    return p_val


def plot_lb(df, freq):
        
    for i in df.index:
    
        plotbar(df.loc[i,:], "H", obj = f"p-value of the Ljung-Box test\n{freq} returns of {i}")


def structure_check(lb):
     
    structured = []
    
    for i in lb.index:
        
        for j in lb.columns:
            
            if lb.loc[i,j] < 0.05:
                
                structured.append(i)
                break
            
    return structured
    

def improvement_1(lb, data, freq):
    
        
    structured = []
    
    for i in lb.index:
        
        for j in lb.columns:
            
            if lb.loc[i,j] < 0.05:
                
                structured.append(i)
                break
    
    """
    Creating a new dataframe with data for the series for which we still found
    structure among their residuals after our first choice for an ARMA model
    """
    df = pd.DataFrame(index =  data.index, columns = structured)
    
    #df["UNEMPLOYMENT"] = df_ret_eco_cut["UNEMPLOYMENT"]
    
    for i in data[structured]:
        
            
        df[i] = data[i]
    
    """
    Finding the best ARMA model for the series, this time using the AIC
    as the information criterion to choose among the possibilities.
    """
    result_2 = arma_sorter(df, freq, 
                                  maxlag = 5,
                                  criterion = "AIC")
    
    """
    Re-estimating the Ljung-Box test for the residuals coming from this new model
    and plotting the p-values. 
    """
    
    lb_try = ljung_box_test(result_2)
    
    plott = rf.plot_lb(lb_try, f"{freq} (AIC)")
    
    return result_2, lb_try

    
    