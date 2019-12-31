# Section 2.3.2 Local Anomaly detection model results
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#import pandas as pd
#annual_complete=pd.read_csv("C:/Users/WSU-PNNL/Desktop/Data-pec/3967_data_2015_2018_all.csv",header=0,index_col=1,parse_dates=True)

#%% Preparing the Annual data for different datasets
dataset_select=0
dataid=2
if(dataset_select==1):
    # 1.Pecan streeet 3967 Data id   
    hist_start_date='2017-01-02'
    hist_end_date='2017-01-08'
    fore_start_date='2017-01-09'
    fore_end_date='2017-01-15'
    
#    #739 id
#    hist_start_date='2015-01-05'
#    hist_end_date='2015-01-11'
#    fore_start_date='2015-01-12'
#    fore_end_date='2015-01-18'
    
elif(dataset_select==2):
    # 2.REDD dataset
    Annual=pd.read_csv("C:/Users/WSU-PNNL/Desktop/Data-pec/Low freq REDD data/low_freq/house_1/channel_1.dat",delimiter='\s+',header=None,names=['mains power'],index_col=0) 
    
elif(dataset_select==3):
    # 3.AMPD dataset
    Annual=pd.read_csv("C:/Users/WSU-PNNL/Desktop/Data-pec/Electricity_P.csv",header=0,index_col=0) # Annual_data.idxmax() 
    Annual=Annual['MHE']
    Annual=Annual/1000
    
    # AMPD dataset
    hist_start_date='2012-04-07'
    hist_end_date='2012-04-13'
    fore_start_date='2012-04-14'
    fore_end_date='2012-04-20'
    
elif(dataset_select==4):
    Annual=pd.read_csv("C:/Users/WSU-PNNL/Desktop/Data-pec/3967_data_2015_2018.csv",header=0,index_col=0,parse_dates=True,usecols=['local_15min','use']) 
    
if ((dataset_select==3) or(dataset_select==2)):
    Annual.index=pd.to_datetime(Annual.index,unit='s')
    Annual=Annual.resample('0.25H').asfreq()
    Annual=Annual.fillna(0)
resample=0
if(resample==1):
    Annual=Annual.resample('H').asfreq()    
#%% Histogram based detection
# For pecan street trying to find max consumption
#Annual.max()
#max_index=Annual.idxmax()
#correspond_breakup=Annual_data.loc[max_index]
#(n,bins)=plt.hist(Annual)    
    
data_individual_check=1
if(data_individual_check):
#    annual=annual_complete['use']
#    Annual_data=Annual_data[Annual_data['use'] > 5]
    Annual_data=annual_complete #Pre-loaded spydata file
    Annual=annual_complete['use']
    annual_individual=Annual_data.drop(columns=['dataid','use','grid','gen']).fillna(0)
    annual_individual_sum=annual_individual.sum(axis=1)
    
    # Check for first value
    first_use=Annual[0]
    first_breakup=annual_individual.iloc[0]
    first_breakup_sum=first_breakup.sum()
    individual_max=annual_individual.max()
    
    # Histogram of difference between individual and total load consumption
    Individual_diff=(Annual-annual_individual_sum).dropna()
    mask=(Individual_diff>5)
    diff1=Individual_diff[mask]

    
    # First order differencing of actual load data
    Annual_diff=Annual.diff().dropna()
    Gaussian_violations= ((abs(Annual_diff) > 3).astype('int')).to_frame()
    Gauss_violation_idx=Gaussian_violations.index[Gaussian_violations['use']==1]
    Individual_diff_violations=Individual_diff[Gauss_violation_idx]
    Individual_diff_violations_large=Individual_diff_violations[Individual_diff_violations > 1]
    
    # Generating figure for probabilistic detection
    data=Annual_diff['2016-12-25']
    individual_diff1=Individual_diff['2016-12-25']
    data.index=np.arange(0,data.shape[0],1)
    indices=data[abs(data) > 3].index.values
    a=format(data[indices[1]])
    
    #Max difference between AMI and corresponding individual energy consumption
    max_value=Individual_diff.max()
    index_max_value=Individual_diff.idxmax()
    max_diff_annual=Annual.loc[index_max_value]
    max_diff_annual_individual=annual_individual.loc[index_max_value]
    max_diff_annual_individual_sum=annual_individual_sum.loc[index_max_value]
    idx=pd.notna(Annual_data['car1']).astype('int')
    A1=Annual_data.loc[Annual_data.index.intersection(idx)]
    
    #%% Smart meter detection figures
    #Figure 2.3 Individual hist
    plt.figure()
    Annual.plot.hist()
    plt.xlabel('Load real power(kW)')
    plt.title('Histogram of 2017 Pecan 3967 dataid')
    
    #Fig 2.4 R program fit of distribution
    
    #Figure 2.5 Differenced data histogram
    plt.figure()
    Annual_diff.plot.hist()
    plt.xlabel('Load real power change(kW)')
    plt.title('Histogram of first order differenced 2017 Pecan 3967 data')
    
    #Fig 2.6 Generating gaussian based anomalies
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.plot(data,linewidth=1.5,color='grey')
    ax.set_xlabel('Time steps')
    ax.set_ylabel('Gaussian Threshold')
#    ax.annotate('Probability based Anomalies', xy=(indices[0],data[indices[0]]),xytext=(indices[0],4), arrowprops=dict(facecolor='black', shrink=0.05))
    ax.plot(indices[0],data[indices[0]], 'o', ms=14, markerfacecolor="None", markeredgecolor='red', markeredgewidth=3)
    ax.annotate("Violation 1:{:.2f}".format(data[indices[0]]), xy=(indices[0]+3,data[indices[0]]), textcoords="offset points", xytext=(0,10),ha='left', va='top')
    ax.plot(indices[1],data[indices[1]], 'o', ms=14, markerfacecolor="None", markeredgecolor='red', markeredgewidth=3)
    ax.annotate("Violation 2:{:.2f}".format(data[indices[1]]), xy=(indices[1]+3,data[indices[1]]), textcoords="offset points", xytext=(0,10),ha='left', va='top')
    ax.set_xlim([0,100])
    ax.set_ylim([-4,4])
    plt.show()
#   plt.savefig("Gaussian_violation",dpi=600)
#%% Reading new data id's
#import pandas as pd
#Annual_8084=pd.read_csv("C:/Users/WSU-PNNL/Desktop/Data-pec/dataid/8084_data_2016_2018_all.csv",header=0,index_col=0,parse_dates=True,usecols=['local_15min','use'])
#Annual_9971=pd.read_csv("C:/Users/WSU-PNNL/Desktop/Data-pec/dataid/9971_data_2016_2017_all.csv",header=0,index_col=0,parse_dates=True,usecols=['local_15min','use'])
#Annual_9982=pd.read_csv("C:/Users/WSU-PNNL/Desktop/Data-pec/dataid/9982_data_2016_2018_all.csv",header=0,index_col=0,parse_dates=True,usecols=['local_15min','use'])
#Annual_7016=pd.read_csv("C:/Users/WSU-PNNL/Desktop/Data-pec/dataid/7016_data_2016_2018_all.csv",header=0,index_col=0,parse_dates=True,usecols=['local_15min','use'])
#Annual_6101=pd.read_csv("C:/Users/WSU-PNNL/Desktop/Data-pec/dataid/6101_data_2016_2018_all.csv",header=0,index_col=0,parse_dates=True,usecols=['local_15min','use'])
#Annual_3967=pd.read_csv("C:/Users/WSU-PNNL/Desktop/Data-pec/dataid/3967_data_2015_2018_all.csv",header=0,index_col=0,parse_dates=True,usecols=['local_15min','use'])
#Annual_3635=pd.read_csv("C:/Users/WSU-PNNL/Desktop/Data-pec/dataid/3635_data_2016_2018_all.csv",header=0,index_col=0,parse_dates=True,usecols=['local_15min','use'])
#Annual_2199=pd.read_csv("C:/Users/WSU-PNNL/Desktop/Data-pec/dataid/2199_data_2016_2018_all.csv",header=0,index_col=0,parse_dates=True,usecols=['local_15min','use'])
#Annual_624=pd.read_csv("C:/Users/WSU-PNNL/Desktop/Data-pec/dataid/624_data_2016_2018_all.csv",header=0,index_col=0,parse_dates=True,usecols=['local_15min','use'])

