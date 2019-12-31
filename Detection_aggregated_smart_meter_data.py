# Arun Master thesis Section 2.5 program 
import pandas as pd
import numpy as np
import random
from matplotlib import pyplot as plt
#import os

class MultiDataidAnnual:
    
    def __init__(self,list_data,n,random_index):
        self.Annual=0
        for i in range(0,n,1):
            self.Annual=self.Annual+list_data[random_index[i]]

hist_start_date='2016-01-05'
hist_end_date='2016-01-10'
fore_start_date='2016-01-11'
fore_end_date='2016-01-17'

#Multi data random aggregation
Annual_list=[Annual_252, Annual_2575, Annual_2829, Annual_2974, Annual_3039, Annual_3500, Annual_3635, Annual_4998, Annual_5738, Annual_6423, Annual_7016, Annual_7504, Annual_8155, Annual_94, Annual_9451, Annual_9919]
Annual_list1=[a[hist_start_date:fore_end_date] for a in Annual_list]
Annual_data_time_interv=[]
Data_id_time=[]
for a in Annual_list1:
    if not a.empty:
        Annual_data_time_interv.append(a["use"])
        Data_id_time.append(a.loc['2016-01-05 03:30:00']["dataid"])
n=len(Annual_data_time_interv)
random_index=random.sample(range(0,n),n)

#%% Training and test error as function of aggregation
Train_MSE,Train_MAE,Train_MAPE=np.zeros((n,1)),np.zeros((n,1)),np.zeros((n,1))
Test_MSE,Test_MAE,Test_MAPE=np.zeros((n,1)),np.zeros((n,1)),np.zeros((n,1))
hist_inp,hist_out=[],[]
fore_inp,fore_out=[],[]
for j in range(0,n,1):
    obj1=MultiDataidAnnual(Annual_data_time_interv,j+1,random_index)
    Annual=obj1.Annual
    exec(open("./smart_meter_forecast_import_classes.py").read())
    hist_inp.append(hist_object.data_output)
    hist_out.append(hist_object.model_forecast_output)
    fore_inp.append(fore_object.data_output)
    fore_out.append(fore_object.model_forecast_output)
    Train_MSE[j],Train_MAE[j],Train_MAPE[j]=hist_perf_obj.MSE,hist_perf_obj.MAE,hist_perf_obj.MAPE
    Test_MSE[j],Test_MAE[j],Test_MAPE[j]=fore_perf_obj.MSE,fore_perf_obj.MAE,fore_perf_obj.MAPE
Error=np.hstack((Train_MSE,Train_MAE,Train_MAPE,Test_MSE,Test_MAE,Test_MAPE))
Error_df=pd.DataFrame(Error,columns=['MSE_train','MAE_train','MAPE_train','MSE_test','MAE_test','MAPE_test'])
#    Error_df.to_excel('Multiple_dataid.xlsx')
#%% Thesis Chap 2 results
save_fig=0

# Figure 2.12
plt.figure()
plt.plot(hist_inp[2][0:96],color='crimson',linewidth=2,linestyle='-',label='Training output')
plt.plot(hist_out[2][0:96],color='gray',linewidth=2,linestyle='-',label='MLP forecasted output')
plt.ylabel('Real power (kW)')
plt.xlabel('Time steps')
plt.legend()
if(save_fig):
    plt.savefig("Three meter aggregation train",dpi=600)

# Figure 2.13
plt.figure()
plt.plot(hist_inp[11][0:96],color='crimson',linewidth=2,linestyle='-',label='Training output')
plt.plot(hist_out[11][0:96],color='gray',linewidth=2,linestyle='-',label='MLP forecasted output')
plt.ylabel('Real power (kW)')
plt.xlabel('Time steps')
plt.legend()
if(save_fig):
    plt.savefig("Twelve meter aggregation train",dpi=600)
    
# Figure 2.14-2.15
plt.figure()
a=np.arange(1,8,1)
plt.plot(Error_df['MAPE_train'],color='crimson',linewidth=2,linestyle='-',label='Train MAPE')
plt.plot(Error_df['MAPE_test'],color='gray',linewidth=2,linestyle='-',label='Test MAPE')
plt.ylabel('Real power (kW)')
plt.xlabel('No of smart meters aggregated')
plt.legend()
if(save_fig):
    plt.savefig('Aggregation plot', dpi=600)
    
# Testing dataset results
plt.figure()
plt.plot(fore_inp[2][0:96],color='crimson',linewidth=2,linestyle='-',label='Testing output')
plt.plot(fore_out[2][0:96],color='gray',linewidth=2,linestyle='-',label='MLP forecasted output')
plt.ylabel('Real power (kW)')
plt.xlabel('Time steps')
plt.legend()
if(save_fig):
    plt.savefig("Three meter aggregation test",dpi=600)

plt.figure()
plt.plot(fore_inp[11],color='crimson',linewidth=2,linestyle='-',label='Testing output')
plt.plot(fore_out[11],color='gray',linewidth=2,linestyle='-',label='MLP forecasted output')
plt.ylabel('Real power (kW)')
plt.xlabel('Time steps')
plt.legend()
if(save_fig):
    plt.savefig("Twelve meter aggregation test full",dpi=600)


