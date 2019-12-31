# -*- coding: utf-8 -*-
"""
Created on Tue May 21 11:37:13 2019

@author: WSU-PNNL
"""

#%% IMPORT FILES
import FeatureNeuralfile
import Plotstorefile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#Annual=pd.read_csv(r'D:\OpenDSS\OpenDSS\IEEETestCases\13Bus_secondary\Line_634_6341.csv',header=None,names=['use'])
#Annual=pd.read_csv("C:/Users/WSU-PNNL/Desktop/Data-pec/3967_data_2015_2018.csv",header=0,index_col=0,parse_dates=True,usecols=['local_15min','use'])
#Annual.index=pd.date_range(start='2017-01-02',freq='15T',periods=len(Annual))
Annual=annual_complete['use']
hist_start_date='2017-01-02'
hist_end_date='2017-01-13'
fore_start_date='2017-01-16'
fore_end_date='2017-01-20'
#%% Neural and LSTM based forecasting
#Specify regressor model
model_select=1
individual_paper_select=0
window_size_final=5
mlp_parm_determination=1
scale_input=0
data_write=0
if model_select==1:
        if(mlp_parm_determination==1):
            window_size_max=30
            neuron_number_max=30
            hist_object=FeatureNeuralfile.obj_create(Annual,hist_start_date,hist_end_date,model_select,individual_paper_select,window_size_final,scale_input,window_size_max,neuron_number_max,data_write)
            fore_object=FeatureNeuralfile.obj_create(Annual,fore_start_date,fore_end_date,model_select,individual_paper_select,window_size_final,scale_input,window_size_max,neuron_number_max,data_write) 
            hist_object.window_size_select(fore_object,'D:/Window_check_3967.xlsx')
            hist_object.neuron_select(fore_object,window_size_final,'D:/Neuron_check_3967.xlsx')
        else:
            hist_object=FeatureNeuralfile.obj_create(Annual,hist_start_date,hist_end_date,model_select,individual_paper_select,window_size_final,scale_input)
            fore_object=FeatureNeuralfile.obj_create(Annual,fore_start_date,fore_end_date,model_select,individual_paper_select,window_size_final,scale_input) 
            hist_object.neural_fit(7)
            hist_object.accuracy_select=0
            hist_object.neural_predict(fore_object)
            hist_perf_obj=hist_object.hist_perf_obj
            fore_perf_obj=fore_object.fore_perf_obj
            
#plt.plot(fore_object.data_output,color='crimson',linewidth=2,linestyle='-',label='ISO NE-England Apr 25-26')
#plt.plot(fore_object.model_forecast_output,color='gray',linewidth=2,linestyle='-',label='MLP forecast')
#plt.ylabel('Load(MW)')
#plt.xlabel('Time(5min)')
#plt.legend()
#plt.show()  
#%% Training plots
# Fig 2.7
plt.figure()
a=np.arange(1,8,1)
plt.plot(hist_object.Metrics_output_window_select1['MAPE_train'],color='crimson',linewidth=2,linestyle='-',label='Train MAPE')
plt.plot(hist_object.Metrics_output_window_select1['MAPE_test'],color='gray',linewidth=2,linestyle='-',label='Test MAPE')
plt.ylabel('Mean Absolute Percentage Error(MAPE)%')
plt.xlabel('Window size')
plt.legend()
#plt.savefig('Train_3967',dpi=600)  

#ax2.plot(multiplier,(np.array(output_change)),color='gray',linewidth=2,linestyle='-',label='MW change')

#hist_object.data_input=np.hstack((hist_object.data_input,hist_object.time_related_features['Day of week'].values.reshape(-1,1)))
#fore_object.data_input=np.hstack((fore_object.data_input,fore_object.time_related_features['Day of week'].values.reshape(-1,1)))
        
#hist_object.data_input=np.hstack((hist_object.data_input,hist_object.time_related_features.values))  
#fore_object.data_input=np.hstack((fore_object.data_input,fore_object.time_related_features.values))
'''
Train_error=np.array((hist_perf_obj.MSE,hist_perf_obj.MAE,hist_perf_obj.MAPE)).reshape(-1,1)
Test_error=np.array((fore_perf_obj.MSE,fore_perf_obj.MAE,fore_perf_obj.MAPE)).reshape(-1,1)
Error=np.hstack((Train_error,Test_error))
Error_df=pd.DataFrame(Error,columns=['Train','Test'])
'''