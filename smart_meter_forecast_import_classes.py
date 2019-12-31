#%% IMPORT FILES
import FeatureNeuralfile
import Plotstorefile
import numpy as np
import pandas as pd
#%%
#Specify MLP regressor model
#x=int(input("Enter 1. MLP 2. LSTM"))
#exec(open("./Data_read.py").read())
model_select=1
individual_paper_select=0
window_size_final=5
mlp_parm_determination=0
scale_input=0

if model_select==1:
        if(mlp_parm_determination==1):
            window_size_max=40
            neuron_number_max=40
            hist_object=FeatureNeuralfile.obj_create(Annual,hist_start_date,hist_end_date,model_select,individual_paper_select,window_size_final,scale_input,window_size_max,neuron_number_max)
            fore_object=FeatureNeuralfile.obj_create(Annual,fore_start_date,fore_end_date,model_select,individual_paper_select,window_size_final,scale_input,window_size_max,neuron_number_max) 
            hist_object.window_size_select(fore_object)
            hist_object.neuron_select(fore_object,5)
        else:
            hist_object=FeatureNeuralfile.obj_create(Annual,hist_start_date,hist_end_date,model_select,individual_paper_select,window_size_final,scale_input)
            fore_object=FeatureNeuralfile.obj_create(Annual,fore_start_date,fore_end_date,model_select,individual_paper_select,window_size_final,scale_input) 
            hist_object.neural_fit(7)
else:
    hist_object=FeatureNeuralfile.obj_create(Annual,hist_start_date,hist_end_date,model_select,individual_paper_select,window_size_final,scale_input)
    hist_object.neural_fit()
    fore_object=FeatureNeuralfile.obj_create(Annual,fore_start_date,fore_end_date,model_select,individual_paper_select,window_size_final,scale_input)
    

#hist_object.data_input=np.hstack((hist_object.data_input,hist_object.time_related_features['Day of week'].values.reshape(-1,1)))
#fore_object.data_input=np.hstack((fore_object.data_input,fore_object.time_related_features['Day of week'].values.reshape(-1,1)))
        
#hist_object.data_input=np.hstack((hist_object.data_input,hist_object.time_related_features.values))  
#fore_object.data_input=np.hstack((fore_object.data_input,fore_object.time_related_features.values))

hist_object.accuracy_select=1
hist_object.neural_predict(fore_object)
hist_perf_obj=hist_object.hist_perf_obj
fore_perf_obj=fore_object.fore_perf_obj
Train_error=np.array((hist_perf_obj.MSE,hist_perf_obj.MAE,hist_perf_obj.MAPE)).reshape(-1,1)
Test_error=np.array((fore_perf_obj.MSE,fore_perf_obj.MAE,fore_perf_obj.MAPE)).reshape(-1,1)
Error=np.hstack((Train_error,Test_error))
Error_df=pd.DataFrame(Error,columns=['Train','Test'])
# plot visualization
#individual_plot_labels=['Training load','MLP Training Load','Actual load without zeros']
individual_plot_labels=['Actual Jan 2','MLP forecast']
#individual_plot_labels=['Actual Jan 2','LSTM forecast']
fig_labels=['Training Set','Time(Datapoint(15min))','Load(KW)']
#plot_list=[annual_data_series[140:170]]
plot_list=[hist_object.data_output[0:97],hist_object.model_forecast_output[0:97]]#97:193 history_output_old[0:97]
histplotobj=Plotstorefile.Plotstore(individual_plot_labels,fig_labels,plot_list) #'Try 2'
#histplotobj.plot_results()
plot_list=[fore_object.data_output[0:97],fore_object.model_forecast_output[0:97]]
individual_plot_labels[0]='Actual Jan 9'
fig_labels[0]='Testing dataset'
foreplotobj=Plotstorefile.Plotstore(individual_plot_labels,fig_labels,plot_list) #'Try 2'
#foreplotobj.plot_results()