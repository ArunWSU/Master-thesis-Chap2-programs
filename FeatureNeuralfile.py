import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.layers import LSTM
from ModelPerformancefile import ModelPerformance
from sklearn.preprocessing import MinMaxScaler

# Fixing the random number for neural net
np.random.seed(1)
# FUNCTIONS AND CLASS DEFINITIONS
# Creates the input and output data for time series forecasting
# Model select 1.MLP 2.LSTM

class FeaturePreparation:
    actual_time_series,time_series_values=[],[] # Available attributes of class
    data_input,data_output,model_forecast_output=[],[],[]
    time_related_features=[]
    window_size=[]
    neural_model=[]
    
    def __init__(self,data,model_select,individual_paper_select,window_size_final,scale_input):
        self.scale_input=scale_input
        self.actual_time_series=data
        if(self.scale_input==1):
             self.scaler=MinMaxScaler(feature_range=(0,1))
             self.time_series_values=self.scaler.fit_transform(self.actual_time_series.values.reshape(-1,1))
        else:
            self.time_series_values=self.actual_time_series.values
        self.model_select=model_select
        self.individual_paper_select=individual_paper_select
        self.window_size=window_size_final
        self.accuracy_select=0
    
    def normalized_data_regenerate(self):
        self.unscaled_output=self.data_output
        self.unscaled_forecast=self.model_forecast_output
        self.data_output=self.scaler.inverse_transform(self.data_output.reshape(-1,1))
        self.model_forecast_output=self.scaler.inverse_transform(self.model_forecast_output.reshape(-1,1))
    
    def prepare_neural_input_output(self):
        self.data_input,self.data_output=[],[]
        for i in range(0,len(self.time_series_values)-self.window_size,1):
               self.last_lag_data=self.time_series_values[i:i+self.window_size]
               if(self.individual_paper_select==1):
                   self.last_lag_data=np.vstack((self.last_lag_data,self.individual_meter_forecast_features(self.last_lag_data).reshape(-1,1))) # Overloading might be possible variable args
               self.data_input.append(self.last_lag_data)
               self.data_output.append(self.time_series_values[i+self.window_size])# Y_t=history[5:]     
        self.data_input=np.array(self.data_input)
        self.data_output=np.array(self.data_output)
        if(self.model_select==1):
            self.data_input=self.data_input.reshape(-1,self.window_size)
            self.data_output=self.data_output.ravel()
             
    def find_time_related_features(self):
        self.time_related_features=pd.DataFrame(self.actual_time_series.index.dayofweek)
        self.time_related_features['Working day']=np.logical_not(((self.time_related_features==5)|(self.time_related_features==6))).astype('int')
        self.time_related_features['Hour']=pd.DataFrame(self.actual_time_series.index.hour)
        self.time_related_features=self.time_related_features[self.window_size:]
        self.time_related_features.reset_index(drop=True,inplace=True)
        self.time_related_features.rename(columns={'local_15min':'Day of week'},inplace=True)
        
    def individual_meter_forecast_features(self,input):
        individual_forecast_paper_features=np.zeros((16,1))
        x=np.array((-3,-6,-12,0))
        split_list=[input[a:] for a in x]
        for i,current_value in enumerate(split_list):
            individual_forecast_paper_features[i],individual_forecast_paper_features[i+4],individual_forecast_paper_features[i+8],individual_forecast_paper_features[i+12]=self.individual_meter_forecast_value_calculate(current_value)
        return individual_forecast_paper_features
    
    def individual_meter_forecast_value_calculate(self,previous_3): # Average, Maximum, Minimum, range of values
        return np.average(previous_3),np.amax(previous_3),np.amin(previous_3),np.ptp(previous_3)
    
    def neural_predict(self,FeaturePrepration):
        self.model_forecast_output=self.neural_model.predict(self.data_input)
        FeaturePrepration.neural_model=self.neural_model
        FeaturePrepration.model_forecast_output=FeaturePrepration.neural_model.predict(FeaturePrepration.data_input)
        if(self.scale_input==1):
            self.normalized_data_regenerate()
            FeaturePrepration.normalized_data_regenerate()
            
        # Model Performance Computation
        self.hist_perf_obj=ModelPerformance(self.data_output.reshape(-1,1),self.model_forecast_output.reshape(-1,1))
        self.hist_perf_obj.error_compute()
        FeaturePrepration.fore_perf_obj=ModelPerformance(FeaturePrepration.data_output.reshape(-1,1),FeaturePrepration.model_forecast_output.reshape(-1,1))
        FeaturePrepration.fore_perf_obj.error_compute()
        self.current_iter_error=np.array((self.hist_perf_obj.MSE,self.hist_perf_obj.MAE,self.hist_perf_obj.MAPE,FeaturePrepration.fore_perf_obj.MSE,FeaturePrepration.fore_perf_obj.MAE,FeaturePrepration.fore_perf_obj.MAPE))
        if(self.accuracy_select==1):
            self.hist_perf_obj.point_accuracy_compute()
            FeaturePrepration.fore_perf_obj.point_accuracy_compute()
                
    
class MLPModelParameters(FeaturePreparation):
    data_write=0
    
    # CHOICE OF WINDOW SIZE FOR FORECASTING      
    def __init__(self,data,model_select,individual_paper_select,window_size_final,scale_input,*args):
        args_len=len(args)
        if(args_len==0):
#            FeaturePreparation.__init__(self,data,model_select,individual_paper_select)
            super(MLPModelParameters,self).__init__(data,model_select,individual_paper_select,window_size_final,scale_input)
        else:
            self.window_size_max=args[0]
            self.neuron_number_max=args[1]
            self.data_write=args[2]
            super(MLPModelParameters,self).__init__(data,model_select,individual_paper_select,window_size_final,scale_input)
        
         
    def window_size_select(self,MLPModelParameters,filename): # self is hist_obj, MLPModelParameters refers to the fore_obj
        if(self.window_size_max > MLPModelParameters.actual_time_series.shape[0]):
            self.window_size_max=MLPModelParameters.actual_time_series.shape[0]
        self.hist_inp_list,self.hist_out_list,self.fore_inp_list,self.fore_out_list=[0]*(self.window_size_max+3),[0]*(self.window_size_max+3),[0]*(self.window_size_max+3),[0]*(self.window_size_max+3)
        self.Metrics_output_window_select=[]
          
          # Data as function of window sizes
        for x in range(1,self.window_size_max+1):
            # Input to MLP is of form No of samples, No of features
            self.window_size,MLPModelParameters.window_size=x,x
            self.prepare_neural_input_output()
            self.hist_inp_list[x-1],self.hist_out_list[x-1]=self.data_input,self.data_output
            MLPModelParameters.prepare_neural_input_output()
            self.fore_inp_list[x-1],self.fore_out_list[x-1]=MLPModelParameters.data_input,MLPModelParameters.data_output
            
            self.neural_fit(10)
            self.neural_predict(MLPModelParameters)
            self.Metrics_output_window_select.append(self.current_iter_error)
          
        self.Metrics_output_window_select1=pd.DataFrame(data=self.Metrics_output_window_select,columns=['MSE_train','MAE_train','MAPE_train','MSE_test','MAE_test','MAPE_test'])
        if(self.data_write==1):
            self.Metrics_output_window_select1.to_excel(filename) 
            
    # CHOICE OF NUMBER OF NEURONS FOR HIDDEN LAYER
    def neuron_select(self,MLPModelParameters,window_size,filename):
        self.data_input,MLPModelParameters.data_input=self.hist_inp_list[window_size-1],self.fore_inp_list[window_size-1]
        self.Metrics_output_neuron_select=[]
        for x in range(1,self.neuron_number_max+1):
            self.data_output,MLPModelParameters.data_output=self.hist_out_list[window_size-1],self.fore_out_list[window_size-1]
            self.neural_fit(x)
            self.neural_predict(MLPModelParameters)
            self.Metrics_output_neuron_select.append(self.current_iter_error)
            
        self.Metrics_output_neurons_select1=pd.DataFrame(data=self.Metrics_output_neuron_select,columns=['MSE_train','MAE_train','MAPE_train','MSE_test','MAE_test','MAPE_test'])
        if(self.data_write==1):
            self.Metrics_output_neurons_select1.to_excel(filename) 
    
    def neural_fit(self,number_of_neurons):
        mlp=MLPRegressor(hidden_layer_sizes=(number_of_neurons,),activation='logistic',solver='lbfgs',random_state=1)
        self.neural_model=mlp
        self.neural_model.fit(self.data_input,self.data_output)

class LSTMModelParameters(FeaturePreparation):
    
    def neural_fit(self):
        print("LSTM Model")
        self.LSTM_model_param()
    
    def LSTM_model_param(LSTMModelParameters):
        # Train LSTM model
        model=Sequential()
        model.add(LSTM(14,input_shape=(LSTMModelParameters.data_input.shape[1],LSTMModelParameters.data_input.shape[2])))
        model.add(Dense(1)) # First argument specifies the output
        model.compile(optimizer='adam',loss='mean_squared_error')
        model.fit(LSTMModelParameters.data_input,LSTMModelParameters.data_output,epochs=5,batch_size=1,verbose=2)
        LSTMModelParameters.neural_model=model

def obj_create(annual_data,start_date,end_date,model_select,individual_paper_select,window_size_final,scale_input,*var_args):
    data=annual_data[start_date:end_date]
#    a=[var_args[i] for i in range(len(var_args))]
    if(model_select==1):
        hist_object=MLPModelParameters(data,model_select,individual_paper_select,window_size_final,scale_input,*var_args) # use parameter
    else:
        hist_object=LSTMModelParameters(data,model_select,individual_paper_select,scale_input,window_size_final)
#    hist_object.find_time_related_features()
    hist_object.prepare_neural_input_output()  
    return hist_object