##############################################################################
#--------------------- Code for direct conversion model ---------------------#
##############################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.models import  Model 
from keras.layers import Input, Dense, Concatenate
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

import sys 
sys.path.insert(0, 'D:\PV_power_postprocessing\code')

from utils.helper_load_preprocess import load_preprocess_data
from utils.helper_train_predict import train_test_split, standardize_predictors
from utils.helper_train_predict import prepare_data_pv_nn

from utils.helper_crps import crps_normal_censored, custom_loss_crps_wrapper_fct

data_path = "D:\PV_power_postprocessing\data"
training_path = "D:\PV_power_postprocessing\code\\models"
results_path = "D:\PV_power_postprocessing\code\\results"

#-----------------------------------Data preparation (preprocessing)-----------------------------------#
GHI_obs, PV_obs, GHI_ens, weather_pred = load_preprocess_data(data_path)

#########################################################################################################
#---------------------------------- Prediction using a neural network ----------------------------------#
#########################################################################################################

#--------------------------------------------Neural networks--------------------------------------------#
# Input: mean and standard deviation of the ensemble data, deterministic forecasts of t2m and windspeed
# Verification data: PV observation
# Output: optimized location and scale parameters for each forecast distribution
# We assume a doubly-censored Gaussian distribution, as PV power is limited to [0,20]

#----------------------------------------------Data Split-----------------------------------------------#
# Train data (2017-2019), test data (2020)
PV_obs_train, PV_obs_test = train_test_split(PV_obs)
GHI_ens_train, GHI_ens_test = train_test_split(GHI_ens)
weather_pred_train, weather_pred_test = train_test_split(weather_pred)

# combine relevant data in one df
train = prepare_data_pv_nn(PV_obs_train, GHI_ens_train, weather_pred_train)
test = prepare_data_pv_nn(PV_obs_test, GHI_ens_test, weather_pred_test)

#--------------------------------------------Standardization--------------------------------------------#
# standardize each hour separately
for hrs in range(24):
    data_train = train[train.hour == hrs].copy(deep = True)
    data_test = test[test.hour == hrs].copy(deep = True)
    data_train, data_test = standardize_predictors(data_train, data_test)

    train.loc[train['hour'] == hrs] = data_train
    test.loc[test['hour'] == hrs] = data_test

#--------------------------------------Neural network architecture--------------------------------------#
# Inputs: mean, sd, windspeed, t2m

# Implement 24 models, one per hour
hourly_model = []
hist = []
for hrs in range(24):#for hrs in range(24):range(16,22): #
    print('Training models for the Hour ' + str(hrs) + ':00')
    inner_model = []
    inner_hist = []
    data = train[train.hour == hrs]

    # Implement 10 models (predictions will be an average over all 10 predictions)
    for i in range(10):
        print ('Training model ' + str(i+1) + ' for hour ' + str(hrs) + ':00')
    
        # Inputlayer:
        features_in = Input(shape=(4,), name='atmospheric_variables')

        # Further layers as desired
        x = Dense(256, activation='relu')(features_in)
        x = Dense(256, activation='relu')(x)

        # Output layers
        out_loc = Dense(1, activation='linear')(x)
        out_sd = Dense(1, activation='softplus')(x)

        # Concatenate output layers into one tensor
        out = Concatenate(name='distribution_parameter')([out_loc, out_sd])
        if hrs in range(6,13): # night
            callback = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        else: # day
            callback = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)


        # build, compile and fit the model
        inner_model.append(Model(inputs=features_in, outputs=out))
        inner_model[i].compile(optimizer = Adam(0.01), loss = custom_loss_crps_wrapper_fct(LB=0, UB=20))
        inner_hist.append(inner_model[i].fit(data[['loc', 'sd', 'wind_speed', 'temp_air']], data[['obs']], 
                                             batch_size=256, epochs=200, validation_split=0.2,
                                             callbacks=[callback]))
    hourly_model.append(inner_model)
    hist.append(inner_hist)

#-------------------------------Look at history of all 10 models per hour-------------------------------#
for hrs in range(len(hist)):
    plt.figure(figsize=(12,4))
    print('Models for the hour ' + str(hrs) + ':00')
    for i in range(len(hist[hrs])):
        plt.subplot(2, 5, i+1)
        plt.title('Run ' + str(i+1))
        plt.plot(hist[hrs][i].history['loss'], label='Training loss', color='Blue')
        plt.plot(hist[hrs][i].history['val_loss'], label='Validation loss', color='Red')
        plt.xlabel('Iteration')
        plt.legend()
    plt.tight_layout()
    plt.savefig(f'{training_path}/NN_training_plots/direct_hourly_{hrs}_pv_train_val_loss.png', bbox_inches='tight')
    plt.close()
plt.close('all')
#--------------------------------------------Look at scores---------------------------------------------#

# Predict y for test and train data set
pred_test_df = pd.DataFrame(columns=['mean', 'sd'])
pred_train_df = pd.DataFrame(columns=['mean', 'sd'])

# For each hour we need to use the correct model hourly_model[hour]
for hrs in range(len(hourly_model)):
    y_test = 0
    y_train = 0
    for model in hourly_model[hrs]:
        y_test  = y_test  + model.predict(test[test['hour'] == hrs][['loc', 'sd', 'wind_speed', 'temp_air']])
        y_train  = y_train  + model.predict(train[train['hour'] == hrs][['loc', 'sd', 'wind_speed', 'temp_air']])
    for i in range(len(y_test)):
        pred_test_df.loc[test[test['hour'] == hrs].index[i]] = y_test[i,:] / 10.0
    for i in range(len(y_train)):
        pred_train_df.loc[train[train['hour'] == hrs].index[i]] = y_train[i,:] / 10.0
        
pred_test_df  = pred_test_df.sort_index()
pred_train_df = pred_train_df.sort_index()


# Results of the model on the TEST DATA
obs_test         = test['obs']

# NN-based forecast
mean_test = pred_test_df['mean']
var_test  = pred_test_df['sd']**2
score_test = crps_normal_censored(obs=obs_test, loc=mean_test, var=var_test, lb=0, ub=20)
print ('The CRPS for NN-based forecasts is: ' + str(score_test) + ' [test data]')

#--------------------------------------------Export results---------------------------------------------#
export = pd.DataFrame({'obs':  test['obs'],
                       'mean': pred_test_df['mean'],
                       'sd':   pred_test_df['sd']
                       })
export.index=PV_obs_test.index
export.to_csv(f'{results_path}\direct_hourly.csv')

#---------------------------------------- Look at train scores -----------------------------------------# 
# Results of the model on the TRAIN DATA
obs_train    = train['obs']

# NN-based forecast
mean_train = pred_train_df['mean']
var_train  = pred_train_df['sd']**2
score_train = crps_normal_censored(obs=obs_train, loc=mean_train, var=var_train, lb=0, ub=20)
print ('The CRPS for NN-based forecasts is: ' + str(score_train) + ' [train data]')

#--------------------------------------------Export results---------------------------------------------#
# export headline scores 
scores_test = pd.DataFrame({'method' : ['direct_hourly'],
                            'test_score' : [score_test],
                            'train_score': [score_train]})
scores_test.to_csv(f'{results_path}\scores_direct_hourly.csv', sep = ';', decimal = ',')