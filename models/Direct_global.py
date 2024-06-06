##############################################################################
#--------------------- Code for direct conversion model ---------------------#
##############################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.models import  Model 
from keras.layers import Input, Dense, Flatten, Concatenate, Embedding
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
train, test = standardize_predictors(train, test)

#-------------------------------------------- Neural network -------------------------------------------#
# Inputs: mean, sd, windspeed, t2m, and hour id of the day for the embedding

# Implement 10 models (predictions will be an average over all 10 predictions)
model = []
hist = []

for i in range(10):
    print ('Training model ' + str(i+1))

    # Two Inputlayer:
    features_in = Input(shape=(4,), name='atmospheric_variables')
    hour_in = Input(shape=(1,), name='hour')

    # Embedding layer
    # embedds hour information into 2d latent space
    hours = pd.concat([train.hour, test.hour]).unique()
    emb = Embedding(len(hours), 2)(hour_in)
    emb = Flatten()(emb)

    # concat input features to embeddings
    x = Concatenate()([features_in, emb])

    x = Dense(256, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    
    # Output layers
    out_loc = Dense(1, activation='linear')(x)
    out_sd = Dense(1, activation='softplus')(x)

    # Concatenate output layers into one tensor
    out = Concatenate(name='distribution_parameter')([out_loc, out_sd])
    callback = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)


    # build, compile and fit the model
    model.append(Model(inputs=[features_in, hour_in], outputs=out))
    model[i].compile(optimizer = Adam(0.01), loss = custom_loss_crps_wrapper_fct(LB=0, UB=20))
    hist.append(model[i].fit([train[['loc', 'sd', 'wind_speed', 'temp_air']], 
                              train[['hour']]], train[['obs']], 
                              batch_size=1000, epochs=50, validation_split=0.2,
                              callbacks=[callback]))

#-------------------------------Look at history of all 10 models per hour-------------------------------#
plt.figure(figsize=(12,4))
for i in range(len(hist)):
    plt.subplot(2, 5, i+1)
    plt.title('Run ' + str(i+1))
    plt.plot(hist[i].history['loss'], label='Training loss', color='Blue')
    plt.plot(hist[i].history['val_loss'], label='Validation loss', color='Red')
    plt.xlabel('Iteration')
    plt.legend()
plt.tight_layout()
plt.savefig(f'{training_path}/NN_training_plots/direct_global_pv_train_val_loss.png', bbox_inches='tight')

plt.close('all')

#--------------------------------------------Look at scores---------------------------------------------#

# Predict y for test and train data set
y_pred_test, y_pred_train = 0, 0

for m in model:
    y_pred_test = y_pred_test + m.predict([test[['loc', 'sd', 'wind_speed', 'temp_air']], test[['hour']]])
    y_pred_train = y_pred_train + m.predict([train[['loc', 'sd', 'wind_speed', 'temp_air']], train[['hour']]])
y_pred_test = y_pred_test / 10.0
y_pred_train = y_pred_train / 10.0


# Results of the model on the TEST DATA
obs_test    = test['obs']

# NN-based forecast
mean_test = y_pred_test[:, 0]
var_test  = y_pred_test[:, 1]**2
score_test = crps_normal_censored(obs=obs_test, loc=mean_test, var=var_test, lb=0, ub=20)
print ('The CRPS for NN-based forecasts is: ' + str(score_test) + ' [test data]')

#--------------------------------------------Export results---------------------------------------------#
export = pd.DataFrame({'obs':  test['obs'],
                       'mean': y_pred_test[:, 0],
                       'sd':   y_pred_test[:, 1]
                       })
export.index=PV_obs_test.index
export.to_csv(f'{results_path}\direct_global.csv')

#---------------------------------------- Look at train scores -----------------------------------------# 

# Results of the model on the TRAIN DATA
obs_train    = train['obs']

# NN-based forecast
mean_train = y_pred_train[:, 0]
var_train  = y_pred_train[:, 1]**2
score_train = crps_normal_censored(obs=obs_train, loc=mean_train, var=var_train, lb=0, ub=20)
print ('The CRPS for NN-based forecasts is: ' + str(score_train) + ' [train data]')

#--------------------------------------------Export results---------------------------------------------#
# export headline scores 
scores_test = pd.DataFrame({'method' : ['direct_global'],
                            'test_score' : [score_test],
                            'train_score': [score_train]})
scores_test.to_csv(f'{results_path}\scores_direct_global.csv', sep = ';', decimal = ',')