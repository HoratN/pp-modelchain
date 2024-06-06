##############################################################################
#----------------------------------NN global---------------------------------#
##############################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import hydrostats.ens_metrics as em

from keras.models import Model
from keras.layers import Input, Dense, Flatten, Concatenate, Embedding
from keras.optimizers import Adam
from keras.activations import relu
from keras.callbacks import EarlyStopping

import sys 
sys.path.insert(0, 'D:\PV_power_postprocessing\code')

from utils.helper_load_preprocess import load_preprocess_data
from utils.helper_train_predict import train_test_split, standardize_predictors
from utils.helper_train_predict import prepare_data_ghi_nn, prepare_data_pv_nn

from utils.helper_crps import crps_normal_censored, custom_loss_crps_wrapper_fct

from utils.helper_model_chain import get_zenith_angle, input_model_chain
from utils.helper_model_chain import run_model_chain, cleanup_model_chain_output

data_path = "D:\PV_power_postprocessing\data"
training_path = "D:\PV_power_postprocessing\code\\models"
results_path = "D:\PV_power_postprocessing\code\\results"

#-----------------------------------Data preparation (preprocessing)-----------------------------------#
GHI_obs, PV_obs, GHI_ens, weather_pred = load_preprocess_data(data_path)

#########################################################################################################
#-------------------------------------------------------------------------------------------------------#
#--------------------------------------Part 1: GHI post-processing--------------------------------------#
#-------------------------------------------------------------------------------------------------------#
#########################################################################################################

#--------------------------------------------Neural networks--------------------------------------------#
# Input: mean and standard deviation of the ensemble data, deterministic forecasts of t2m and windspeed
# Verification data: GHI observation
# Output: optimized location and scale parameters for each forecast distribution
# We assume a left censored Gaussian distribution, as irradiance can only be >= 0

#----------------------------------------------Data Split-----------------------------------------------#
# Train data (2017-2019), test data (2020)
GHI_obs_train, GHI_obs_test = train_test_split(GHI_obs)
GHI_ens_train, GHI_ens_test = train_test_split(GHI_ens)
weather_pred_train, weather_pred_test = train_test_split(weather_pred)

# combine relevant data in one df
GHI_train = prepare_data_ghi_nn(GHI_obs_train, GHI_ens_train, weather_pred_train)
GHI_test = prepare_data_ghi_nn(GHI_obs_test, GHI_ens_test, weather_pred_test)

#--------------------------------------------Standardization--------------------------------------------#
# Standarize all input variables
GHI_train, GHI_test = standardize_predictors(GHI_train, GHI_test)

#------------------------------------------- Neural network --------------------------------------------#

# Inputs: mean, sd, and embedded hour of the day

# Implement 10 models (predictions will be an average over all 10 predictions)
GHI_model = []
hist = []

for i in range(10):
    print ('Training model ' + str(i+1))

    # Two Input layers
    features_in = Input(shape=(4,))
    hour_in = Input(shape=(1,))

    # Embedding layer
    # embedds hour information into 2d latent space
    hours = pd.concat([GHI_train.hour, GHI_test.hour]).unique() # all hours occuring in the dataset
    emb = Embedding(len(hours), 2)(hour_in)
    emb = Flatten()(emb)

    # concat input features and embeddings
    x = Concatenate()([features_in, emb])

    x = Dense(256, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    
    # Define custom activation function (relu plus the constant 10^-3)
    # This is needed to enforce variance > 0.
    def relu_plus(x):
        return relu(x) + 10**-3
    
    # Output layers
    out_loc = Dense(1, activation='linear')(x)
    out_sd = Dense(1, activation=relu_plus)(x)

    # Concatenate output layers into one tensor
    out = Concatenate()([out_loc, out_sd])
    callback = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)


    # build, compile and fit the model
    GHI_model.append(Model(inputs=[features_in, hour_in], outputs=out))
    GHI_model[i].compile(optimizer = Adam(0.01), loss = custom_loss_crps_wrapper_fct(LB=0))
    hist.append(GHI_model[i].fit([GHI_train[['loc', 'sd', 'wind_speed', 'temp_air']], GHI_train[['hour']]],
                                 GHI_train[['obs']], batch_size=1000, epochs=50, validation_split=0.2,
                                 callbacks=[callback])
                )

#------------------------------ Look at history of all 10 models per hour ------------------------------#
plt.figure(figsize=(12,4))
for i in range(len(hist)):
    plt.subplot(2, 5, i+1)
    plt.title('Run ' + str(i+1))
    plt.plot(hist[i].history['loss'], label='Training loss', color='Blue')
    plt.plot(hist[i].history['val_loss'], label='Validation loss', color='Red')
    plt.xlabel('Iteration')
plt.tight_layout()
plt.savefig(f'{training_path}/NN_training_plots/nn_global_ghi_train_val_loss.png', bbox_inches='tight')

#-------------------------------------------- Predict for train and test ---------------------------------------------#

# Predict y for test and train data set and average over the 10 models
y_pred_test, y_pred_train = 0, 0

for m in GHI_model:
    y_pred_test = y_pred_test + m.predict([GHI_test[['loc', 'sd', 'wind_speed', 'temp_air']], GHI_test[['hour']]])
    y_pred_train = y_pred_train + m.predict([GHI_train[['loc', 'sd', 'wind_speed', 'temp_air']], GHI_train[['hour']]])

y_pred_test = y_pred_test / len (GHI_model) # 10.0
y_pred_train = y_pred_train / len(GHI_model) # 10.0

#---------------------------------------- Look at test scores -----------------------------------------#       
# Results of the model on the TEST DATA
obs_ghi_test         = GHI_test['obs']

# Raw ensemble forecast
ens_ghi_test = GHI_ens_test.values
score_ghi_raw_test = em.crps_kernel (obs_ghi_test, ens_ghi_test)['crpsMean']
print ('The CRPS for ensemble forecasts with NO post-processing is: ' + str(score_ghi_raw_test) + ' [test data]')

# NN post-processed forecast
mean_ghi_test = y_pred_test[:, 0]
var_ghi_test  = y_pred_test[:, 1]**2
score_ghi_pp_test = crps_normal_censored(obs=obs_ghi_test, loc=mean_ghi_test, var=var_ghi_test, lb=0)
print ('The CRPS for forecasts with NN post-processing is: ' + str(score_ghi_pp_test) + ' [test data]')

#--------------------------------------------Export results---------------------------------------------#
export_test_ghi = pd.DataFrame({'mean': y_pred_test[:,0],
                                'sd': y_pred_test[:,1]
                                })
export_test_ghi.index=GHI_obs_test.index
export_test_ghi.to_csv(f'{results_path}\ghi_pp_NN_global_test.csv')

#---------------------------------------- Look at train scores -----------------------------------------# 
# Results of the model on the TRAIN DATA
obs_ghi_train         = GHI_train['obs']

# Raw ensemble forecast
ens_ghi_train = GHI_ens_train.values
score_ghi_raw_train = em.crps_kernel (obs_ghi_train, ens_ghi_train)['crpsMean']
print ('The CRPS for ensemble forecasts with NO post-processing is: ' + str(score_ghi_raw_train) + ' [train data]')

# NN post-processed forecast
mean_ghi_train = y_pred_train[:, 0]
var_ghi_train  = y_pred_train[:, 1]**2
score_ghi_pp_train = crps_normal_censored(obs=obs_ghi_train, loc=mean_ghi_train, var=var_ghi_train, lb=0)
print ('The CRPS for forecasts with NN post-processing is: ' + str(score_ghi_pp_train) + ' [train data]')

#--------------------------------------------Export results---------------------------------------------#
# export headline scores 
scores_test_ghi = pd.DataFrame({'method' : ['-', 'NN_global'],
                                'pp': ['raw', 'pp'],
                                'test_score' : [score_ghi_raw_test, score_ghi_pp_test],
                                'train_score': [score_ghi_raw_train, score_ghi_pp_train]})
scores_test_ghi.to_csv(f'{results_path}\scores_ghi_NN_global.csv', sep = ';', decimal = ',')


#########################################################################################################
#-------------------------------------------------------------------------------------------------------#
#------------------------------------------Part 2: Model chain------------------------------------------#
#-------------------------------------------------------------------------------------------------------#
#########################################################################################################

#---------------------------------------------Explanations----------------------------------------------#
# The following steps will be performed on
# A: the raw ensemble with no post-processing
# B: ensembles of 50 members that are randomly drawn from the distribution found through post-processing

#-----------------------------------------Create both ensembles-----------------------------------------#
ENS_A = GHI_ens.copy() # not standardized
ENS_B = GHI_ens.copy()

# Fill the B ensemble with post-processed values
y_pred_all = np.concatenate([y_pred_train, y_pred_test])
mean = y_pred_all[:,0]
sd   = y_pred_all[:,1]

for i in range(len(ENS_B)):
    ENS_B.iloc[i,0:50] = np.random.normal(mean[i], sd[i], 50)
ENS_B[ENS_B < 0 ] = 0

# Add both ensembles in the list ENS (first entry is raw, second entry is post-processed)
ENS = []
ENS.append(ENS_A)
ENS.append(ENS_B)

#-----------------------------------------Run model chain-----------------------------------------------#
# zenith angle
zenith_angle = get_zenith_angle(GHI_obs)
for ens in ENS:
    ens['zenith'] = zenith_angle['zenith']

# run model chain for all members and the two ensembles
PV_ens = []
for n in (0,1):
    PV_ens.append(pd.DataFrame())
    
    for i in range(50):
        print('member: ', i)
        # prepare input
        input = input_model_chain(ENS[n], i, weather_pred)
        # Estimate the power output of the entire photovoltaic power station
        PV_ens[n]['PV AC' + str(i+1)] = run_model_chain(input)

# convert to MW, prune data at physical boundaries  
for n in (0, 1):
    PV_ens[n] = cleanup_model_chain_output(PV_ens[n], zenith_angle)


#########################################################################################################
#-------------------------------------------------------------------------------------------------------#
#---------------------------------------Part 3: PV post-processing--------------------------------------#
#-------------------------------------------------------------------------------------------------------#
#########################################################################################################

#--------------------------------------------Neural networks--------------------------------------------#
# Input: mean and standard deviation of the ensemble data, deterministic forecasts of t2m and windspeed
# Verification data: PV observation
# Output: optimized location and scale parameters for each forecast distribution
# We assume a doubly-censored Gaussian distribution, as PV power is limited to [0,20]

#----------------------------------------------Data Split-----------------------------------------------#
# Train data (2017-2019), test data (2020)
PV_train, PV_test = [],[]

PV_obs_train, PV_obs_test = train_test_split(PV_obs)
for n in (0,1):
    _PV_ens_train, _PV_ens_test = train_test_split(PV_ens[n].drop(columns='zenith'))

    PV_train.append(prepare_data_pv_nn(PV_obs_train, _PV_ens_train, weather_pred_train))
    PV_test.append(prepare_data_pv_nn(PV_obs_test, _PV_ens_test, weather_pred_test))

#--------------------------------------------Standardization--------------------------------------------#
for n in (0,1):
    PV_train[n], PV_test[n] = standardize_predictors(PV_train[n], PV_test[n])

#--------------------------------------Neural network architecture--------------------------------------#

PV_model = []
hist = []
for n in range(2):
    if (n == 0):
        print ('Training on raw ensemble data')
        status = ' [raw ensemble]'
    else:
        print ('Training on post-processed ensemble data')
        status = ' [post-processed ensemble]'
    
    
    # Implement 10 models for more stability (predictions will be an average over all 10 predictions)
    inner_model = []
    inner_hist = []
    for i in range(10):
        print ('Training model ' + str(i+1) + status)

        # Two Input layers
        features_in = Input(shape=(4,))
        hour_in     = Input(shape=(1,))

        # Embedding layer
        # embedds hour information into 2d latent space
        hours = pd.concat([PV_train[n].hour, PV_test[n].hour]).unique() # all hours occuring in the dataset
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
        out = Concatenate()([out_loc, out_sd])
        callback = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    
        # build, compile and fit the model
        inner_model.append(Model(inputs=[features_in, hour_in], outputs=out))

        inner_model[i].compile(optimizer = Adam(0.01), loss = custom_loss_crps_wrapper_fct(LB=0, UB=20))
        inner_hist.append(inner_model[i].fit([PV_train[n][['loc', 'sd', 'wind_speed', 'temp_air']], PV_train[n][['hour']]], 
                                             PV_train[n][['obs']], batch_size=1000, epochs=50, validation_split=0.2,
                                             callbacks=[callback])
                          )

    PV_model.append(inner_model)
    hist.append(inner_hist)

#------------------------------ Look at history of all 10 models per hour ------------------------------#
for n in range(2):
    if n == 0:
        print('Loss curves for the post-processing of a raw GHI-Ensemble')
    else:
        print('Loss curves for the post-processing of post-processed data')
    plt.figure(figsize=(12,4))
    for i in range(len(hist[n])):
        plt.subplot(2, 5, i+1)
        plt.title('Run ' + str(i+1))
        plt.plot(hist[n][i].history['loss'], label='Training loss', color='Blue')
        plt.plot(hist[n][i].history['val_loss'], label='Validation loss', color='Red')
        plt.xlabel('Iteration')
    plt.tight_layout()
    plt.savefig(f'{training_path}/NN_training_plots/nn_global_pv_train_val_loss.png', bbox_inches='tight')


#--------------------------------------------Look at scores---------------------------------------------#

# Predict y for test and train data set
y_pred_test_raw_ens, y_pred_train_raw_ens = 0, 0
y_pred_test_pp_ens, y_pred_train_pp_ens = 0, 0

for m in PV_model[0]:
    y_pred_test_raw_ens = y_pred_test_raw_ens + m.predict([PV_test[0][['loc', 'sd', 'wind_speed', 'temp_air']], PV_test[0][['hour']]])
    y_pred_train_raw_ens = y_pred_train_raw_ens + m.predict([PV_train[0][['loc', 'sd', 'wind_speed', 'temp_air']], PV_train[0][['hour']]])
y_pred_test_raw_ens = y_pred_test_raw_ens / len(PV_model[0]) # 10.0
y_pred_train_raw_ens = y_pred_train_raw_ens / len(PV_model[0]) # 10.0

for m in PV_model[1]:
    y_pred_test_pp_ens = y_pred_test_pp_ens + m.predict([PV_test[1][['loc', 'sd', 'wind_speed', 'temp_air']], PV_test[1][['hour']]])
    y_pred_train_pp_ens = y_pred_train_pp_ens + m.predict([PV_train[1][['loc', 'sd', 'wind_speed', 'temp_air']], PV_train[1][['hour']]])
y_pred_test_pp_ens = y_pred_test_pp_ens / len(PV_model[1]) # 10.0
y_pred_train_pp_ens = y_pred_train_pp_ens / len(PV_model[1]) # 10.0


# We have four scenarios:
# A.A: raw GHI ensemble with raw PV ensemble
# A.B: raw GHI ensemble with post-processed PV ensemble
# B.A: post-processed GHI ensemble with raw PV ensemble
# B.B: post-processed GHI ensemble with post-processed PV ensemble

# Results of the model on the TEST DATA
obs_pv_test = PV_test[0]['obs']

# A.A:
ens_pv_test_raw = PV_ens[0].iloc[:,1:]["2020-01-01 00:00:00":].values
score_pv_test_rawraw = em.crps_kernel (obs_pv_test, ens_pv_test_raw)['crpsMean']
print ('The CRPS for forecasts with NO GHI post-processing and NO PV post-processing is: ' + str(score_pv_test_rawraw) + ' [test data]')

# A.B:
mean_pv_test_rawpp = y_pred_test_raw_ens[:, 0]
var_pv_test_rawpp  = y_pred_test_raw_ens[:, 1]**2
score_pv_test_rawpp = crps_normal_censored(obs=obs_pv_test, loc=mean_pv_test_rawpp, var=var_pv_test_rawpp, lb=0, ub=20)
print ('The CRPS for forecasts with NO GHI post-processing and PV post-processing is: ' + str(score_pv_test_rawpp) + ' [test data]')

# B.A:
ens_pv_test_ppraw = PV_ens[1].iloc[:,1:]["2020-01-01 00:00:00":].values
score_pv_test_ppraw = em.crps_kernel (obs_pv_test, ens_pv_test_ppraw)['crpsMean']
print ('The CRPS for forecasts with GHI post-processing and NO PV post-processing is: ' + str(score_pv_test_ppraw) + ' [test data]')

# B.B:
mean_pv_test_pppp = y_pred_test_pp_ens[:, 0]
var_pv_test_pppp  = y_pred_test_pp_ens[:, 1]**2
score_pv_test_pppp = crps_normal_censored(obs=obs_pv_test, loc=mean_pv_test_pppp, var=var_pv_test_pppp, lb=0, ub=20)
print ('The CRPS for forecasts with GHI post-processing and PV post-processing is: ' + str(score_pv_test_pppp) + ' [test data]')

#--------------------------------------------Export results---------------------------------------------#
#PV_test[0][['obs']].to_csv('obs.csv')
export_pv_test_rawraw = pd.DataFrame.from_records(ens_pv_test_raw)
export_pv_test_rawraw.index = PV_obs_test.index
export_pv_test_rawraw.to_csv(f'{results_path}\pv_rawraw_test.csv')

export_pv_test_ppraw = pd.DataFrame.from_records(ens_pv_test_ppraw)
export_pv_test_ppraw.index = PV_obs_test.index
export_pv_test_ppraw.to_csv(f'{results_path}\pv_ppraw_NN_global_test.csv')

export_pv_test_rawpp = pd.DataFrame({'mean': y_pred_test_raw_ens[:,0],
                                     'sd':   y_pred_test_raw_ens[:,1]})
export_pv_test_rawpp.index= PV_obs_test.index
export_pv_test_rawpp.to_csv(f'{results_path}\pv_rawpp_NN_global_test.csv')

export_pv_test_pppp = pd.DataFrame({'mean': y_pred_test_pp_ens[:,0],
                                    'sd':   y_pred_test_pp_ens[:,1]})
export_pv_test_pppp.index=PV_obs_test.index
export_pv_test_pppp.to_csv(f'{results_path}\pv_pppp_NN_global_test.csv')

#---------------------------------------- Look at train scores -----------------------------------------# 
# Results of the model on the TRAIN DATA
obs_pv_train = PV_train[0]['obs']

# A.A:
ens_pv_train = PV_ens[0].iloc[:,1:][:"2019-12-31 23:00:00"].values
score_pv_train_rawraw = em.crps_kernel (obs_pv_train, ens_pv_train)['crpsMean']
print ('The CRPS for forecasts with NO GHI post-processing and NO PV post-processing is: ' + str(score_pv_train_rawraw) + ' [train data]')

# A.B:
mean_pv_train_rawpp = y_pred_train_raw_ens[:, 0]
var_pv_train_rawpp  = y_pred_train_raw_ens[:, 1]**2
score_pv_train_rawpp = crps_normal_censored(obs=obs_pv_train, loc=mean_pv_train_rawpp, var=var_pv_train_rawpp, lb=0, ub=20)
print ('The CRPS for forecasts with NO GHI post-processing and PV post-processing is: ' + str(score_pv_train_rawpp) + ' [train data]')

# B.A:
ens_pv_train_ppraw = PV_ens[1].iloc[:,1:][:"2019-12-31 23:00:00"].values
score_pv_train_ppraw = em.crps_kernel (obs_pv_train, ens_pv_train_ppraw)['crpsMean']
print ('The CRPS for forecasts with GHI post-processing and NO PV post-processing is: ' + str(score_pv_train_ppraw) + ' [train data]')

# B.B:
mean_pv_train_pppp = y_pred_train_pp_ens[:, 0]
var_pv_train_pppp  = y_pred_train_pp_ens[:, 1]**2
score_pv_train_pppp = crps_normal_censored(obs=obs_pv_train, loc=mean_pv_train_pppp, var=var_pv_train_pppp, lb=0, ub=20)
print ('The CRPS for forecasts with GHI post-processing and PV post-processing is: ' + str(score_pv_train_pppp) + ' [train data]')

#---------------------------------------- save headline scores -----------------------------------------#  
scores_pv = pd.DataFrame({'method' : ['-', 'NN_global', 'NN_global', 'NN_global'],
                        'pp': ['rawraw', 'rawpp', 'ppraw', 'pppp'],
                        'test_score' : [score_pv_test_rawraw, score_pv_test_rawpp, score_pv_test_ppraw, score_pv_test_pppp],
                        'train_score': [score_pv_train_rawraw, score_pv_train_rawpp, score_pv_train_ppraw, score_pv_train_pppp]})
scores_pv.to_csv(f'{results_path}\scores_pv_NN_global.csv', sep = ';', decimal = ',')