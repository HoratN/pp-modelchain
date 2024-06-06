##############################################################################
#------------------------------EMOS hourly-----------------------------------#
##############################################################################
import pandas as pd
import numpy as np

from scipy.optimize import Bounds
from scipy.optimize import minimize
import hydrostats.ens_metrics as em

import sys 
sys.path.insert(0, 'D:\PV_power_postprocessing\code')

from utils.helper_load_preprocess import load_preprocess_data
from utils.helper_train_predict import train_test_split, prepare_data_ghi, prepare_data_pv

from utils.helper_crps import crps_normal_censored

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

#-------------------------------------------------EMOS--------------------------------------------------#
# GHI: use normal distribution that is left-censored at 0, since during nighttime GHI is 0.
# approximate function params with:
#        loc = a + b*(GHI_mem1 + GHI_mem2 + ... + GHI_mem50)
#        var = c + d*var(GHI_Ensemble)
# optimise CRPS to find  optimal parameters a, b, c and d

#----------------------------------------------Data Split-----------------------------------------------#
## Train data (2017-2019), test data (2020)
GHI_obs_train, GHI_obs_test = train_test_split(GHI_obs)
GHI_ens_train, GHI_ens_test = train_test_split(GHI_ens)

#-------------------------------------------Data Preparation--------------------------------------------#
# Prepare two dataframes with all variables used for EMOS: obs, hour_ID, GHI_mean, GHI_var

# Train data (2017-2019)
GHI_data_train = prepare_data_ghi(GHI_obs_train, GHI_ens_train)
# Test data (2020)
GHI_data_test = prepare_data_ghi(GHI_obs_test, GHI_ens_test)

#-------------------------------------------Optimization step-------------------------------------------#
# Optimization objective: 4 parameters (a=x[0], b=x[1], c=x[2], d=x[3])
# Optimization function:  continous ranked probability score (CRPS)

# we define a, b, c and d as arrays as we will obtain 24 different values (one for each hour of the day)
a, b, c, d = [], [], [], []
for i in range(24):
    print(i)
    def objective(x):
        return crps_normal_censored(obs=GHI_data_train[GHI_data_train.hour_ID==i]['obs'],
                                    loc=x[0] + x[1]*GHI_data_train[GHI_data_train.hour_ID==i]['GHI_mean'],
                                    var=x[2] + x[3]*GHI_data_train[GHI_data_train.hour_ID==i]['GHI_var'],
                                    lb=0)

    bounds = Bounds([-100, 0.001, 0.1, 0.001],[100, 100, 100, 15])
    # initial guess of parameters to start optimization from
    x0 = np.array([1,1,1,1])
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, options={'ftol': 1e-9, 'maxiter': 200,  'disp': True})

    a.append(res.x[0])
    b.append(res.x[1])
    c.append(res.x[2])
    d.append(res.x[3])

# write out coeffs
coeffs=pd.DataFrame({'a': a, 'b': b, 'c':  c, 'd':  d})
coeffs.to_csv(f'{training_path}/EMOS_coeffs/coeffs_emos_hourly.csv', header=True)

#--------------------------------------------create predictions---------------------------------------------#

def compute_preds(data, a, b, c, d):
    mean = []
    var  = []
    for i in range(len(data)):
        hour_ID = data.hour_ID[i]
        mean.append(a[hour_ID] + b[hour_ID] * data.GHI_mean[i])
        var.append(c[hour_ID] + d[hour_ID] * data.GHI_var[i])
    return pd.Series(mean), pd.Series(var)

# test data
mean_ghi_test, var_ghi_test = compute_preds(GHI_data_test, a, b, c, d)

# train data
mean_ghi_train, var_ghi_train = compute_preds(GHI_data_train, a, b, c, d)

#--------------------------------------------Check scores---------------------------------------------#
# test data
obs_ghi_test = GHI_data_test['obs']

# score for raw forecasts
ens_ghi_test = GHI_ens_test.values
score_ghi_raw_test = em.crps_kernel (obs_ghi_test, ens_ghi_test)['crpsMean']
print ('The CRPS for ensemble forecasts with NO post-processing is: ' + str(score_ghi_raw_test) + ' [test data]')

# score for post-processed forecasts
score_ghi_pp_test = crps_normal_censored(obs=obs_ghi_test.values, loc=mean_ghi_test, var=var_ghi_test, lb=0)
print ('The CRPS for forecasts with EMOS post-processing is: ' + str(score_ghi_pp_test) + ' [test data]')


# train data
obs_ghi_train = GHI_obs_train['GHI']

# score for raw forecasts
ens_ghi_train = GHI_ens_train.values
score_ghi_raw_train = em.crps_kernel (obs_ghi_train, ens_ghi_train)['crpsMean']
print ('The CRPS for ensemble forecasts with NO post-processing is: ' + str(score_ghi_raw_train) + ' [train data]')

# score for post-processed forecasts
score_ghi_pp_train = crps_normal_censored(obs=obs_ghi_train.values, loc=mean_ghi_train, var=var_ghi_train, lb=0)
print ('The CRPS for forecasts with EMOS post-processing is: ' + str(score_ghi_pp_train) + ' [train data]')

#--------------------------------------------Export results---------------------------------------------#
export_ghi_test = pd.DataFrame({'mean': mean_ghi_test,
                                'sd':   np.sqrt(var_ghi_test)
                     })
export_ghi_test.index = GHI_ens_test.index
export_ghi_test.to_csv(f'{results_path}\ghi_pp_EMOS_hourly_test.csv')

# export headline scores 
scores_test_ghi = pd.DataFrame({'method' : ['-', 'EMOS_hourly'],
                                'pp': ['raw', 'pp'],
                                'test_score' : [score_ghi_raw_test, score_ghi_pp_test],
                                'train_score': [score_ghi_raw_train, score_ghi_pp_train]})
scores_test_ghi.to_csv(f'{results_path}\scores_ghi_EMOS_hourly.csv', sep = ';', decimal = ',')


#########################################################################################################
#-------------------------------------------------------------------------------------------------------#
#--------------------------------------Part 2: Model chain----------------------------------------------#
#-------------------------------------------------------------------------------------------------------#
#########################################################################################################

#---------------------------------------------Explanations----------------------------------------------#
# The following steps will be performed on
# A: the raw ensemble with no post-processing
# B: ensembles of 50 members that are randomly drawn from the distribution found through EMOS postprocessing

#-----------------------------------------Create both ensembles-----------------------------------------#
ENS_A = GHI_ens.copy()
ENS_B = GHI_ens.copy()

# Fill the B ensemble with post-processed values
GHI_data = prepare_data_ghi(GHI_obs, GHI_ens)

def draw_members(data, ens):
    coeffs = pd.read_csv(f'{training_path}\EMOS_coeffs\coeffs_emos_hourly.csv')
    a = coeffs.a
    b = coeffs.b
    c = coeffs.c
    d = coeffs.d

    for i in range(len(data)):
        hour_ID = ens.index.hour.values[i]
        mean = a[hour_ID] + b[hour_ID] * data.GHI_mean[i]
        sd = np.sqrt(c[hour_ID] + d[hour_ID] * data.GHI_var[i])
        ens.iloc[i,0:50] = np.random.normal(mean, sd, 50)
    ens[ens < 0 ] = 0
    return ens

ENS_B = draw_members(GHI_data, ENS_B)
        
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

#-------------------------------------------------EMOS--------------------------------------------------#
# PV: use normal distribution that is censored at [0,20], matching the production limits of the PV unit
# approximate function params with:
#        loc = a + b*(PV_mem1 + PV_mem2 + ... + PV_mem50)
#        var = c + d*var(PV_Ensemble)
# optimise CRPS to find  optimal parameters a, b, c and d

#----------------------------------------------Data Split-----------------------------------------------#

# Train data (2017-2019), test data (2020)
PV_obs_train, PV_obs_test = train_test_split(PV_obs['SAM_gen'])
PV_ens_train = list([0,0]); PV_ens_test = list([0,0])
PV_ens_train[0], PV_ens_test[0] = train_test_split(PV_ens[0].drop(columns='zenith')) # raw ensemble
PV_ens_train[1], PV_ens_test[1] = train_test_split(PV_ens[1].drop(columns='zenith')) # pp ensemble

#-------------------------------------------Data Preparation--------------------------------------------#
# Prepare two dataframes with all variables used for EMOS: obs, hour_ID, PV_mean, PV_var

PV_data_train, PV_data_test = [], []
for n in range(2):
    PV_data_train.append(prepare_data_pv(PV_obs_train, PV_ens_train[n]))
    PV_data_test.append(prepare_data_pv(PV_obs_test, PV_ens_test[n]))

#-------------------------------------------Optimization step-------------------------------------------#
# Optimization objective: 4 parameters (a=x[0], b=x[1], c=x[2], d=x[3])
# Optimization function:  continous ranked probability score (CRPS)

# we define a, b, c and d as arrays as we will obtain 2x24 different values (for each hour of the day)
a, b, c, d = [], [], [], []
for n in range(2):
    if n == 0:
        print('Optimizing parameters for the raw GHI-Ensemble')
        status = ' [raw ensemble]'
    else:
        print('Optimizing parameters for the post-processed GHI-Ensemble')
        status = ' [post-processed ensemble]'
    a_inner, b_inner, c_inner, d_inner = [], [], [], []
    for i in range(24):
        print('\nOptimizing parameters for the hour ' + str(i) + ':00' + status)
        def objective(x):
            return crps_normal_censored(obs=PV_data_train[n][PV_data_train[n].hour_ID==i]['obs'],
                                        loc=x[0] + x[1]*PV_data_train[n][PV_data_train[n].hour_ID==i]['PV_mean'],
                                        var=x[2] + x[3]*PV_data_train[n][PV_data_train[n].hour_ID==i]['PV_var'],
                                        lb=0,
                                        ub=20)


        bounds = Bounds([-10, 0.001, -5, 0.00001],[100, 100, 100, 100])
        # initial guess of parameters to start optimization from
        x0 = np.array([1,1,1,1])
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, options={'ftol': 1e-9, 'maxiter': 200,  'disp': True})

        a_inner.append(res.x[0])
        b_inner.append(res.x[1])
        c_inner.append(res.x[2])
        d_inner.append(res.x[3])
        
    a.append(a_inner)
    b.append(b_inner)
    c.append(c_inner)
    d.append(d_inner)

# write out coeffs
for n in range(2):
    coeffs_pv=pd.DataFrame({'a': a[n], 'b': b[n], 'c':  c[n], 'd':  d[n]})
    if n == 0:
        coeffs_pv.to_csv(f'{training_path}\EMOS_coeffs\coeffs_emos_hourly_pv_raw_pp.csv', header=True)
    else:
        coeffs_pv.to_csv(f'{training_path}\EMOS_coeffs\coeffs_emos_hourly_pv_pp_pp.csv', header=True)

#--------------------------------------------create predictions---------------------------------------------#
def compute_preds_pv(data, a, b, c, d):
    mean = []
    var  = []
    for i in range(len(data)):
        hour_ID = data.hour_ID[i]
        mean.append(a[hour_ID] + b[hour_ID] * data.PV_mean[i])
        var.append(c[hour_ID] + d[hour_ID] * data.PV_var[i])
    return pd.Series(mean), pd.Series(var)

#--------------------------------------------Look at scores---------------------------------------------#
# Ultimately we have four models:
# A.A: raw GHI ensemble with raw PV ensemble
# A.B: raw GHI ensemble with post-processed PV ensemble
# B.A: post-processed GHI ensemble with raw PV ensemble
# B.B: post-processed GHI ensemble with post-processed PV ensemble

# test data
obs_pv_test = PV_obs_test

# A.A:
ens_pv_test_raw = PV_ens_test[0].values
score_pv_test_rawraw = em.crps_kernel(obs_pv_test, ens_pv_test_raw)['crpsMean']
print ('The CRPS for forecasts with NO GHI post-processing and NO PV post-processing is: ' + str(score_pv_test_rawraw) + ' [test data]')

# A.B:
mean_pv_test_rawpp, var_pv_test_rawpp = compute_preds_pv(PV_data_test[0], a[0], b[0], c[0], d[0])
score_pv_test_rawpp = crps_normal_censored(obs=obs_pv_test.values, loc=mean_pv_test_rawpp, var=var_pv_test_rawpp, lb=0, ub=20)
print ('The CRPS for forecasts with NO GHI post-processing and PV post-processing is: ' + str(score_pv_test_rawpp) + ' [test data]')

# B.A:
ens_pv_test_ppraw = PV_ens_test[1].values
score_pv_test_ppraw = em.crps_kernel (obs_pv_test, ens_pv_test_ppraw)['crpsMean']
print ('The CRPS for forecasts with GHI post-processing and NO PV post-processing is: ' + str(score_pv_test_ppraw) + ' [test data]')

# B.B:
mean_pv_test_pppp, var_pv_test_pppp = compute_preds_pv(PV_data_test[1], a[1], b[1], c[1], d[1])
score_pv_test_pppp = crps_normal_censored(obs=obs_pv_test.values, loc=mean_pv_test_pppp, var=var_pv_test_pppp, lb=0, ub=20)
print ('The CRPS for forecasts with GHI post-processing and PV post-processing is: ' + str(score_pv_test_pppp) + ' [test data]')

#--------------------------------------------Export results---------------------------------------------#
export_pv_test_rawraw = pd.DataFrame.from_records(ens_pv_test_raw)
export_pv_test_rawraw.index = PV_obs_test.index
export_pv_test_rawraw.to_csv(f'{results_path}\pv_rawraw_test.csv')

export_pv_test_ppraw = pd.DataFrame.from_records(ens_pv_test_ppraw)
export_pv_test_ppraw.index = PV_obs_test.index
export_pv_test_ppraw.to_csv(f'{results_path}\pv_ppraw_EMOS_hourly_test.csv')

export_pv_test_rawpp = pd.DataFrame({'mean': mean_pv_test_rawpp,
                                     'sd':   np.sqrt(var_pv_test_rawpp)
                                    })
export_pv_test_rawpp.index=PV_ens_test[0].index
export_pv_test_rawpp.to_csv(f'{results_path}\pv_rawpp_EMOS_hourly_test.csv')

export_pv_test_pppp = pd.DataFrame({'mean': mean_pv_test_pppp,
                                    'sd':   np.sqrt(var_pv_test_pppp)
                                    })
export_pv_test_pppp.index=PV_ens_test[0].index
export_pv_test_pppp.to_csv(f'{results_path}\pv_pppp_EMOS_hourly_test.csv')

#---------------------------------------- Look at train scores -----------------------------------------# 
# train
obs_pv_train = PV_obs_train

# A.A:
ens_pv_train = PV_ens_train[0].values
score_pv_train_rawraw = em.crps_kernel (obs_pv_train, ens_pv_train)['crpsMean']
print ('The CRPS for forecasts with NO GHI post-processing and NO PV post-processing is: ' + str(score_pv_train_rawraw) + ' [train data]')

# A.B:
mean_pv_train_rawpp, var_pv_train_rawpp = compute_preds_pv(PV_data_train[0], a[0], b[0], c[0], d[0])
score_pv_train_rawpp = crps_normal_censored(obs=obs_pv_train.values, loc=mean_pv_train_rawpp, var=var_pv_train_rawpp, lb=0, ub=20)
print ('The CRPS for forecasts with NO GHI post-processing and PV post-processing is: ' + str(score_pv_train_rawpp) + ' [train data]')

# B.A:
ens_pv_train_ppraw = PV_ens_train[1].values
score_pv_train_ppraw = em.crps_kernel (obs_pv_train, ens_pv_train_ppraw)['crpsMean']
print ('The CRPS for forecasts with GHI post-processing and NO PV post-processing is: ' + str(score_pv_train_ppraw) + ' [train data]')

# B.B:
mean_pv_train_pppp, var_pv_train_pppp = compute_preds_pv(PV_data_train[1], a[1], b[1], c[1], d[1])
score_pv_train_pppp = crps_normal_censored(obs=obs_pv_train.values, loc=mean_pv_train_pppp, var=var_pv_train_pppp, lb=0, ub=20)
print ('The CRPS for forecasts with GHI post-processing and PV post-processing is: ' + str(score_pv_train_pppp) + ' [train data]')

#--------------------------------------------Export results---------------------------------------------#
scores_pv = pd.DataFrame({'method' : ['-', 'EMOS_hourly', 'EMOS_hourly', 'EMOS_hourly'],
                        'pp': ['rawraw', 'rawpp', 'ppraw', 'pppp'],
                        'test_score' : [score_pv_test_rawraw, score_pv_test_rawpp, score_pv_test_ppraw, score_pv_test_pppp],
                        'train_score': [score_pv_train_rawraw, score_pv_train_rawpp, score_pv_train_ppraw, score_pv_train_pppp]})
scores_pv.to_csv(f'{results_path}\scores_pv_EMOS_hourly.csv', sep = ';', decimal = ',')