##############################################################################
#------------------------------EMOS global-----------------------------------#
##############################################################################
import pandas as pd
import numpy as np
from scipy.optimize import Bounds
from scipy.optimize import minimize
import hydrostats.ens_metrics as em

import sys 
sys.path.insert(0, 'D:\PV_power_postprocessing\code')

from utils.helper_load_preprocess import load_preprocess_data
from utils.helper_train_predict import train_test_split

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
# Train data (2017-2019), test data (2020)
GHI_obs_train, GHI_obs_test = train_test_split(GHI_obs)
GHI_ens_train, GHI_ens_test = train_test_split(GHI_ens)

#-------------------------------------------Data Preparation--------------------------------------------#
# Define helping variables

# Mean of all ensemble members
GHI_ens_train_MEAN = GHI_ens_train.mean(axis=1).values
GHI_ens_test_MEAN  = GHI_ens_test.mean(axis=1).values

# Variance of the ensemble
GHI_ens_train_VAR = GHI_ens_train.var(axis=1)
GHI_ens_test_VAR  = GHI_ens_test.var(axis=1)

#-------------------------------------------Optimization step-------------------------------------------#
# Optimization objective: 4 parameters (a=x[0], b=x[1], c=x[2], d=x[3])
# Optimization function:  continous ranked probability score (CRPS)

def objective(x):
    return crps_normal_censored(obs=GHI_obs_train['GHI'],
                                loc=x[0] + x[1]*GHI_ens_train_MEAN,
                                var=x[2] + x[3]*GHI_ens_train_VAR,
                                lb=0)

bounds = Bounds([-100, 0.001, -100, 0.001],[100, 100, 100, 100])
# initial guess of parameters to start optimization from
x0 = np.array([1,1,1,1])
res = minimize(objective, x0, method='SLSQP', bounds=bounds, options={'ftol': 1e-9, 'maxiter': 200,  'disp': True})

a = res.x[0]
b = res.x[1]
c = res.x[2]
d = res.x[3]

# write out coeffs
coeffs=pd.DataFrame({'a': [a], 'b': [b], 'c': [c], 'd': [d]})
coeffs.to_csv(f'{training_path}/EMOS_coeffs/coeffs_emos_global.csv', header=True)

#--------------------------------------------create predictions---------------------------------------------#

# test data
mean_ghi_test = a + b*GHI_ens_test_MEAN
var_ghi_test = c + d*GHI_ens_test_VAR

# train data
mean_ghi_train = a + b*GHI_ens_train_MEAN
var_ghi_train = c + d*GHI_ens_train_VAR

#--------------------------------------------Check scores---------------------------------------------#
# test data
obs_ghi_test = GHI_obs_test['GHI']

# score for raw forecasts
ens_ghi_test = GHI_ens_test.values
score_ghi_raw_test = em.crps_kernel (obs_ghi_test, ens_ghi_test)['crpsMean']
print ('The CRPS for ensemble forecasts with NO post-processing is: ' + str(score_ghi_raw_test) + ' [test data]')

# score for post-processed forecasts
score_ghi_pp_test = crps_normal_censored(obs=obs_ghi_test, loc=mean_ghi_test, var=var_ghi_test, lb=0)
print ('The CRPS for forecasts with EMOS post-processing is: ' + str(score_ghi_pp_test) + ' [test data]')

# train data
obs_ghi_train = GHI_obs_train['GHI']

# score for raw forecasts
ens_ghi_train = GHI_ens_train.values
score_ghi_raw_train = em.crps_kernel (obs_ghi_train, ens_ghi_train)['crpsMean']
print ('The CRPS for ensemble forecasts with NO post-processing is: ' + str(score_ghi_raw_train) + ' [train data]')

# score for post-processed forecasts
score_ghi_pp_train = crps_normal_censored(obs=obs_ghi_train, loc=mean_ghi_train, var=var_ghi_train, lb=0)
print ('The CRPS for forecasts with EMOS post-processing is: ' + str(score_ghi_pp_train) + ' [train data]')

#--------------------------------------------Export results---------------------------------------------#
export_ghi_test = pd.DataFrame({'mean': mean_ghi_test,
                                'sd':   np.sqrt(var_ghi_test)
                     })
export_ghi_test.index = GHI_ens_test.index
export_ghi_test.to_csv(f'{results_path}\ghi_pp_EMOS_global_test.csv')

# export headline scores 
scores_test_ghi = pd.DataFrame({'method' : ['-', 'EMOS_global'],
                                'pp': ['raw', 'pp'],
                                'test_score' : [score_ghi_raw_test, score_ghi_pp_test],
                                'train_score': [score_ghi_raw_train, score_ghi_pp_train]})
scores_test_ghi.to_csv(f'{results_path}\scores_ghi_EMOS_global.csv', sep = ';', decimal = ',')


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
# combine train and test period
GHI_mean_all = np.append(GHI_ens_train_MEAN, GHI_ens_test_MEAN)
GHI_var_all = np.append(GHI_ens_train_VAR, GHI_ens_test_VAR)

for i in range(len(ENS_B)):											 
    mean = a + b * GHI_mean_all[i]
    sd = np.sqrt(c + d * GHI_var_all[i])
    ENS_B.iloc[i,0:50] = np.random.normal(mean, sd, 50)
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
# Define helping variables

# Mean of all ensemble members
PV_ens_train_MEAN = []; PV_ens_test_MEAN=[]
for n in range(2):
    PV_ens_train_MEAN.append(PV_ens_train[n].mean(axis=1).values)
    PV_ens_test_MEAN.append(PV_ens_test[n].mean(axis=1).values)

# Variance of the ensemble
PV_ens_train_VAR = []; PV_ens_test_VAR=[]
for n in range(2):
    PV_ens_train_VAR.append(PV_ens_train[n].var(axis=1).values)
    PV_ens_test_VAR.append(PV_ens_test[n].var(axis=1).values)


export_pv_ens_test = pd.DataFrame({'mean': PV_ens_test_MEAN[0],
                                   'sd':   np.sqrt(PV_ens_test_VAR[0])
                                  })
export_pv_ens_test.index = GHI_ens_test.index
export_pv_ens_test.to_csv(f'{results_path}\pv_ens_test.csv')

#-------------------------------------------Optimization step-------------------------------------------#
# Optimization objective: 4 parameters (a=x[0], b=x[1], c=x[2], d=x[3])
# Optimization function:  continous ranked probability score (CRPS)

a,b,c,d = [],[],[],[]
for n in range(2):
    def objective(x):
        return crps_normal_censored(obs=PV_obs_train,
                                    loc=x[0] + x[1]*PV_ens_train_MEAN[n],
                                    var=x[2] + x[3]*PV_ens_train_VAR[n],
                                    lb=0,
                                    ub=20)

    bounds = Bounds([-10, 0.001, -5, 0.00001],[100, 100, 100, 100])
    # initial guess of parameters to start optimization from
    x0 = np.array([1,1,1,1])
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, options={'ftol': 1e-9, 'maxiter': 200,  'disp': True})

    a.append(res.x[0])
    b.append(res.x[1])
    c.append(res.x[2])
    d.append(res.x[3])

# write out coeffs
for n in range(2):
    coeffs_pv=pd.DataFrame({'a': [a[n]], 'b': [b[n]], 'c':  [c[n]], 'd':  [d[n]]})
    if n == 0:
        coeffs_pv.to_csv(f'{training_path}/EMOS_coeffs/coeffs_emos_global_pv_raw_pp.csv', header=True)
    else:
        coeffs_pv.to_csv(f'{training_path}/EMOS_coeffs/coeffs_emos_global_pv_pp_pp.csv', header=True)

#--------------------------------------------Look at scores---------------------------------------------#

# Ultimately we have four models:
# A.A: raw GHI ensemble with raw PV ensemble
# A.B: raw GHI ensemble with post-processed PV ensemble
# B.A: post-processed GHI ensemble with raw PV ensemble
# B.B: post-processed GHI ensemble with post-processed PV ensemble

# test data
obs_pv_test = PV_obs_test
obs_pv_test.to_csv(f'{results_path}\pv_obs_test.csv')

# A.A:
ens_pv_test_raw = PV_ens_test[0].values
score_pv_test_rawraw = em.crps_kernel(obs_pv_test, ens_pv_test_raw)['crpsMean']
print ('The CRPS for forecasts with NO GHI post-processing and NO PV post-processing is: ' + str(score_pv_test_rawraw) + ' [test data]')

# A.B:
mean_pv_test_rawpp = a[0] + b[0]*PV_ens_test_MEAN[0]
var_pv_test_rawpp  = c[0] + d[0]*PV_ens_test_VAR[0]
score_pv_test_rawpp = crps_normal_censored(obs=obs_pv_test, loc=mean_pv_test_rawpp, var=var_pv_test_rawpp, lb=0, ub=20)
print ('The CRPS for forecasts with NO GHI post-processing and PV post-processing is: ' + str(score_pv_test_rawpp) + ' [test data]')

# B.A:
ens_pv_test_ppraw = PV_ens_test[1].values
score_pv_test_ppraw = em.crps_kernel (obs_pv_test, ens_pv_test_ppraw)['crpsMean']
print ('The CRPS for forecasts with GHI post-processing and NO PV post-processing is: ' + str(score_pv_test_ppraw) + ' [test data]')

# B.B:
mean_pv_test_pppp = a[1] + b[1]*PV_ens_test_MEAN[1]
var_pv_test_pppp  = c[1] + d[1]*PV_ens_test_VAR[1]
score_pv_test_pppp = crps_normal_censored(obs=obs_pv_test, loc=mean_pv_test_pppp, var=var_pv_test_pppp, lb=0, ub=20)
print ('The CRPS for forecasts with GHI post-processing and PV post-processing is: ' + str(score_pv_test_pppp) + ' [test data]')

#--------------------------------------------Export results---------------------------------------------#
export_pv_test_rawraw = pd.DataFrame.from_records(ens_pv_test_raw)
export_pv_test_rawraw.index = PV_obs_test.index
export_pv_test_rawraw.to_csv(f'{results_path}\pv_rawraw_test.csv')

export_pv_test_ppraw = pd.DataFrame.from_records(ens_pv_test_ppraw)
export_pv_test_ppraw.index = PV_obs_test.index
export_pv_test_ppraw.to_csv(f'{results_path}\pv_ppraw_EMOS_global_test.csv')

export_pv_test_rawpp = pd.DataFrame({'mean': mean_pv_test_rawpp,
                                     'sd':   np.sqrt(var_pv_test_rawpp)
                                    })
export_pv_test_rawpp.index=PV_ens_test[0].index
export_pv_test_rawpp.to_csv(f'{results_path}\pv_rawpp_EMOS_global_test.csv')

export_pv_test_pppp = pd.DataFrame({'mean': mean_pv_test_pppp,
                                    'sd':   np.sqrt(var_pv_test_pppp)
                                    })
export_pv_test_pppp.index=PV_ens_test[0].index
export_pv_test_pppp.to_csv(f'{results_path}\pv_pppp_EMOS_global_test.csv')

#---------------------------------------- Look at train scores -----------------------------------------# 
# train data
obs_pv_train = PV_obs_train

# A.A:
ens_pv_train = PV_ens_train[0].values
score_pv_train_rawraw = em.crps_kernel (obs_pv_train, ens_pv_train)['crpsMean']
print ('The CRPS for forecasts with NO GHI post-processing and NO PV post-processing is: ' + str(score_pv_train_rawraw) + ' [train data]')

# A.B:
mean_pv_train_rawpp = a[0] + b[0]*PV_ens_train_MEAN[0]
var_pv_train_rawpp  = c[0] + d[0]*PV_ens_train_VAR[0]
score_pv_train_rawpp = crps_normal_censored(obs=obs_pv_train, loc=mean_pv_train_rawpp, var=var_pv_train_rawpp, lb=0, ub=20)
print ('The CRPS for forecasts with NO GHI post-processing and PV post-processing is: ' + str(score_pv_train_rawpp) + ' [train data]')

# B.A:
ens_pv_train_ppraw = PV_ens_train[1].values
score_pv_train_ppraw = em.crps_kernel (obs_pv_train, ens_pv_train_ppraw)['crpsMean']
print ('The CRPS for forecasts with GHI post-processing and NO PV post-processing is: ' + str(score_pv_train_ppraw) + ' [train data]')

# B.B:
mean_pv_train_pppp = a[1] + b[1]*PV_ens_train_MEAN[1]
var_pv_train_pppp  = c[1] + d[1]*PV_ens_train_VAR[1]
score_pv_train_pppp = crps_normal_censored(obs=obs_pv_train, loc=mean_pv_train_pppp, var=var_pv_train_pppp, lb=0, ub=20)
print ('The CRPS for forecasts with GHI post-processing and PV post-processing is: ' + str(score_pv_train_pppp) + ' [train data]')

#--------------------------------------------Export results---------------------------------------------#
scores_pv = pd.DataFrame({'method' : ['-', 'EMOS_global', 'EMOS_global', 'EMOS_global'],
                        'pp': ['rawraw', 'rawpp', 'ppraw', 'pppp'],
                        'test_score' : [score_pv_test_rawraw, score_pv_test_rawpp, score_pv_test_ppraw, score_pv_test_pppp],
                        'train_score': [score_pv_train_rawraw, score_pv_train_rawpp, score_pv_train_ppraw, score_pv_train_pppp]})
scores_pv.to_csv(f'{results_path}\scores_pv_EMOS_global.csv', sep = ';', decimal = ',')
