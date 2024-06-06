##############################################################################
#------------------ helper functions for train and predict-------------------#
##############################################################################

import pandas as pd
import numpy as np

def train_test_split(data):
    # Train data (2017-2019)
    data_train = data[:"2019-12-31 23:00:00"]
    
    # Test data (2020)
    data_test = data["2020-01-01 00:00:00":]
    
    return data_train, data_test


def prepare_data_ghi(obs, fcst):
    """
    gather data for GHI post-processing with EMOS in one dataframe
    """
    df = pd.DataFrame()
    df['obs']     = obs.GHI
    df['hour_ID'] = obs.index.hour
    df['GHI_mean'] = fcst.mean(axis=1).values
    df['GHI_var'] = fcst.var(axis=1).values
    return df


def prepare_data_pv(obs, fcst):
    """
    gather data for PV post-processing with EMOS in one dataframe
    """
    df = pd.DataFrame()
    df['obs']     = obs
    df['hour_ID'] = fcst.index.hour
    df['PV_mean']  = fcst.mean(axis=1).values
    df['PV_var']  = fcst.var(axis=1).values
    df.index      = fcst.index
    return df


def prepare_data_ghi_nn(obs, fcsts, weather):
    """
    gather data for GHI post-processing with NNs in one dataframe
    """
    df = pd.DataFrame()
    df['obs']   = obs.GHI.values.astype(np.float32)
    df['loc']   = fcsts.mean(axis=1).values
    df['sd']    = fcsts.std(axis=1).values
    df['hour']  = fcsts.index.hour
    df['wind_speed'] = weather.wind_speed.values
    df['temp_air']   = weather.t2m.values
    return df


def prepare_data_pv_nn(obs, fcsts, weather):
    """
    gather data for PV post-processing with NNs in one dataframe
    """
    df = pd.DataFrame()
    df['obs']   = obs['SAM_gen'].values.astype(np.float32)
    df['loc']   = fcsts.mean(axis=1).values
    df['sd']    = fcsts.std(axis=1).values
    df['hour']  = fcsts.index.hour
    df['wind_speed'] = weather.wind_speed.values
    df['temp_air']   = weather.t2m.values
    return df


def standardize_predictors(df_train, df_test):
    """
    standardize the predictors for the NN models
    Note that for the hourly NNs, we standardize the predictors separately for each hour.
    """
    mean = df_train.mean()
    sd   = df_train.std()

    df_train['loc']        = (df_train['loc']        - mean['loc'])        / sd['loc']
    df_train['sd']         = (df_train['sd']         - mean['sd'])         / sd['sd']
    df_train['wind_speed'] = (df_train['wind_speed'] - mean['wind_speed']) / sd['wind_speed']
    df_train['temp_air']   = (df_train['temp_air']   - mean['temp_air'])   / sd['temp_air']

    df_test['loc']        = (df_test['loc']        - mean['loc'])        / sd['loc']
    df_test['sd']         = (df_test['sd']         - mean['sd'])         / sd['sd']
    df_test['wind_speed'] = (df_test['wind_speed'] - mean['wind_speed']) / sd['wind_speed']
    df_test['temp_air']   = (df_test['temp_air']   - mean['temp_air'])   / sd['temp_air']
    return df_train, df_test