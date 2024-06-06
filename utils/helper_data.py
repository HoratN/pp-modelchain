##############################################################################
#---- helper functions to load and preprocess data (e.g. time alignment) ----#
##############################################################################
import pandas as pd
import numpy as np

data_path = "D:\PV_power_postprocessing\data"

def load_data(data_path):
    """
    Load all data and correctly assign the time stamps
    """

    # Clear-sky GHI
    McClear = pd.read_csv(f'{data_path}/McClear_Jacumba.csv', sep=';', decimal='.', usecols=[0,2])
    McClear.index = pd.date_range(start='2017-01-01 00:00:00', end='2020-12-31 23:45:00', freq='15min')

    # GHI observations
    GHI_obs = pd.read_csv(f'{data_path}/NSRDB_irradiance.csv', sep=';', decimal=',', usecols=[1])
    # update index to same format as above (stamp at the end of the time period, i.e. index = data.index + 30 min)
    # before start date was 2017-01-01 00:30, end 2020-12-31 23:30 
    GHI_obs.index = pd.date_range(start='2017-01-01 01:00:00', end='2021-01-01 00:00:00', freq='1h')
    GHI_obs = GHI_obs[:"2020-12-31 23:00:00"] # delete last entry
    

    PV_obs = pd.read_csv(f'{data_path}/60947.csv', index_col=0, sep=',', decimal='.', usecols=[0, 1])
    PV_obs.index = pd.date_range(start='2017-01-01 08:00:00', end='2021-01-01 07:00:00', freq='1h')
    PV_obs = PV_obs[:"2021-01-01 06:00:00"]
    # update index to same format as above (stamp at the end of the time period)
    # data is stamped at the beginning of the observation period: https://live-etabiblio.pantheonsite.io/sites/default/files/user_guide_for_data_file.pdf
    PV_obs.index = PV_obs.index + pd.Timedelta("1h")
    
    GHI_ens = pd.read_csv(f'{data_path}/Jacumba_ENS.csv', index_col=0, sep=',', decimal='.', usecols=list({i for i in range(1,52)}))
    GHI_ens.index = pd.date_range(start='2017-01-01 01:00:00', end='2020-12-31 23:00:00', freq='1h')

    weather_pred = pd.read_csv(f'{data_path}/ECMWF_HRES.csv', index_col=0, sep=',', decimal='.', usecols=[0,4,5,6]) 
    weather_pred.index = pd.date_range(start='2017-01-01 01:00:00', end='2020-12-31 23:00:00', freq='1h')

    return McClear, GHI_obs, PV_obs, GHI_ens, weather_pred


def aggregate_clearsky_GHI(McClear):
    """
    adapted from https://github.com/wentingwang94/probabilistic-solar-forecasting/blob/main/code/model%20chain.py

    aggregate Clearsky GHI observations to hourly values
    
    Idea: create two dataframes: McClear_agg_1h_raw and McClear_agg_1h_advance_30min
    McClear_agg_1h_raw:           aggregate 00:00:00, 00:15:00, 00:30:00, 00:45:00 to 01:00:00
    McClear_agg_1h_advance_30min: aggregate 00:30:00, 00:45:00, 01:00:00, 01:15:00 to 01:00:00
    
    McClear_agg_1h_raw is used to standardize the GHI ensembles
    McClear_agg_1h_advance_30min will be used to advance the ensembles in advance_GHI_ens
    (GHI-Ensemble / McClear_agg_1h_raw) * McClear_agg_1h_advance_30min
    
    """

    McClear_agg_1h_raw = McClear.copy()
    McClear_agg_1h_raw = McClear_agg_1h_raw.resample("1h").sum()              # rounds 'down':                00:45 -> 00:00 
    McClear_agg_1h_raw.index = McClear_agg_1h_raw.index + pd.Timedelta("1h")  # add 1 hrs to all data points: 00:45 -> 01:00
    McClear_agg_1h_raw = McClear_agg_1h_raw[0:-1]

    McClear_advance_30min = McClear.copy()
    McClear_advance_30min.index = McClear_advance_30min.index - pd.Timedelta("30min")            # move index back 30 min -> data is 'in the future'
    McClear_agg_1h_advance_30min = McClear_advance_30min.resample("1h").sum()                    # rounds 'down':                00:45 -> 00:30
    McClear_agg_1h_advance_30min.index = McClear_agg_1h_advance_30min.index + pd.Timedelta("1h") # add 1 hrs to all data points: 00:45 -> 01:30
    # delete first and last index as we miss information about 23:30 and 23:45 on the first day and about 00:00 and 00:15 on the last day
    McClear_agg_1h_advance_30min = McClear_agg_1h_advance_30min[1:-1]
    
    return McClear_agg_1h_raw, McClear_agg_1h_advance_30min


def advance_GHI_ens(GHI_ens, McClear_agg_1h_raw, McClear_agg_1h_advance_30min):
    """
    adapted from https://github.com/wentingwang94/probabilistic-solar-forecasting/blob/main/code/model%20chain.py

    Use clearsky GHI data to advance the ensemble:
    (GHI-Ensemble / McClear_agg_1h_raw) * McClear_agg_1h_advance_30min

    Returns:
    GHI_ens: dataframe, contains advanced GHI ensemble data
    """

    for i in range (50):
        GHI_ens.iloc[:,i] = GHI_ens.iloc[:,i] / McClear_agg_1h_raw['Clear sky GHI'] * McClear_agg_1h_advance_30min['Clear sky GHI']
    # Timestamp now is in the middle of the timespan (01:00 -> 00:30 ~ 01:29)
    
    # Replace all NaN values with 0
    # Due to clearsky GHI values of 0 during night, the ensemble takes on NaN values (during advancement happens a divide by 0 operation)
    GHI_ens = GHI_ens.replace(np.nan, 0)

    # Replace inf values with 0
    # During night, some GHI ensemble members might have single values that are slightly over 0, 
    # during the advancement, all other 0-value entries will take on NaN values (divide by 0 operation)
    # If the ensemble member is over 0 though, the value INF will be returned (but should also be 0)
    GHI_ens = GHI_ens.replace(np.inf, 0)

    return GHI_ens
    

def compute_windspeed(weather_pred):
    """
    taken from https://github.com/wentingwang94/probabilistic-solar-forecasting/blob/main/code/model%20chain.py

    Compute the windspeed from u10 and v10 (m/s):
    windspeed = sqrt(u10^2  + v10^2)
        
    as proposed by: Eq.(1) in https://doi.org/10.1016/j.solener.2021.12.011
    """ 
    weather_pred.insert(2, "wind_speed", np.sqrt((weather_pred.u10)**2 + (weather_pred.v10)**2))
    
    return weather_pred


def align_time(GHI_obs, PV_obs, GHI_ens, weather_pred, McClear_agg_1h_raw, McClear_agg_1h_advance_30min):
    """
    The first available PV power information are recorded for July 30, 2017 at 00 UTC when the Jacumba solar project was put into operation.
    Therefore we drop all days before that date and give an additional grace period of 1 day.
    
    """
    GHI_obs                        = GHI_obs["2017-07-31 00:00:00":"2020-12-31 23:00:00"]
    PV_obs                         = PV_obs["2017-07-31 00:00:00":"2020-12-31 23:00:00"]
    GHI_ens                        = GHI_ens["2017-07-31 00:00:00":"2020-12-31 23:00:00"]
    weather_pred                   = weather_pred["2017-07-31 00:00:00":"2020-12-31 23:00:00"]

    McClear_agg_1h_raw             = McClear_agg_1h_raw["2017-07-31 00:00:00":"2020-12-31 23:00:00"]
    McClear_agg_1h_advance_30min   = McClear_agg_1h_advance_30min["2017-07-31 00:00:00":"2020-12-31 23:00:00"]
    
    return GHI_obs, PV_obs, GHI_ens, weather_pred, McClear_agg_1h_raw, McClear_agg_1h_advance_30min
