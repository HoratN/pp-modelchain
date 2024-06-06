##############################################################################
#------------------------- load and preprocess data -------------------------#
##############################################################################

from datetime import *

from utils.helper_data import load_data, aggregate_clearsky_GHI, advance_GHI_ens
from utils.helper_data import compute_windspeed, align_time

data_path = "D:\PV_power_postprocessing\data"

def load_preprocess_data(data_path, advance = False):
    """
    load data, align time periods and compute wind speed
    If required, advances the GHI ensemble forecast by 30 minutes.
    """

    # load data
    McClear, GHI_obs, PV_obs, GHI_ens, weather_pred = load_data(data_path)
   
    McClear_agg_1h_raw, McClear_agg_1h_advance_30min = aggregate_clearsky_GHI(McClear)

    # advance GHI: used in the original code from Wang et al 2022 to align instantaneous wind speed and 
    # temperature forecasts from ECMWF with the corresponding GHI ensemble forecasts.
    # However, we decided to shift the data within the model chain and to not use the clear-sky GHI
    # adaptation for our study.
    if advance == True:
        GHI_ens = advance_GHI_ens(GHI_ens, McClear_agg_1h_raw, McClear_agg_1h_advance_30min)

    # align time periods of data sets
    GHI_obs, PV_obs, GHI_ens, weather_pred, McClear_agg_1h_raw, McClear_agg_1h_advance_30min = align_time(GHI_obs, PV_obs, GHI_ens, weather_pred, McClear_agg_1h_raw, McClear_agg_1h_advance_30min)

    # add wind speed as var to weather_pred
    weather_pred = compute_windspeed(weather_pred)

    return GHI_obs, PV_obs, GHI_ens, weather_pred