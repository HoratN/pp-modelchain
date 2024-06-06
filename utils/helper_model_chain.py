#########################################################################################################
#--------------------------------- helper functions for model chain ------------------------------------#
# adapted from https://github.com/wentingwang94/probabilistic-solar-forecasting/blob/main/code/model%20chain.py
#--- all functions are modified to first shift the time stamp to the middle of the averaging window ----#
#---- and then apply the model chain operations --------------------------------------------------------#
#########################################################################################################

import pandas as pd

import pvlib
from pvlib.pvsystem import PVSystem
from pvlib.location import Location
from pvlib.modelchain import ModelChain

from datetime import *


def get_zenith_angle(GHI_obs):
    """
    returns zenith angle for the sun at Jacumba power plant.
    The time is derived from the time stamp of the input data.
    """
    # position of the sun
    # spa_python: the solar positionig algorithm (SPA) is commonly regarded as the most accurate one to date.
    position = pvlib.solarposition.spa_python(time=GHI_obs.index - pd.Timedelta("30min"), latitude=32.6193, longitude=-116.13) 
    position.index = position.index  + pd.Timedelta("30min");

    # the position of the sun is described by the solar azimuth and zenith angles.
    # Azimuth: to what side is the plant oriented? -> South: 0, East: 90, North: 180, West: 270
    # Zenith angle: on what angle does the sun shine on the horizontal ground? -> straight: 0, from the side: 90
    zenith_angle = pd.DataFrame(position.zenith)
    return zenith_angle


def separate_GHI(ens, member):
    """
    Apply separation model to split GHI into beam normal irradiation, DNI(=BNI), and diffuse horizontal irradiance, DHI.
    DNI and GHI are required to estimate the global tilted irradiance (GTI) in the model chain.
    """
    # Estimate DNI and DHI from GHI using the Erbs model.
    # The Erbs model estimates the diffuse fraction from global horizontal irradiance 
    # through an empirical relationship between DF and the ratio of GHI to extraterrestrial irradiance, Kt.
    sep = pvlib.irradiance.erbs(ghi=ens.iloc[:, member], zenith=ens.zenith, datetime_or_doy=ens.index)
    return sep


def input_model_chain(ens_, member, weather_pred):
    """
    Collect all information required as input to the model chain in one dataframe:
    DHI, DNI, wind speed, air temperature
    """

    ens = ens_.copy(deep = True)
    df = pd.DataFrame(columns=['ghi','dhi','dni','wind_speed','temp_air'], index=ens.index)

    # GHI (W/m2)
    df['ghi'] = ens.iloc[:,member].values

    ens.index = ens.index - pd.Timedelta("30min") # shift time stamp to middle of averaging window
    dhi_dni = separate_GHI(ens, member)
    # undo time stamp shift after applying the separation model
    ens.index = ens.index + pd.Timedelta("30min")
    dhi_dni.index = dhi_dni.index + pd.Timedelta("30min")

    # dhi (W/m2)
    df['dhi'] = dhi_dni['dhi'].values
    # dni (W/m2)
    df['dni'] = dhi_dni['dni'].values

    # wind_speed (m/s)
    df['wind_speed'] = weather_pred['wind_speed'].values
    # air_temp (℃)
    df['temp_air'] = weather_pred['t2m'].values

    return df


def run_model_chain(chain_input):
    """
    Set up model chain and apply it to chain_input, 
    which is a dataframe containing information on time, irradiance, wind and temperature.
    Create such a dataframe with the function input_model_chain.

    Return PV estimates for the Jacumba solar plant.
    """
    
    #----------------------------- Specify PV model for the Jacumba solar plant ----------------------------#

    # load some module and inverter specifications
    # 'CECMod': the CEC module database
    cec_modules = pvlib.pvsystem.retrieve_sam('CECMod')
    cec_module = cec_modules['Jinko_Solar_Co___Ltd_JKM350M_72B']
    # 'cecinverter': the CEC Inverter database
    cec_inverters = pvlib.pvsystem.retrieve_sam('cecinverter')
    cec_inverter = cec_inverters['SMA_America__SC_2200_US__385V_']
    # inverter parameters: https://www.sandiegocounty.gov/content/dam/sdc/pds/ceqa/JVR/AdminRecord/IncorporatedByReference/Section-2-9---Noise-References/SC2200-3000-EV-DS-en-59.pdf
    cec_inverter['Vdcmax'] = 1100
    cec_inverter['Idcmax'] = 3960

    # SET PARAMETERS
    array_kwargs = dict(module_parameters=cec_module,
                        temperature_model_parameters=dict(a=-3.56, b=-0.075, deltaT=3))

    # Location object as container for latitude, longitude, timezone, and altitude data
    location = Location(latitude=32.6193, longitude=-116.13, tz='UTC')
    # The angle is based on your latitude minus about 15 degrees.
    mount = pvlib.pvsystem.FixedMount(surface_tilt=32.6193-14.58, surface_azimuth=180)
    # https://www.gridinfo.com/plant/jacumba-solar-farm/60947
    # 28x224 total modules arranged in 224 strings of 28 modules each 
    arrays = [pvlib.pvsystem.Array(mount=mount,modules_per_string=28,strings=224,**array_kwargs)]
    # The 'PVSystem' represents one inverter and the PV modules that supply DC power to the inverter.
    system = PVSystem(arrays=arrays, inverter_parameters=cec_inverter)
 
    #----------------------------------------- Run the model chain -----------------------------------------#
    # initialize model chain
    mc = ModelChain(system, location, aoi_model='no_loss', spectral_model='no_loss', transposition_model='reindl')
    
    # Run multi-step model chain: 
    # most important components are separation - previously done and transposition - chosen with the parameters
    # use time shift to apply model chain to the time stamp pointing to the middle of the averaging window
    chain_input.index = chain_input.index - pd.Timedelta("30min")
    mc.run_model(chain_input)
    
    # output AC power (Alternating Current - Wechselstrom)
    results_ModelChain = mc.results.ac

    # Estimate the power output of the entire photovoltaic power station
    # Wang et al 2022 assume that the Jacumba solar plant consists of approximately 11.7 28x224 arrays.
    pv = results_ModelChain*11.7
    pv.index = pv.index + pd.Timedelta("30min")
    return pv

def cleanup_model_chain_output(pv_ens, zenith_angle):
        """
        Clean up output of model chain: convert to MW, make sure that upper production limit of 20MW is respected,
        and PV power estimate is only positive when the zenith angle is large enough.
        """
        # AC power should be within the rated capacity -> set anything below 0 to 0 and anything above 2000000 to 2000000
        pv_ens.where(pv_ens > 0, 0, inplace = True)
        pv_ens.where(pv_ens < 20000000, 20000000, inplace = True)

        # convert to MW Megawatt (1.000.000 Watt = 1 Megawatt)
        pv_ens = pv_ens/1000000
        
        # insert zenith angle into dataframe
        pv_ens.insert(0, "zenith", zenith_angle.zenith)

        # Set rows where zenith angle > 85° to 0 (irradiance too low to generate power)
        pv_ens.where(pv_ens['zenith'] < 85, 0, inplace = True)

        return pv_ens