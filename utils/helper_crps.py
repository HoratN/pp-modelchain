##############################################################################
#--------------- helper functions related to crps computation ---------------#
#----- and evaluation (central prediction intervals and rank histograms) ----#
##############################################################################
import numpy as np
import math

import tensorflow_probability as tfp
import tensorflow as tf
import keras.backend as K

from scipy.stats import norm
from datetime import *
import random

# CRPS for Ensembles: Use function em.crps_kernel(observation=obs, ensemble=ens)
# CRPS for censored normal distributions defined by distribution parameters: 
# crps_normal_censored and crps_normal_censored_separate_results

# Censored Normal Distribution
def crps_normal_censored (obs, loc, var, lb=np.nan, ub=np.nan):
    
    ####################################################################################################
    # Function that computes the CRPS of a prediction based on a censored normal distribution, it only
    # takes the provided bounds into account
    #--------------------------------------------------------------------------------------------------#
    # Inputs: obs: observation value
    #         loc: location parameter of the predictive distribution
    #         var: scale parameter (variance)
    #         lb:  lower bound (if not provided, default value: NaN)
    #         ub:  upper bound (if not provided, default value: NaN)
    #--------------------------------------------------------------------------------------------------#
    # Output: mean CRPS over all datapoints of the input
    ####################################################################################################
    
    sd = np.sqrt(var)
    XX = (obs - loc) / sd

    # No bounds provided
    if math.isnan(lb) & math.isnan(ub):
        z = XX
        part1 = 0
        part2 = z * (2*norm.cdf(z) - 1)
        part3 = 2*norm.pdf(z)
        part4 = (1 / np.sqrt(np.pi))
    
    # Only upper bound provided
    elif math.isnan(lb) & (not math.isnan(ub)):
        ub = (ub-loc) / sd
        z = np.minimum(ub, XX)
        part1 = abs(XX - z) + ub * norm.cdf(-ub)**2
        part2 = z * (2*norm.cdf(z) - 1)
        part3 = 2*norm.pdf(z) - 2*norm.pdf(ub)*norm.cdf(-ub)
        part4 = (1 / np.sqrt(np.pi)) * (norm.cdf(np.sqrt(2)*ub))
        
    # Only lower bound provided
    elif (not math.isnan(lb)) & math.isnan(ub):
        lb = (lb-loc) / sd
        z = np.maximum (XX, lb)
        part1 = abs(XX - z) - lb * norm.cdf(lb)**2
        part2 = z * (2*norm.cdf(z) - 1)
        
        part3 = 2*norm.pdf(z) - 2*norm.pdf(lb)*norm.cdf(lb)
        part4 = (1 / np.sqrt(np.pi)) * (1 - norm.cdf(np.sqrt(2)*lb))
        
    # Both upper and lower bounds provided
    else:
        lb = (lb-loc) / sd
        ub = (ub-loc) / sd
        z = np.maximum (lb, np.minimum(ub, XX))
        part1 = abs(XX - z) + ub * norm.cdf(-ub)**2 - lb * norm.cdf(lb)**2
        part2 = z * (2*norm.cdf(z) - 1)
        part3 = 2*norm.pdf(z) - 2*norm.pdf(ub)*norm.cdf(-ub) - 2*norm.pdf(lb)*norm.cdf(lb)
        part4 = (1 / np.sqrt(np.pi)) * (norm.cdf(np.sqrt(2)*ub) - norm.cdf(np.sqrt(2)*lb))
        
    return np.mean(np.sqrt(var) * (part1 + part2 + part3 - part4))


# Similar to the above formula but returns all CRPS values instead of the mean
def crps_normal_censored_separate_results (obs, loc, var, lb=np.nan, ub=np.nan):
    
    ####################################################################################################
    # Function that computes the CRPS of a prediction based on a censored normal distribution, it only
    # takes the provided bounds into account
    #--------------------------------------------------------------------------------------------------#
    # Inputs: obs: observation value
    #         loc: location parameter of the predictive distribution
    #         var: scale parameter (variance)
    #         lb:  lower bound (if not provided, default value: NaN)
    #         ub:  upper bound (if not provided, default value: NaN)
    #--------------------------------------------------------------------------------------------------#
    # Output: CRPS for each datapoint of the input
    ####################################################################################################
    
    sd = np.sqrt(var)
    XX = (obs - loc) / sd

    # No bounds provided
    if math.isnan(lb) & math.isnan(ub):
        z = XX
        part1 = 0
        part2 = z * (2*norm.cdf(z) - 1)
        part3 = 2*norm.pdf(z)
        part4 = (1 / np.sqrt(np.pi))
    
    # Only upper bound provided
    elif math.isnan(lb) & (not math.isnan(ub)):
        ub = (ub-loc) / sd
        z = np.minimum(ub, XX)
        part1 = abs(XX - z) + ub * norm.cdf(-ub)**2
        part2 = z * (2*norm.cdf(z) - 1)
        part3 = 2*norm.pdf(z) - 2*norm.pdf(ub)*norm.cdf(-ub)
        part4 = (1 / np.sqrt(np.pi)) * (norm.cdf(np.sqrt(2)*ub))
        
    # Only lower bound provided
    elif (not math.isnan(lb)) & math.isnan(ub):
        lb = (lb-loc) / sd
        z = np.maximum (XX, lb)
        part1 = abs(XX - z) - lb * norm.cdf(lb)**2
        part2 = z * (2*norm.cdf(z) - 1)
        part3 = 2*norm.pdf(z) - 2*norm.pdf(lb)*norm.cdf(lb)
        part4 = (1 / np.sqrt(np.pi)) * (1 - norm.cdf(np.sqrt(2)*lb))
        
    # Both upper and lower bounds provided
    else:
        lb = (lb-loc) / sd
        ub = (ub-loc) / sd
        z = np.maximum (lb, np.minimum(ub, XX))
        part1 = abs(XX - z) + ub * norm.cdf(-ub)**2 - lb * norm.cdf(lb)**2
        part2 = z * (2*norm.cdf(z) - 1)
        part3 = 2*norm.pdf(z) - 2*norm.pdf(ub)*norm.cdf(-ub) - 2*norm.pdf(lb)*norm.cdf(lb)
        part4 = (1 / np.sqrt(np.pi)) * (norm.cdf(np.sqrt(2)*ub) - norm.cdf(np.sqrt(2)*lb))
        
    return np.sqrt(var) * (part1 + part2 + part3 - part4)


# Custom loss function for training the neural networks
def custom_loss_crps_wrapper_fct (LB=np.nan, UB=np.nan):
    def custom_loss_crps (y_true, y_pred):
    
        ##########################################################################################################
        # Function that computes the CRPS of a possibly censored distribution for usage as a
        # loss function in a neural network
        #--------------------------------------------------------------------------------------------------------#
        # Inputs: LB, UB: lower and upper bounds (both optional)
        #         y_true: true values (observations)
        #         y_pred: Tensor containing [mean, sd] for each datapoint in the batch (represents a distribution)
        #--------------------------------------------------------------------------------------------------------#
        # Output: mean CRPS over the input batch
        ##########################################################################################################
        y_true = tf.squeeze(y_true)
        loc = tf.squeeze(y_pred[:, 0])
        sd = tf.squeeze(y_pred[:, 1])

        # CDF and PDF need to be newly defined to use them in a tensorflow environment
        def pdf (x):
            return 1 / (tf.sqrt(2.0*math.pi)) * tf.exp(-0.5 * x**2)
        cdf = tfp.distributions.Normal(0, 1).cdf

        # define further variables to make the computation of the CRPS easier
        XX = (y_true - loc) / sd
        
        # Cover all possible combinations of inputs        
        # No bounds provided
        if math.isnan(LB) & math.isnan(UB):
            z = XX
            part1 = 0
            part2 = z * (2*cdf(z) - 1)
            part3 = 2*pdf(z)
            part4 = (1 / tf.sqrt(math.pi))
    
        # Only upper bound provided
        elif math.isnan(LB) & (not math.isnan(UB)):
            ub = (UB-loc) / sd
            #ub = tf.cast(ub, tf.float32)
            z = tf.minimum(ub, XX)
            part1 = abs(XX - z) + ub * cdf(-ub)**2
            part2 = z * (2*cdf(z) - 1)
            part3 = 2*pdf(z) - 2*pdf(ub)*cdf(-ub)
            part4 = (1 / tf.sqrt(math.pi)) * (cdf(tf.sqrt(2.0)*ub))
        
        # Only lower bound provided
        elif (not math.isnan(LB)) & math.isnan(UB):
            lb = (LB-loc) / sd
            #lb = tf.cast(lb, tf.float32)
            z = tf.maximum (XX, lb)
            part1 = abs(XX - z) - lb * cdf(lb)**2
            part2 = z * (2*cdf(z) - 1)
            part3 = 2*pdf(z) - 2*pdf(lb)*cdf(lb)
            part4 = (1 / tf.sqrt(math.pi)) * (1 - cdf(tf.sqrt(2.0)*lb))
        
        # Both upper and lower bounds provided
        else:
            lb = (LB-loc) / sd
            #lb = tf.cast(lb, tf.float32)
            ub = (UB-loc) / sd
            #ub = tf.cast(ub, tf.float32)
            z = tf.maximum (lb, tf.minimum(ub, XX))
            part1 = abs(XX - z) + ub * cdf(-ub)**2 - lb * cdf(lb)**2
            part2 = z * (2*cdf(z) - 1)
            part3 = 2*pdf(z) - 2*pdf(ub)*cdf(-ub) - 2*pdf(lb)*cdf(lb)
            part4 = (1 / tf.sqrt(math.pi)) * (cdf(tf.sqrt(2.0)*ub) - cdf(tf.sqrt(2.0)*lb))

        return K.mean(sd * (part1 + part2 + part3 - part4))
    return custom_loss_crps


# CDF for censored normal distribution
def cdf_norm_cens (x, loc, sd, lb=np.nan, ub=np.nan, PIT=False):
    
    ####################################################################################################
    # Function that computes the cumulative distribution function (CDF) value of a censored normal
    # distribution, it only takes the provided bounds into account
    #--------------------------------------------------------------------------------------------------#
    # Inputs: x:   value for which the CDF should be computed
    #         loc: location parameter of the distribution
    #         sd:  scale parameter (standard deviation)
    #         lb:  lower bound (if not provided, default value: NaN)
    #         ub:  upper bound (if not provided, default value: NaN)
    #         PIT: if False, 0 or 1 will be returned for out-of-bounds-values,
    #              if True, a random value will be returned in the interval from the CDF of the bound
    #              until 0 or 1, this is needed if a PIT histogram is created so that not all values
    #              outside of the bounds return exactly 0 or 1
    #--------------------------------------------------------------------------------------------------#
    # Output: CDF value at point x
    ####################################################################################################
    
    # No Bounds provided
    if math.isnan(lb) & math.isnan(ub):
        return norm.cdf(x, loc, sd)
    
    # Only upper bound provided
    elif math.isnan(lb) & (not math.isnan(ub)):
        if x >= ub:
            if PIT == False:
                return 1
            else:
                return random.uniform(norm.cdf(ub, loc, sd), 1)
        else:
            return norm.cdf(x, loc, sd)
        
    # Only lower bound provided
    elif (not math.isnan(lb)) & math.isnan(ub):
        if x <= lb:
            if PIT == False:
                return 0
            else:
                return random.uniform(0, norm.cdf(lb, loc, sd))
        else:
            return norm.cdf(x, loc, sd)
        
    # Both upper and lower bounds provided
    else:
        if x >= ub:
            if PIT == False:
                return 1
            else:
                return random.uniform(norm.cdf(ub, loc, sd), 1)
        elif x <= lb:
            if PIT == False:
                return 0
            else:
                return random.uniform(0, norm.cdf(lb, loc, sd))
        else:
            return norm.cdf(x, loc, sd)

        
# PPF for censored normal distribution (inverse CDF)
def get_perc (perc, loc, sd, lb=np.nan, ub=np.nan):
    
    ####################################################################################################
    # Function that returns lower and upper bounds (percentiles) for the central prediction interval 
    # with probability given in perc.
    # The function computes the inverse cumulative distribution function (PPF) values of a censored
    # normal distribution, it only takes the provided bounds into account.
    # perc = 0 returns the median of the distribution twice.
    #--------------------------------------------------------------------------------------------------#
    # Inputs: perc: percentage that should be confined by the central prediction interval
    #         loc:  location parameter of the distribution
    #         sd:   scale parameter (standard deviation)
    #         lb:   lower bound (if not provided, default value: NaN)
    #         ub:   upper bound (if not provided, default value: NaN)
    #--------------------------------------------------------------------------------------------------#
    # Output: lower bound x of the interval
    #         upper bound x of the interval
    ####################################################################################################
    
    # compute lower and upper percentile
    lower_perc = 0.5 - perc/2
    upper_perc = 0.5 + perc/2
    
    # No Bounds provided
    if math.isnan(lb) & math.isnan(ub):
        return norm.ppf(lower_perc, loc, sd), norm.ppf(upper_perc, loc, sd)
    
    # Only upper bound provided
    elif math.isnan(lb) & (not math.isnan(ub)):
        ub_cdf = norm.cdf(ub, loc, sd)
        if ub_cdf > upper_perc:
            return norm.ppf(lower_perc, loc, sd), norm.ppf(upper_perc, loc, sd)
        elif ub_cdf > lower_perc:
            return norm.ppf(lower_perc, loc, sd), ub
        else:
            return ub, ub
        
    # Only lower bound provided
    elif (not math.isnan(lb)) & math.isnan(ub):
        lb_cdf = norm.cdf(lb, loc, sd)
        if lb_cdf < lower_perc:
            return norm.ppf(lower_perc, loc, sd), norm.ppf(upper_perc, loc, sd)
        elif lb_cdf < upper_perc:
            return lb, norm.ppf(upper_perc, loc, sd)
        else:
            return lb, lb
        
    # Both upper and lower bounds provided
    else:
        lb_cdf = norm.cdf(lb, loc, sd)
        ub_cdf = norm.cdf(ub, loc, sd)
        if lb_cdf < lower_perc:
            if ub_cdf > upper_perc:
                return norm.ppf(lower_perc, loc, sd), norm.ppf(upper_perc, loc, sd)
            elif ub_cdf > lower_perc:
                return norm.ppf(lower_perc, loc, sd), ub
            else:
                return ub, ub
        elif lb_cdf < upper_perc:
            if ub_cdf > upper_perc:
                return lb, norm.ppf(upper_perc, loc, sd)
            else:
                return lb, ub
        else:
            return lb, lb

        
# Similar to the above function but for ensembles percentiles
def get_perc_ens (perc, ens):
    
    ####################################################################################################
    # Function that returns lower and upper bounds (percentiles) for the central prediction interval 
    # with probability given in perc.
    #--------------------------------------------------------------------------------------------------#
    # Inputs: perc: percentage that should be confined by the central prediction interval
    #         ens:  the ensemble
    #--------------------------------------------------------------------------------------------------#
    # Output: lower bound x of the interval
    #         upper bound x of the interval
    ####################################################################################################
    
    lower_perc = 0.5 - perc/2
    upper_perc = 0.5 + perc/2
    
    lower = sorted(ens)[round(lower_perc * len(ens))]
    upper = sorted(ens)[round(upper_perc * len(ens))]
    return lower, upper   


# Computation of rank for verification rank histogram
def rank (ens, obs):
    
    ####################################################################################################
    # Function returns the bin the observation falls into for verification rank histogram
    #--------------------------------------------------------------------------------------------------#
    # Inputs: ens:  ensemble
    #         obs:  observation that should be ranked
    #--------------------------------------------------------------------------------------------------#
    # Output: rank: returns the index of the position the observation would be sorted in after sorting
    #               all ensemble members by value
    ####################################################################################################
    
    ranks = []
    for i in range(len(ens)):
        if ens[ens == obs[i]].count(axis=1)[i] > 1:
            min_rank = np.sum([member < obs[i] for member in ens.values[i]])
            max_rank = min_rank + ens[ens == obs[i]].count(axis=1)[i]
            ranks.append(random.randint(min_rank, max_rank))
        else:
            ranks.append(np.sum([member < obs[i] for member in ens.values[i]]))
    return ranks
