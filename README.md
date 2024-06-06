# Improving Model Chain Approaches for Probabilistic Solar Energy Forecasting through Post-processing and Machine Learning

This repository provides Python code accompanying the paper
> Horat, N., Klerings S. and Lerch, S. (2024). Improving Model Chain Approaches for Probabilistic Solar Energy Forecasting through Post-processing and Machine Learning. arXiv
     

# Data
Our study is based on hourly data for the Jacumba Solar Project in southern California, U.S., covering the years 2017 to 2020.  
The large majority of the data is taken from [Wang et al. 2022](https://github.com/wentingwang94/probabilistic-solar-forecasting), GHI observations were downloaded from [NSRDB](https://nsrdb.nrel.gov/).

# Post-processing methods
We compare two post-processing methods, Ensemble Model Output Statistcs (EMOS) and machine learning-based distributional regression models. 
For each architecure, we train two variants: one that uses data from all hours of the day (global) and one trained separately for each hour of the day (hourly).

# Direct forecasting
As alternative to combining a model chain approach with post-processing, we train machine learning models that directly forecast PV production from GHI ensemble forecasts. 
As for the post-processing methods we train separate models for each hour of the day (hourly) and one joint model (global).

# Code
The *models* folder contains code for the different post-processing and direct forecasting methods. All helper functions are gathered in the folder *utils*.
