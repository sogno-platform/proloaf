# -*- coding: utf-8 -*-
"""
Sarimax evaluate Parameter und Functionen
"""

import json
import os
import sys
import warnings
import argparse

import pandas as pd
import torch
import numpy as np

torch.set_printoptions(linewidth=120) # Display option for output
torch.set_grad_enabled(True)
# from torch.utils.tensorboard import SummaryWriter
from time import perf_counter,localtime
from itertools import product

MAIN_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(MAIN_PATH)

import plf_util.datatuner as dt
import plf_util.tensorloader as dl
import plf_util.fc_network as fc_net
import plf_util.modelhandler as mh
import plf_util.parameterhandler as ph
import plf_util.eval_metrics as metrics
from plf_util.config_util import read_config, write_config, parse_with_loss

from fc_evaluate import results_table, evaluate_hours, plot_metrics

import copy
import matplotlib.pyplot as plt
import pandas as pd
# from sklearn.metrics import r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.sarimax import SARIMAXResults
# from sklearn.metrics import mean_squared_error


class flag_and_store(argparse._StoreAction):
    def __init__(self, option_strings, dest, dest_const, const, nargs = 0, **kwargs):
        self.dest_const = dest_const
        if isinstance(const, list):
            self.val = const[1:]
            self.flag = const[0]
        else:
            self.val = []
            self.flag = const
        if isinstance(nargs, int):
            nargs_store = nargs+len(self.val)
        else:
            nargs_store = nargs
        super(flag_and_store, self).__init__(option_strings, dest, const = None, nargs = nargs_store, **kwargs)
        self.nargs = nargs

    def __call__(self, parser, namespace, values, option_strings = None):
        setattr(namespace, self.dest_const,self.flag)
        if isinstance(values,list):
            self.val.extend(values)
        elif values is not None:
            self.val.append(values)

        if len(self.val) == 1:
            self.val = self.val[0]
        super().__call__(parser, namespace, self.val, option_strings)

# =============================================================================
# import data
# =============================================================================

#df = pd.read_csv(r'C:\Users\Harman\OneDrive - rwth-aachen.de\Work\Arbeit\Aufgabe_Sarimax\fc\data\last_sege.csv', sep=';',index_col=0)

# =============================================================================
# Parameter TODO: Remove params
# =============================================================================

#save_path = '..'

#load_path = '..'

# forecast_time = '2017-02-28 09:00:00'

# window_size = 200 #8760 #4272 #None

# target = ['Load']

# season = 'winter' #'sommer' 'winter' 'gesamt

# time_steps = 39

# my_order = (1,0,3)

# my_seasonal_order = (2,1,1,24)

# trend = 'n'

# number_forecasts = 4

# intervall = 168

# exog = ['limit_exceeded_D-1', 'Production 1'] #Auswahl externer Features

# alpha = 1.96

# =============================================================================
# daten strukturieren für mehrere forecasts
# =============================================================================

def constructDf(data, columns, season, train_split, time_steps, number_forecasts, intervall, history_horizon):

    """
    Construcs and reorders data for training

    Parameters
    ----------
    data                     : DataFrame with all data

    columns                  : columns used in model target+exog

    forecast_time            : str Time where forecasts should start

    time_steps               : number of time_steps

    number_forecasts         : number od forecasts

    intervall                : number of time_steps between every forecast

    window_size              : number of past time_steps if None all past data is used


    Returns
    -------
    1)input_matrix          : list of DataFrames for input
    2)output_matrix         : list of DataFrames for output

    """

    input_matrix = []
    output_matrix = []

    forecast_start = int(train_split * len(data))

    if number_forecasts == 0:
        number_forecasts = len(data)

    if history_horizon==0:
        if season=='gesamt':
            df = data[columns]
            for i in range(number_forecasts):
                new_input = df.iloc[:forecast_start+1 + intervall*i]
                new_output = df.iloc[forecast_start+1 + intervall*i:forecast_start+1 + intervall*i + time_steps]
                if len(new_output)<time_steps:
                    break
                input_matrix.append(new_input)
                output_matrix.append(new_output)

        elif season=='winter':
            df = data[columns].get(data['s_w']==0)
            for i in range(number_forecasts):
                new_input = df.iloc[:forecast_start+1 + intervall*i]
                new_output = df.iloc[forecast_start+1 + intervall*i:forecast_start+1 + intervall*i + time_steps]
                if len(new_output)<time_steps:
                    break
                input_matrix.append(new_input)
                output_matrix.append(new_output)

        elif season=='sommer':
            df = data[columns].get(data['s_w']==1)
            for i in range(number_forecasts):
                new_input = df.iloc[:forecast_start+1 + intervall*i]
                new_output = df.iloc[forecast_start+1 + intervall*i:forecast_start+1 + intervall*i + time_steps]
                if len(new_output)<time_steps:
                    break
                input_matrix.append(new_input)
                output_matrix.append(new_output)

    elif history_horizon!=0:
        window = forecast_start - history_horizon
        if window < 0:
            window = 0

        if season=='gesamt':
            df = data[columns]
            for i in range(number_forecasts):
                new_input = df.iloc[window:forecast_start+1 + intervall*i]
                new_output = df.iloc[forecast_start+1 + intervall*i:forecast_start+1 + intervall*i + time_steps]
                if len(new_output)<time_steps:
                    break
                input_matrix.append(new_input)
                output_matrix.append(new_output)

        elif season=='winter':
            df = data[columns].get(data['s_w']==0)
            for i in range(number_forecasts):
                new_input = df.iloc[window:forecast_start+1 + intervall*i]
                new_output = df.iloc[forecast_start+1 + intervall*i:forecast_start+1 + intervall*i + time_steps]
                if len(new_output)<time_steps:
                    break
                input_matrix.append(new_input)
                output_matrix.append(new_output)

        elif season=='sommer':
            df = data[columns].get(data['s_w']==1)
            for i in range(number_forecasts):
                new_input = df.iloc[window:forecast_start+1 + intervall*i]
                new_output = df.iloc[forecast_start+1 + intervall*i:forecast_start+1 + intervall*i + time_steps]
                if len(new_output)<time_steps:
                    break
                input_matrix.append(new_input)
                output_matrix.append(new_output)

    return input_matrix, output_matrix


# =============================================================================
# train model
# =============================================================================

def train_SARIMAX(input_matrix, target, output_matrix, exog, order, seasonal_order, trend, number_set=0):

    """
    Trains a SRAMIAX model

    Parameters
    ----------
    input_matrix            : List of input values

    output_matrix           : List of true output values Values

    target                  : str target column

    exog                    : str features

    order                   : order of model

    seasonal_order          : seasonal order of model

    trend                   : trend method used for training

    time_steps              : number of time_steps

    number_set              : choose index of input_matrix and output_matrix for training

    Returns
    -------
    1)fitted                : trained model instance
    2)model                 : model instance

    """

    model = SARIMAX(input_matrix[number_set][target],
                    exog = input_matrix[number_set][exog],
                    order=order, seasonal_order=seasonal_order, trend=trend,
    #                simple_differencing=True,
    #                enforce_stationarity=False,
    #                enforce_invertibility=False,
    #                mle_regression=False
                    )
    fitted = model.fit(disp=-1)
    #TODO: model performance can be described with fitted.llf too.
    return fitted, model, fitted.aic



# =============================================================================
# mehrere Forecasts mit selben Modell
# =============================================================================


def make_forecasts(input_matrix, output_matrix, target, exog, fitted, time_steps):

    """
    Calculates a numberr of forecasts

    Parameters
    ----------
    input_matrix            : List of input values

    output_matrix           : List of True output Values

    target                  : str target column

    exog                    : str features

    fitted                  : trained model

    time_steps              : number of time_steps

    Returns
    -------
    1)forecasts             : List of calculated forecasts

    """
    forecasts = []
    for i in range(len(input_matrix)):
        new_fitted = fitted.apply(input_matrix[i][target], input_matrix[i][exog]) # vielleicht mit i+1 statt i wenn es zu fehlern kommt
        forecast = new_fitted.forecast(time_steps, exog= output_matrix[i][exog])
        forecasts.append(forecast)
    return forecasts


# =============================================================================
# predictionintervall functions
# =============================================================================

def naive_predictionintervall(forecasts, output_matrix, target, sigma_factor, time_steps):

    """
    Calculates predictionintervall for given forecasts with Naive method https://otexts.com/fpp2/prediction-intervals.html

    Parameters
    ----------
    forecasts               : List of forecasts

    output_matrix           : List of True Values for given forecasts

    target                  : str target column

    alpha                   :

    time_steps              : number of time_steps

    Returns
    -------
    1)upper_limits          : List of upper intervall for given forecasts
    2)lower_limits          : List of lower intervall for given forecasts

    """

    upper_limits = []
    lower_limits = []

    for i in range(len(forecasts)):
        sigma = np.std(output_matrix[i][target].values-forecasts[i].values)
        sigma_h = []

        for r in range(time_steps):
            sigma_h.append(sigma * np.sqrt(r+1))

        sigma_hn = pd.Series(sigma_h)
        sigma_hn.index = forecasts[i].index

        fc_u = forecasts[i] + sigma_factor * sigma_hn
        fc_l = forecasts[i] - sigma_factor * sigma_hn

        upper_limits.append(fc_u)
        lower_limits.append(fc_l)

    return upper_limits, lower_limits

def apply_PI_params(model, fitted, input_matrix, output_matrix, target, exog):

    """
    Calculates custom predictionintervall --> Konfidenzintervalle werden für die Modellparameter. Parameterintervalle werden für forecasts genutzt.

    Parameters
    ----------
    model                   : model instance

    fitted                 : fitted model instance

    input_matrix           : list of DataFrames for input

    output_matrix          : list of DataFrames for output

    target                  : str target column

    exog                    : str features


    Returns
    -------
    1)upper_limits          : List of upper intervall for given forecasts
    2)lower_limits          : List of lower intervall for given forecasts

    """

    lower_limits = []
    upper_limits = []

    # params = fitted.params

    params_conf = fitted.conf_int()

    lower_params = params_conf[0]
    upper_params = params_conf[1]

    m_l = copy.deepcopy(fitted)
    m_u = copy.deepcopy(fitted)

    m_l.initialize(model=model, params=lower_params)
    m_u.initialize(model=model, params=upper_params)

    for i in range(len(input_matrix)):
        m_l = m_l.apply(input_matrix[i][target], exog = input_matrix[i][exog])
        fc_l = m_l.forecast(time_steps, exog= output_matrix[i][exog])
        lower_limits.append(fc_l)

        m_u = m_u.apply(input_matrix[i][target], exog = input_matrix[i][exog])
        fc_u = m_u.forecast(time_steps, exog= output_matrix[i][exog])
        upper_limits.append(fc_u)

    return upper_limits, lower_limits

# =============================================================================
# evaluate forecasts
# =============================================================================

def eval_SARIMX(forecasts, output_matrix, target, upper_limits, lower_limits):

    """
    Calculates evaluation mmetrics

    Parameters
    ----------
    forecasts               : List of calculated forecasts

    output_matrix           : list of DataFrames for output

    target                  : str target column

    upper_limits            : List of upper intervall for given forecasts

    lower_limits            : List of lower intervall for given forecasts


    Returns
    -------
    1)mse_horizon
    2)rmse_horizon
    4)sharpness_horizon
    5)coverage_horizon
    6)mis_horizon

    """

    forecasts = torch.tensor(forecasts)
    true_values = torch.tensor([i[target].T.values for i in output_matrix]).reshape(forecasts.shape).type(torch.FloatTensor)
    upper_limits = torch.tensor(upper_limits)
    lower_limits = torch.tensor(lower_limits)

    mse_horizon = metrics.mse(true_values, [forecasts], total=False)
    rmse_horizon = metrics.rmse(true_values, [forecasts], total=False)
    sharpness_horizon = metrics.sharpness(None,[upper_limits, lower_limits],total=False)
    coverage_horizon = metrics.picp(true_values, [upper_limits, lower_limits],total=False)
    mis_horizon = metrics.mis(true_values, [upper_limits, lower_limits], alpha=0.05,total=False)

    return mse_horizon, rmse_horizon, sharpness_horizon, coverage_horizon, mis_horizon

# #plot metrics
# plot_metrics(np.array(rmse_horizon), np.array(sharpness_horizon), np.array(coverage_horizon), np.array(mis_horizon), fc, 'metrics-evaluation')

# =============================================================================
# safe and load model
# =============================================================================

def save_SARIMAX(path, fitted):
    fitted.save(path)
    return

def load_SARIMAX(path):
    fitted = SARIMAXResults.load(path)
    return fitted

# =============================================================================
# test run
# =============================================================================


# input_matrix, output_matrix = constructDf(df, target+exog, 'gesamt', forecast_time, time_steps, number_forecasts, intervall, window_size)

# fitted, model = train_SARIMAX(input_matrix, target, output_matrix, exog, my_order, my_seasonal_order, trend)

# forecasts = make_forecasts(input_matrix, output_matrix, target, exog, fitted, time_steps)

# upper_limits, lower_limits = naive_predictionintervall(forecasts, output_matrix, target, alpha, time_steps)

# mse_horizon, rmse_horizon, sharpness_horizon, coverage_horizon, mis_horizon = eval_SARIMX(forecasts, output_matrix, target, upper_limits, lower_limits)

def main(infile, outmodel, target_column, logging_csv = False, log_path = None):
    # Read load data
    df = pd.read_csv(infile, sep=';',index_col=0)

    dt.fill_if_missing(df)

    if ('col_list' in PAR):
        if(PAR['col_list'] != None):
            df[target_column] = df[PAR['col_list']].sum(axis=1)

    # selected_features, scalers = dt.scale_all(df, **PAR) #TODO: nicht relevant für SARIMAX

    print('creating/reading parameters ...')

  #  if PAR['exploration']:
        # hyperparam_path = os.path.join(MAIN_PATH, PAR['exploration_config_path'])
        #parameter_combination = ph.make_from_config(model_name=ARGS.station, config_path=PAR['exploration_config_path'], main_path = MAIN_PATH)
        #TODO: apply parameter tuning script for SARIMAX,

    print('done')
    # print(parameter_combination.parameters)
    if logging_csv:
        try:
            log_df = pd.read_csv(os.path.join(MAIN_PATH, log_path), sep=';', index_col=0)
        except FileNotFoundError:
            log_df = pd.DataFrame(columns = ['time_stamp','train_loss','val_loss','tot_time','activation_function',
                                        'cuda_id','max_epochs','lr','batch_size', 'history_horizon','forecast_horizon',
                                        'core','optimizer','activation_param', 'drop_lin', 'drop_core','lin_size', 'core_size','station'])
    min_model = None
    if 'best_loss' in PAR and PAR['best_loss'] is not None:
        min_val_loss = PAR['best_loss']
    else:
        min_val_loss = np.inf
    try:
        if PAR['exploration']: #TODO: apply parameter tuning script for SARIMAX,
            for params in parameter_combination:
                print('training with ', params)
                PAR.update(params)
                net, val_loss, log_df = train(selected_features, scalers, log_df=log_df, **PAR)
                if min_val_loss > val_loss:
                    min_net = net
                    PAR['best_loss'] = val_loss
                    min_val_loss = val_loss
                    PAR.update(params)
        else:
            print('Read inputs')
            input_matrix, output_matrix = constructDf(df, PAR['target']+PAR['features'], PAR['season'], PAR['train_split'], PAR['time_steps'], PAR['number_forecasts'], PAR['intervall'], PAR['history_horizon'])
            print('training with standard hyper parameters')
            model, untrained_model, val_loss = train_SARIMAX(input_matrix, PAR['target'], output_matrix, PAR['features'],
                                                             (PAR['my_order'].get('p'),PAR['my_order'].get('d'),PAR['my_order'].get('q')),
                                                             (PAR['my_seasonal_order'].get('p'),PAR['my_seasonal_order'].get('d'),PAR['my_seasonal_order'].get('q'),PAR['my_seasonal_order'].get('s')),
                                                             PAR['trend'], number_set=0)
            if min_val_loss > val_loss:
                min_model = model
                PAR['best_loss'] = val_loss
                min_val_loss = val_loss
            forecasts = make_forecasts(input_matrix, output_matrix, PAR['target'], PAR['features'], model, PAR['time_steps'])
            if PAR['pi_method']=='Naive':
                upper_limits, lower_limits = naive_predictionintervall(forecasts, output_matrix, PAR['target'], PAR['sigma_factor'], PAR['time_steps'])
            elif PAR['pi_method']=='apply_PI_params':
                upper_limits, lower_limits = apply_PI_params(untrained_model, model, input_matrix, output_matrix, PAR['target'], PAR['features'])

            mse_horizon, rmse_horizon, sharpness_horizon, coverage_horizon, mis_horizon = eval_SARIMX(forecasts, output_matrix, PAR['target'], upper_limits, lower_limits)
            print(results_table(None, mse_horizon, rmse_horizon, sharpness_horizon, coverage_horizon, mis_horizon))
            #plot metrics
            OUTDIR = os.path.join(MAIN_PATH, PAR['EVALUATION_DIR'])
            plot_metrics(rmse_horizon.numpy(), sharpness_horizon.numpy(), coverage_horizon.numpy(), mis_horizon.numpy(), OUTDIR, 'metrics-evaluation')

    except KeyboardInterrupt:
        print()
        print('manual interrupt')
    finally:
        if min_model is not None:
            print('saving model')
            save_SARIMAX(outmodel,model)
            write_config(PAR, model_name = ARGS.station, config_path=ARGS.config, main_path=MAIN_PATH, suffix='SARIMAX')
            # parameter_combination.write_new_base()
            if logging_csv:
                print('saving log')
                log_df.to_csv(os.path.join(MAIN_PATH, log_path), sep=';')

if __name__ == '__main__':
    ARGS, LOSS_OPTIONS = parse_with_loss()
    PAR = read_config(model_name = ARGS.station, config_path=ARGS.config, main_path=MAIN_PATH, suffix='SARIMAX')
    main(infile = os.path.join(MAIN_PATH, PAR['data_path']), outmodel = os.path.join(MAIN_PATH, PAR['model_path']),
            target_column = PAR['target_column'], log_path=PAR['log_path'],logging_csv = True)
