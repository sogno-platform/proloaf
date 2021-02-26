# Copyright 2021 The ProLoaF Authors. All Rights Reserved.
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# ==============================================================================
"""
Provide functions for logging training results and reading and writing those logs to/from csv files.
"""

from plf_util.config_util import *

import pandas as pd
import os


# Loads a logfile if one exists, else creates a pd dataframe (create log)
def create_log(log_path=None, station_path=None) -> pd.DataFrame:
    """
    Provide a pandas.Dataframe to use for logging the training results.

    Load a .csv log file, if one exists, into the DataFrame.
    If no .csv file exists, create the DataFrame using the parameters stored in the log.json config.

    Parameters
    ----------
    log_path : string
        Path to where logfiles are stored
    station_path : string
        Path to where the log.json file is stored for a given station

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the loaded or created logfile

    Raises
    ------
    Bare except
        If unable to load log feature set from config file or if no existing log file found
    """
    try:
        feature_list = [x['name'] for x in read_config(config_path=os.path.join(station_path, "log.json"))['features']]
    except:
        print("--- Couldn't load log feature set from config file ---")
        return None

    try:
        print("--- Loading existing log file ---")
        log_df = pd.read_csv(log_path, sep=";", index_col=0)
        features_to_add_list = [x for x in feature_list if x not in list(log_df.head())]
        if features_to_add_list:
            log_df = pd.concat([log_df, pd.DataFrame(columns=features_to_add_list)])

    except:
        print("--- No existing log file found, creating... ---")
        log_df = pd.DataFrame(columns=feature_list)

    return log_df


    # try:
    #     log_df = pd.read_csv(path, sep=';', index_col=0)
    #     print("init try")
    # except FileNotFoundError:
    #     log_df = pd.DataFrame(columns=['time_stamp', 'train_loss', 'val_loss', 'tot_time', 'activation_function',
    #                                    'cuda_id', 'max_epochs', 'lr', 'batch_size', 'train_split',
    #                                    'validation_split', 'history_horizon', 'forecast_horizon', 'cap_limit',
    #                                    'drop_lin', 'drop_core', 'core', 'core_layers', 'core_size', 'optimizer_name',
    #                                    'activation_param', 'lin_size', 'scalers', 'encoder_features', 'decoder_features',
    #                                    'station', 'criterion', 'loss_options', 'score'])
    # return log_df


# recieves a df and some data to log and writes the data to that df
def log_data(PAR, ARGS, LOSS_OPTIONS, total_time, final_epoch_loss, val_loss_min, score, log_df = None) -> pd.DataFrame:
    """
    Recieve a pandas.DataFrame and some data to log and write the data to that DataFrame

    Only the features defined in the log.json config should be logged.

    Parameters
    ----------
    PAR : dict
        A dictionary containing all model parameters originally read from the config file
    ARGS : Namespace
        An object that stores the station name and loss function as attributes
    LOSS_OPTIONS : dict
        Contains extra options if the loss functions mis or quantile score are used
    total_time : float
        Total training time
    final_epoch_loss : float
        The value of the final epoch loss
    val_loss_min : float
        The value of the minimum validation loss
    score : torch.Tensor or float
        The score after training
    log_df : pandas.DataFrame
        The DataFrame that will be written to

    Returns
    -------
    pandas.DataFrame
        The DataFrame that has been written to
    """
    if log_df is not None:
        print("--saving to log--")
        # Fetch all data that is of interest
        logdata_complete = {
                'time_stamp': pd.Timestamp.now(),
                 'train_loss': final_epoch_loss,
                 'val_loss': val_loss_min,
                 'total_time': total_time,
                 'activation_function': 'leaky_relu',
                 'cuda_id': PAR["cuda_id"],
                 'max_epochs': PAR["max_epochs"],
                 'lr': PAR["learning_rate"],
                 'batch_size': PAR["batch_size"],
                 'train_split': PAR['train_split'],
                 'validation_split': PAR['validation_split'],
                 'history_horizon': PAR["history_horizon"],
                 'forecast_horizon': PAR["forecast_horizon"],
                 'cap_limit': PAR['cap_limit'],
                 'drop_lin': PAR["dropout_fc"],
                 'drop_core': PAR["dropout_core"],
                 'core': PAR["core_net"],
                 'core_layers': PAR['core_layers'],
                 'core_size': PAR["rel_core_hidden_size"],
                 'optimizer_name': PAR["optimizer_name"],
                 'activation_param': PAR["relu_leak"],
                 'lin_size': PAR["rel_linear_hidden_size"],
                 'scalers': PAR['feature_groups'],
                 'encoder_features': PAR['encoder_features'],
                 'decoder_features': PAR['decoder_features'],
                 'station': ARGS.station,
                 'criterion': ARGS.loss,
                 'loss_options': LOSS_OPTIONS,
                 'score': score
            }
        logdata_selected = {x: logdata_complete[x] for x in list(log_df.columns)}   # Filter data and only log features defined in log.json
        log_df = log_df.append(logdata_selected, ignore_index=True)
    return log_df


# recieves a df and tries to save that to a path, if the path is non-existent it gets created
def write_log_to_csv(log_df, path = None, model_name = None):
    """
    Recieve a pandas.DataFrame and save it as a .csv file using the given path.

    If the path doesn't exist, create it. If no path or no model name are given, do nothing.

    Parameters
    ----------
    log_df : pandas.DataFrame
        The DataFrame that should be saved as a .csv file
    path : string, default = None
        Path to where the log (.csv file) should be saved
    model_name : string, default = None
        The name of the resulting .csv file. Should include '.csv' file extension

    Returns
    -------
    No return value
    """
    if not log_df.empty:
        if path and model_name:
            if not os.path.exists(path):
                os.makedirs(path)
            log_df.to_csv(os.path.join(path, model_name), sep=';')
