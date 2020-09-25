from plf_util.config_util import *

import pandas as pd
import os


# Loads a logfile if one exists, else creates a pd dataframe (create log)
def create_log(log_path=None, station_path=None) -> pd.DataFrame:
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
def log_data(data, log_df = None) -> pd.DataFrame:
    log_df = log_df.append(data, ignore_index=True)
    return log_df


# recieves a df and tries to save that to a path, if the path is nonexistant it gets created
def write_log_to_csv(log_df, path = None, model_name = None):
    if not log_df.empty:
        if path and model_name:
            if not os.path.exists(path):
                os.makedirs(path)
            log_df.to_csv(os.path.join(path, model_name), sep=';')
