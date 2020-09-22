import pandas as pd
import os


# Loads a logfile if one exists, else creates a pd dataframe
def load_or_create_log(path = None) -> pd.DataFrame:
    try:
        log_df = pd.read_csv(path, sep=';', index_col=0)
        print("init try")
    except FileNotFoundError:
        log_df = pd.DataFrame(columns=['time_stamp', 'train_loss', 'val_loss', 'tot_time', 'activation_function',
                                       'cuda_id', 'max_epochs', 'lr', 'batch_size', 'train_split',
                                       'validation_split', 'history_horizon', 'forecast_horizon', 'cap_limit',
                                       'drop_lin', 'drop_core', 'core', 'core_layers', 'core_size', 'optimizer_name',
                                       'activation_param', 'lin_size', 'scalers', 'encoder_features', 'decoder_features',
                                       'station', 'criterion', 'loss_options', 'score'])
    return log_df


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
