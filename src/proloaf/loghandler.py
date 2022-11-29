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
Provides functions for logging training results and reading and writing those logs to/from csv or tensorboard files.
"""
from typing import Any, Dict, Union
import pandas as pd
import shutil
import torch
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from proloaf.confighandler import *
from proloaf.event_logging import create_event_logger

logger = create_event_logger(__name__)

## TensorBoard ##


def clean_up_tensorboard_dir(run_dir):
    """
    Move hyperparameter logs out of subfolders

    Iterate over all subdirectories of run_dir, moving any files they contain into run_dir.
    Neither sub-subdirectories nor their contents are moved. Delete them along with the subdirectories.

    Parameters
    ----------
    run_dir : string
        The path to the run directory

    Returns
    -------
    No return value
    """
    # move hparam logs out of subfolders
    subdir_list = [
        x for x in os.listdir(run_dir) if os.path.isdir(os.path.join(run_dir, x))
    ]  # gets a list of all subdirectories in the run directory

    for dir in subdir_list:
        subdir = os.path.join(run_dir, dir)  # complete path from root to current subdir
        files = [
            x for x in os.listdir(subdir) if os.path.isfile(os.path.join(subdir, x))
        ]  # gets all files in the current subdir
        for f in files:
            shutil.move(
                os.path.join(subdir, f), run_dir
            )  # moves the file out of the subdir
        # shutil.rmtree(subdir)  # removes the now empty subdir
        # !! only files located directly in the subdir are moved, sub-subdirs are not iterated and deleted with all their content !!


def log_tensorboard(
    work_dir: str,
    logname: str = str(datetime.now()).replace(":", "-").split(".")[0],
    trial_id: str = "main_run",
    **_,
):
    """
    Log the training performance using TensorBoard's SummaryWriter

    Parameters
    ----------
    work_dir: str
        Main folder in respect to which the tensorboard log will be saved.
    logname: str, default = datetime string for now
        Name for the sub folder of this training (`<work_dir>/runs/<logname>/).
    trial_id: str = "main_run"
        Can be used to distinguish multiple logs from the same training.
    Returns
    -------
    torch.utils.tensorboard.SummeryWriter
    """
    # log_name default if no run_name is given in command line with --logname, use timestamp

    # if exploration is True, don't save each trial in the same folder, that confuses tensorboard.
    # Instead, make a subfolder for each trial and name it Trial_{ID}.
    # If Trial ID > n_trials, actual training has begun; name that folder differently
    # if exploration:

    logname = os.path.join(logname, f"trial_{trial_id}")
    run_dir = os.path.join(work_dir, "runs", logname)
    tb = SummaryWriter(log_dir=run_dir)

    logger.info("Tensorboard log here:\t {!s}".format(tb.log_dir))
    return tb


def add_tb_element(
    net: torch.nn.Module,
    tb: SummaryWriter,
    epoch_loss: float,
    validation_loss: float,
    training_start: float,
    epoch_stop: float,
    epoch_start: float,
    next_epoch: int,
    step_counter: int,
):
    """
    Add scalars using TensorBoard's SummaryWriter. Corresponds to logging the status of one epoch.
    Logs warning if unable to use TensorBoard.

    Parameters
    ----------
    net: torch.nn.Module
        Model to be logged.
    tb: SummaryWriter
        SummaryWriter to be used in logging.
    epoch_loss: float
        In sample error (training error) achieved in the epoch.
    validation_loss: float
        Out of sample error (validation error) achieved in the epoch.
    training_start: float
        Value of `perf_counter()` from the start of training.
    epoch_stop: float
        Value of `perf_counter()` from the stop of the logged epoch.
    epoch_start: float
        Value of `perf_counter()` from the start of the logged epoch.
    next_epoch: int
        Number of the next epoch (or current if counting starts at 1 instead of 0)
    step_counter: int,
        Total number of optimization steps done during the training so far
    Returns
    -------
    SummaryWriter
        The SummaryWriter given in the input
    """
    tb.add_scalar("train_loss", epoch_loss, next_epoch)
    tb.add_scalar("val_loss", validation_loss, next_epoch)
    tb.add_scalar("train_time", epoch_stop - epoch_start, next_epoch)
    tb.add_scalar("total_time", epoch_stop - training_start, next_epoch)
    tb.add_scalar("val_loss_steps", validation_loss, step_counter)

    for name, weight in net.named_parameters():
        try:
            tb.add_histogram(name, weight, next_epoch)
        except:
            logger.warning(f"{name} could not be logged to tensorboard")
        try:
            tb.add_histogram(f"{name}.grad", weight.grad, next_epoch)
        except:
            logger.warning(f"{name}.grad could not be logged to tensorboard")
        # .add_scalar(f'{name}.grad', weight.grad, epoch + 1)
    return tb


def end_tensorboard(
    tb: SummaryWriter,
    params: Dict[str, Any],
    metric_dict: Dict[str, Union[bool,str,int,float,None]],
):
    """
    Wrap up TensorBoard's SummaryWriter

    Parameters
    ----------
    tb: SummaryWriter
        Tensorboard SummaryWriter to be cleaned up
    params: Dict[str, Any]
        Model and Trainingsettings, needs to conform to the config structur used in the whole ProLoaF project.
    metric_dict: Dict[str, Union[bool,str,int,float,None]],
        Simple dict with additional metrics to be logged, e.g. Errors, trainingtime etc. 
    """
    # TODO: we could add the fc_evaluate images and metrics to tb to inspect the best model here.
    # tb.add_figure(f'{hour}th_hour-forecast')
    # update this to use run_name as soon as the feature is available in pytorch (currently nightly at 02.09.2020)
    # https://pytorch.org/docs/master/tensorboard.html
    if params:
        key = "model_parameters"
        hparams = params.pop(key, None)

        if hparams:
            model_type = params.get("model_class",next(iter(hparams.keys())))
            params.update(hparams[model_type])

        tb.add_hparams(params, metric_dict)
    tb.close()
    # run_dir = os.path.join(work_dir, str("runs/" + logname))
    clean_up_tensorboard_dir(tb.get_logdir())


## DataFrame Logging ##


# Loads a logfile if one exists, else creates a pd dataframe (create log)
def create_log(log_path=None, station_path=None) -> pd.DataFrame:
    """
    Provide a pandas.DataFrame to use for logging the training results.

    Load a .csv log file, if one exists, into the DataFrame.
    If no .csv file exists, create the DataFrame using the parameters stored in the log.json config.

    Parameters
    ----------
    log_path : string, default = None
        Path to where logfiles are stored. If None, a new DataFrame will be created
    station_path : string, default = None
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
        feature_list = [
            x["name"]
            for x in read_config(config_path=os.path.join(station_path, "log.json"))[
                "features"
            ]
        ]
    except:
        logger.error("--- Couldn't load log feature set from config file ---")
        return None

    try:
        logger.info("--- Loading existing log file ---")
        log_df = pd.read_csv(log_path, sep=";", index_col=0)
        features_to_add_list = [x for x in feature_list if x not in list(log_df.head())]
        if features_to_add_list:
            log_df = pd.concat([log_df, pd.DataFrame(columns=features_to_add_list)])

    except:
        logger.info("--- No existing log file found, creating... ---")
        log_df = pd.DataFrame(columns=feature_list)

    return log_df


# TODO Unused? only for df logging.
def log_data(
    PAR,
    ARGS,
    LOSS_OPTIONS,
    total_time,
    final_epoch_loss,
    val_loss_min,
    score,
    log_df=None,
) -> pd.DataFrame:
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
    log_df : pandas.DataFrame, default = None
        The DataFrame that will be written to

    Returns
    -------
    pandas.DataFrame
        The DataFrame that has been written to
    """
    if log_df is not None:
        logger.info("--saving to log--")
        # Fetch all data that is of interest
        logdata_complete = {
            "time_stamp": pd.Timestamp.now(),
            "train_loss": final_epoch_loss,
            "val_loss": val_loss_min,
            "total_time": total_time,
            "activation_function": "leaky_relu",
            "cuda_id": PAR["cuda_id"],
            "max_epochs": PAR["max_epochs"],
            "lr": PAR["learning_rate"],
            "batch_size": PAR["batch_size"],
            "train_split": PAR["train_split"],
            "validation_split": PAR["validation_split"],
            "history_horizon": PAR["history_horizon"],
            "forecast_horizon": PAR["forecast_horizon"],
            "cap_limit": PAR["cap_limit"],
            "drop_lin": PAR["dropout_fc"],
            "drop_core": PAR["dropout_core"],
            "core": PAR["core_net"],
            "core_layers": PAR["core_layers"],
            "core_size": PAR["rel_core_hidden_size"],
            "optimizer_name": PAR["optimizer_name"],
            "activation_param": PAR["relu_leak"],
            "lin_size": PAR["rel_linear_hidden_size"],
            "scalers": PAR["feature_groups"],
            "encoder_features": PAR["encoder_features"],
            "decoder_features": PAR["decoder_features"],
            "station": ARGS.station,
            "criterion": ARGS.loss,
            "loss_options": LOSS_OPTIONS,
            "score": score,
        }
        logdata_selected = {
            x: logdata_complete[x] for x in list(log_df.columns)
        }  # Filter data and only log features defined in log.json
        log_df = log_df.append(logdata_selected, ignore_index=True)
    return log_df


# TODO unused? only for df logging
def write_log_to_csv(log_df, path=None, model_name=None):
    """
    Receive a pandas.DataFrame and save it as a .csv file using the given path.

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
            logger.info("saving log")
            log_df.to_csv(os.path.join(path, model_name), sep=";")


# TODO unused? Only for DF logging
def init_logging(
    model_name,
    work_dir,
    config,
):
    """
    Init train log process

    Start train log process, by storing a training log file in .CSV format in the local working directory.

    Parameters
    ----------
    work_dir : string
        The path to the user directory

    Returns
    -------
    log_df, TODO: describe further
    """
    path = os.path.join(
        work_dir,
        config["log_path"],
        config["model_name"],
        config["model_name"] + "_training.csv",
    )

    log_df = create_log(
        os.path.join(
            work_dir,
            config["log_path"],
            config["model_name"],
            config["model_name"] + "_training.csv",
        ),
        os.path.join(work_dir, "targets", model_name),
    )
    return log_df


def end_logging(
    model_name,
    work_dir,
    log_path,
    df,
):
    write_log_to_csv(
        df,
        os.path.join(work_dir, log_path, model_name),
        model_name + "_training.csv",
    )
