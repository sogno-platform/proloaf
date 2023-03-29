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
Train an RNN model for load forecasting based on provided data.

Train an RNN model on prepared data loaded as pandas dataframe from a csv file. 
Hyperparameter exploration using optuna is also possible if desired. 
The trained model is saved at the location specified under "output_path": in the corresponding 
config.json and can be loaded via torch.load() or evaluated by using the evaluate.py script.
This script scales the data, loads a custom datastructure and then generates and trains a neural net.

Notes
-----

"""

from functools import partial
import os
import sys
from typing import Callable

import warnings
from copy import deepcopy

import pandas as pd
from sklearn.utils import validation
import torch


MAIN_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(MAIN_PATH)
sys.path.append("../")

# Do relative imports below this
from proloaf import metrics
import proloaf.loghandler as log
import proloaf.confighandler as ch
import proloaf.datahandler as dh
import proloaf.modelhandler as mh
import proloaf.tensorloader as tl

# TODO: tensorboard necessitates chardet, which is licensed under LGPL: https://pypi.org/project/chardet/
from proloaf.confighandler import read_config
from proloaf.cli import parse_with_loss
from proloaf.event_logging import create_event_logger

torch.set_printoptions(linewidth=120)  # Display option for output
torch.set_grad_enabled(True)
torch.manual_seed(1)

warnings.filterwarnings("ignore")

# create event logger
logger = create_event_logger("train")


def main(
    infile: str,
    config: dict,
    work_dir: str,
    loss: str,
    loss_kwargs: dict = {},
    # log_path: str = None,
    device: str = "cpu",
):

    logger.info("Current working directory is {:s}".format(work_dir))

    # Read load data
    config = deepcopy(config)
    # log_df = log.init_logging(model_name=station_name, work_dir=work_dir, config=config)
    try:
        df = pd.read_csv(infile, sep=";", index_col=0)

        train_df, val_df = dh.split(df, [config.get("train_split", 0.7)])

        scaler = dh.MultiScaler(config["feature_groups"])
        train_dataset = tl.TimeSeriesData(
            train_df,
            device=device,
            preparation_steps=[
                dh.set_to_hours,
                # TODO check if periodicity is correct
                partial(dh.fill_if_missing, periodicity=config.get("periodicity", 24)),
                dh.add_cyclical_features,
                dh.add_onehot_features,
                scaler.fit_transform,
                dh.check_continuity,
            ],
            **config,
        )
        val_dataset = tl.TimeSeriesData(
            val_df,
            device=device,
            preparation_steps=[
                dh.set_to_hours,
                # TODO check if periodicity is correct
                partial(dh.fill_if_missing, periodicity=config.get("periodicity", 24)),
                dh.add_cyclical_features,
                dh.add_onehot_features,
                scaler.transform,
                dh.check_continuity,
            ],
            **config,
        )

        if config.get("exploration_path") is None:
            tuning_config = None
        else:
            tuning_config = read_config(
                config_path=config["exploration_path"],
                main_path=work_dir,
            )

        modelhandler = mh.ModelHandler(
            work_dir=work_dir,
            config=config,
            tuning_config=tuning_config,
            scalers=scaler,
            loss=loss,
            loss_kwargs=loss_kwargs,
            device=device,
        )

        modelhandler.fit(
            train_dataset,
            val_dataset,
        )
        train_dataset = None
        try:
            ref_model_1 = modelhandler.load_model(
                os.path.join(
                    work_dir,
                    config.get("output_path", ""),
                    f"{config['model_name']}.pkl",
                )
            )
        except FileNotFoundError:
            ref_model_1 = None
            logger.info(
                "No old version of the trained model was found for the new one to compare to"
            )
        except Exception:
            ref_model_1 = None
            logger.warning(
                "An older version of this model was found but could not be loaded, this is likely due to diverignig ProLoaF versions."
            )
        try:
            if ref_model_1 is not None:
                modelhandler.select_model(
                    val_dataset,
                    [ref_model_1, modelhandler.model_wrap],
                    modelhandler._model_wrap.loss_metric,
                )
        except:
            logger.warning(
                "An older version of this model was found but could not be compared to the new one, this might be due to diverging model structures."
            )
            # TODO should the old model be overwritten anyways?
        modelhandler.save_current_model(
            os.path.join(
                work_dir, config.get("output_path", ""), f"{config['model_name']}.pkl"
            )
        )
        config.update(modelhandler.get_config())
        ch.write_config(
            config,
            model_name=ARGS.station,
            config_path=ARGS.config,
            main_path=work_dir,
        )

    except KeyboardInterrupt:
        logger.info("manual interrupt")


if __name__ == "__main__":
    ARGS, LOSS_OPTIONS = parse_with_loss()
    PAR = read_config(
        model_name=ARGS.station, config_path=ARGS.config, main_path=MAIN_PATH
    )
    if torch.cuda.is_available():
        DEVICE = "cuda"
        if PAR["cuda_id"] is not None:
            torch.cuda.set_device(PAR["cuda_id"])
    else:
        DEVICE = "cpu"

    main(
        infile=os.path.join(MAIN_PATH, PAR["data_path"]),
        config=PAR,
        device=DEVICE,
        work_dir=MAIN_PATH,
        loss=ARGS.loss,
        loss_kwargs=LOSS_OPTIONS,
    )
