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

#TODO: tensorboard necessitates chardet, which is licensed under LGPL: https://pypi.org/project/chardet/
#if 'exploration' is used, this would violate our license policy as LGPL is not compatible with apache
from utils.confighandler import read_config, get_existing_score
from utils.cli import parse_with_loss

import os
import sys
import warnings

import pandas as pd
import torch

MAIN_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
print(MAIN_PATH)
sys.path.append(MAIN_PATH)

# Do relative imports below this
import utils.loghandler as log
import utils.datahandler as dh
import utils.modelhandler as mh

torch.set_printoptions(linewidth=120)  # Display option for output
torch.set_grad_enabled(True)
torch.manual_seed(1)

warnings.filterwarnings("ignore")

def main(infile, outmodel, log_path=None):
    # Read load data
    df = pd.read_csv(infile, sep=";", index_col=0)

    dh.fill_if_missing(df, periodicity=24)

    #TODO: cleanup work on target list
    #only use target list if you want to predict the summed value. Actually the use is not recommended.
    #if "target_list" in PAR:
    #    if PAR["target_list"] is not None:
    #        df[target_id] = df[PAR["target_list"]].sum(axis=1)
    selected_features, scalers = dh.scale_all(df, **PAR)

    log_df = log.init_logging(
        model_name=ARGS.station,
        work_dir=MAIN_PATH,
        config=PAR,
    )

    old_val_loss = get_existing_score(config=PAR)

    try:
        study = None
        trial = None
        if PAR["exploration"]:
            hparams, sampler, study = mh.init_tuning(
                work_dir=MAIN_PATH,
                model_name=ARGS.station,
                config=PAR
            )

            study, trial, config = mh.hparam_tuning(
                work_dir=MAIN_PATH,
                config=PAR,
                hparam=hparams,
                study=study,
                selected_features=selected_features,
                scalers=scalers,
                log_df=log_df,
                args=ARGS,
                loss_options=LOSS_OPTIONS,
                device=DEVICE,
                min_val_loss=old_val_loss,
                interactive=False,
                **hparams,
            )
            PAR["exploration"]=False

        # If no exploration is done or hyperparameter tuning has ended in the best trial,
        # construct and train the model with current (updated) config in a "main-run"
        PAR["trial_id"] = "main_run"
        train_dl, validation_dl, test_dl = dh.transform(selected_features, device=DEVICE, **PAR)

        model = mh.make_model(
            scalers=scalers,
            enc_size=train_dl.number_features1(),
            dec_size=train_dl.number_features2(),
            num_pred=ARGS.num_pred,
            loss=ARGS.loss,
            **PAR,
        )

        net, log_df, loss, new_score, config = mh.train(
            work_dir=MAIN_PATH,
            train_data_loader=train_dl,
            validation_data_loader=validation_dl,
            test_data_loader=test_dl,
            net=model,
            device=DEVICE,
            args=ARGS,
            loss_options=LOSS_OPTIONS,
            log_df=log_df,
            logging_tb=True,
            config=PAR,
            **PAR,
        )

        old_score = get_existing_score(PAR)

        min_net = mh.save(
            work_dir=MAIN_PATH,
            model_name=ARGS.station,
            out_model=outmodel,
            old_score=old_score,
            achieved_score=new_score,
            achieved_loss=loss,
            achieved_net=net,
            config=config,
            config_path=ARGS.config,
            study=study,
            trial=trial,
            interactive=not ARGS.ci # TODO: make this more intuitive independent from CI args
        )

    except KeyboardInterrupt:
        print("manual interrupt")

    finally:
        if log_df is not None: log.end_logging(
            model_name=PAR["model_name"],
            work_dir=MAIN_PATH,
            log_path=log_path,
            df=log_df)


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
        DEVICE = 'cpu'

    if PAR["exploration"]:
        PAR["trial_id"] = 0  # set global trial ID for logging trials in subfolders
    main(
        infile=os.path.join(MAIN_PATH, PAR["data_path"]),
        outmodel=os.path.join(MAIN_PATH, PAR["output_path"], PAR["model_name"]),
        log_path=os.path.join(MAIN_PATH, PAR["log_path"]),
    )
