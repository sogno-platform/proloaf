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
Provides functions and classes for using the models to get predictions
and testing model performance.
"""

import numpy as np
import pandas as pd
import os
import sys
import tempfile
import optuna
import torch
from typing import Any, Callable, Union, List, Dict
from copy import deepcopy
import utils

from time import perf_counter
from utils import models
from utils import metrics
from utils.loghandler import (
    log_data,
    clean_up_tensorboard_dir,
    log_tensorboard,
    add_tb_element,
    end_tensorboard,
)
from utils.cli import query_true_false
from utils.confighandler import write_config


class EarlyStopping:
    """
    Early stop the training if validation loss doesn't improve after a given patience

    Parameters
    ----------
    patience : int, default = 7
        How long to wait after last time validation loss improved.
    verbose : bool, default = False
        If True, prints a message for each validation loss improvement.
    delta : float, default = 0.0
        Minimum change in the monitored quantity to qualify as an improvement.

    Notes
    -----
    Reference: https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
    """

    # implement early stopping
    # Reference: https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.temp_dir = ""

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score - self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """
        Save model when validation loss decreases

        Parameters
        ----------
        val_loss : float
            The validation loss, as calculated by one of the metrics in utils.metrics
        model : utils.models.EncoderDecoder
            The model being trained
        """
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        temp_dir = tempfile.mktemp()
        torch.save(model.state_dict(), temp_dir + "checkpoint.pt")
        self.val_loss_min = val_loss
        self.temp_dir = temp_dir


class ModelWrapper:
    def __init__(
        self,
        # model: torch.nn.Module = None,
        name: str = "model",
        target_id: Union[str, int] = "target",
        # output_labels: List[str] = None,
        scalers=None,
        training_metric: str = None,
        metric_options: Dict[str, Any] = {},
    ):
        self.model = None  # model
        self.name = name
        self.target_id = target_id

        self.scalers = scalers  # TODO make custom wrapper for scaler that know what data goes where?
        # self.previous_loss = None
        self.training_metric = training_metric
        self._loss = training_metric
        self._loss_options = deepcopy(metric_options)
        if self._loss is not None:
            self.output_labels = deepcopy(self.loss_metric.input_labels)
        else:
            self.output_labels = None

    @property
    def loss_metric(self):
        return metrics.get_metric(self._loss, **self._loss_options)

    @loss_metric.setter
    def loss_metric(self, var):
        raise AttributeError("Can't set loss manually")

    def wrap(self, model: torch.nn.Module):  # , previous_loss: float = None):
        self.model = model
        # self.previous_loss = previous_loss

    def copy(self):
        return ModelWrapper(
            name=self.name,
            target_id=self.target_id,
            scalers=self.scalers,
            training_metric=self._loss,
            metric_options=self._loss_options,
        )

    def init_model(
        self,
        enc_size: int,
        dec_size: int,
        out_size: int,
        rel_linear_hidden_size: float = 1.0,
        rel_core_hidden_size: float = 1.0,
        core_layers: int = 1,
        dropout_fc: float = 0.0,
        dropout_core: float = 0.0,
        core_net: str = "torch.nn.GRU",
        relu_leak: float = 0.1,
    ):
        self.model = models.EncoderDecoder(
            enc_size=enc_size,
            dec_size=dec_size,
            out_size=out_size,
            dropout_fc=dropout_fc,
            dropout_core=dropout_core,
            rel_linear_hidden_size=rel_linear_hidden_size,
            rel_core_hidden_size=rel_core_hidden_size,
            core_net=core_net,
            relu_leak=relu_leak,
            core_layers=core_layers,
        )
        for param in self.model.parameters():
            torch.nn.init.uniform_(param.data, -0.08, 0.08)
        self.previous_loss = None
        return self

    def init_model_from_config(self, config: dict):
        if not self.output_labels:
            raise AttributeError(
                "The output labels for the model have to be specified to create the forecasting model, they determine the number of predicted values."
            )
        self.init_model(
            enc_size=len(config["encoder_features"]),
            dec_size=len(config["decoder_features"]),
            out_size=len(self.output_labels),
            rel_linear_hidden_size=config.get("rel_linear_hidden_size", 1.0),
            rel_core_hidden_size=config.get("rel_core_hidden_size", 1.0),
            core_layers=config.get("core_layers", 1),
            dropout_fc=config.get("dropout_fc", 0),
            dropout_core=config.get("dropout_core", 0),
            core_net=config.get("core_net", "torch.nn.GRU"),
            relu_leak=config.get("relu_leak", 0.01),
        )

    def to(self, device):
        if self.model:
            self.model.to(device)
        return self

    def predict(self, inputs_enc: torch.Tensor, inputs_dec: torch.Tensor):
        """
        Get the predictions for the given model and data

        Parameters
        ----------
        net : utils.models.EncoderDecoder
            The model with which to calculate the predictions
        data_loader : utils.tensorloader.CustomTensorDataLoader
            Contains the input data and targets
        horizon : int
            The horizon for the prediction
        number_of_targets : int
            The number of targets

        Returns
        -------
        torch.Tensor
            The targets (actual values)
        torch.Tensor
            The predictions from the given model
        """
        if self.model is None:
            raise RuntimeError("No model has been created to perform a prediction with")
        # TODO this array of numbers is not very readable, maybe a dict would be helpful
        val, _ = self.model(inputs_enc, inputs_dec)
        return val

    def add_scalers(self, scalers):
        self.scalers = scalers
        return self


class ModelHandler:
    """
    Utility wrapper for the model, implementing sklearns Predictor interface in addition to providing methods for initializing and handling the forecasting model.

    Parameters
    ----------
    TODO
    patience : int, default = 7
        How long to wait after last time validation loss improved.
    verbose : bool, default = False
        If True, prints a message for each validation loss improvement.
    delta : float, default = 0.0
        Minimum change in the monitored quantity to qualify as an improvement.
    model : utils.models.EncoderDecoder
        The model which is to be used for forecasting, if one was perpared separately. It is however recommended to initialize and train the model using the modelhandler.
    Notes
    -----
    Reference: https://scikit-learn.org/stable/developers/develop.html
    """

    def __init__(
        self,
        work_dir: str,  # TODO this is not great if the modelhandler is loaded on a different machine
        config: dict,
        # reference_model: Union[torch.nn.Module, ModelWrapper] = None,
        tuning_config: dict = None,
        scalers=None,  # TODO think about we can reasonably provide typing here
        loss: str = "nllgauss",
        loss_kwargs: dict = {},
        device: str = "cpu",
        log_df=None,
        log_tb=False,
    ):
        print(f"{os.path.dirname(os.path.abspath(sys.argv[0])) = }")
        self.work_dir = work_dir
        self.config = deepcopy(config)
        # if isinstance(reference_model, ModelWrapper):
        #     self.reference_model = reference_model
        #     # rename model if name is specified
        # else:
        #     self.reference_model = ModelWrapper(
        #         name=self.config.get("model_name", "temp_model"),
        #         target_id=self.config["target_id"],
        #         training_metric=loss,
        #         metric_options=loss_kwargs,
        #         scalers=scalers,
        #     )
        #     if isinstance(reference_model, torch.nn.Module):
        #         self.reference_model.model = reference_model

        self.tuning_config = deepcopy(tuning_config)
        # self.set_loss(loss, **loss_kwargs)
        self._model_wrap: ModelWrapper = ModelWrapper(
            name=self.config.get("model_name", "temp_model"),
            target_id=self.config["target_id"],
            scalers=scalers,
            training_metric=loss,
            metric_options=loss_kwargs,
        )
        self.to(device)

        if config.get("exploration", False):
            if not self.tuning_config:
                raise AttributeError(
                    "Exploration was requested but no configuration for it was provided. Define the relative path to the hyper parameter tuning config as 'exploration_path' in the main config or provide a dict to the modelhandler"
                )

    @property
    def model_wrap(self):
        return self._model_wrap

    @model_wrap.setter
    def model_wrap(self, model_wrap: torch.nn.Module):
        self._model_wrap = model_wrap
        if self._model_wrap:
            self._model_wrap.to(self._device)

    def to(self, device):
        self._device = device
        if self.model_wrap:
            self.model_wrap.to(device)
        return self

    # def set_loss(self, metric: str = "nllgauss", **kwargs):
    #     """
    #     Set the meric which is to be used as loss for training

    #     Parameters
    #     ----------
    #     metric_id : string, default = 'mis'
    #         The name of the metric to use for scoring. See functions in utils.eval_metrics for
    #         possible values.
    #     **kwargs:
    #         arguments provided to the chosen metric.
    #     Returns
    #     -------
    #     self: ModelHandler
    #     """
    #     self._loss = metrics.get_metric(metric, **kwargs)
    #     return self

    def tune_hyperparameters(
        self,
        train_data_loader: utils.tensorloader.CustomTensorDataLoader,
        validation_data_loader: utils.tensorloader.CustomTensorDataLoader,
    ):
        """
        TODO: description

        ToDo: long_description

        Notes
        -----
        Hyperparameter exploration

        - Any training parameter is considered a hyperparameter as long as it is specified in
            either config.json or tuning.json. The latter is the standard file where the (so far)
            best found configuration is saved and should usually not be manually adapted unless
            new tests are for some reason not comparable to older ones (e.g. after changing the loss function).
        - Possible hyperparameters are: target_column, encoder_features, decoder_features,
            max_epochs, learning_rate, batch_size, shuffle, history_horizon, forecast_horizon,
            train_split, validation_split, core_net, relu_leak, dropout_fc, dropout_core,
            rel_linear_hidden_size, rel_core_hidden_size, optimizer_name, cuda_id

        Parameters
        ----------
        TODO: complete
        config : dict, default = {}
            dict of model configurations
        Returns
        -------
        TODO: complete
        """
        print(
            "Max. number of iteration trials for hyperparameter tuning: ",
            self.tuning_config["number_of_tests"],
        )
        study = self.make_study()
        study.optimize(
            self.tuning_objective(
                self.tuning_config["settings"],
                train_data_loader=train_data_loader,
                validation_data_loader=validation_data_loader,
            ),
            n_trials=self.tuning_config["number_of_tests"],
            n_jobs=self.tuning_config.get(
                "parallel_jobs", 2
            ),  # TODO set to -1 (=processors)
            timeout=self.tuning_config.get("timeout", None),
        )

        print("Number of finished trials: ", len(study.trials))
        # trials_df = study.trials_dataframe()

        if not os.path.exists(os.path.join(self.work_dir, self.config["log_path"])):
            os.mkdir(os.path.join(self.work_dir, self.config["log_path"]))

        print("Best trial:")
        trial = study.best_trial
        print("  Value: ", trial.value)
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

        # TODO this is wonky self.model and self.config should ALWAYS be related.
        # How can we compare to previous performance. Probably by just running both models on the same dataset.
        # if previous_min_val_loss is None:
        #     previous_min_val_loss = np.inf
        # if previous_min_val_loss < study.best_value:
        #     print("None of the tested setups performed better than the loss provided.")
        # else:
        #     if interactive:
        #         if query_true_false(
        #             "Found a better performing setup.\nAccept changes?"
        #         ):
        #             self.config.update(trial.params)
        #             print(
        #                 "Saved ... do not forget to save the in memory changes to disc."
        #             )
        #     else:
        self.config.update(trial.params)
        self.model_wrap = trial.user_attrs["wrapped_model"]
        return self

    def benchmark(
        self,
        data: utils.tensorloader.CustomTensorDataLoader,
        models: List[ModelWrapper],
        test_metrics: List[metrics.Metric],
    ):
        np.empty(shape=(len(data), len(test_metrics), len(models)), dtype=np.float)
        for inputs_enc, inputs_dec, targets in data:
            intervals = [
                model.loss_metric.get_prediction_interval(
                    model.predict(inputs_enc, inputs_dec)
                )
                for model in models
            ]
            scores = [
                [
                    metric.from_interval(targets, upper, lower, expected)
                    for upper, lower, expected in intervals
                ]
                for metric in test_metrics
            ]
        return pd.DataFrame(
            scores,
            index=[met.id for met in test_metrics],
            columns=[mod.name for mod in models],
        )

    # def compare_to_old_model(self, data: utils.tensorloader.CustomTensorDataLoader):
    #     cumulative_loss = 0
    #     with torch.no_grad():
    #         for input_enc, input_dec, target in data:
    #             predictions_old = self.reference_model.predict(input_enc, input_dec)
    #             predictions_current = self.model_wrap.predict(input_enc, input_dec)
    #             loss_old = self._loss(predictions=predictions_old, target=target)
    #             loss_current = self._loss(
    #                 predictions=predictions_current, target=target
    #             )
    #             cumulative_loss = (
    #                 cumulative_loss + loss_current.item() - loss_old.item()
    #             )
    #             break
    #     return cumulative_loss

    def run_training(
        self,  # Maybe use the datahandler as "data" which than provides all the data_loaders,for unifying the interface.
        train_data_loader,
        validation_data_loader,
        trial_id=None,
        hparams={},
    ):
        """
        Train the given model.

        Train the provided model using the given parameters for up to the specified number of epochs, with early stopping.
        Log the training data (optionally using TensorBoard's SummaryWriter)
        Finally, determine the score of the resulting best net.

        Parameters
        ----------
        train_data_loader : utils.tensorloader.CustomTensorDataLoader
            The training data loader
        validation_data_loader : utils.tensorloader.CustomTensorDataLoader
            The validation data loader
        test_data_loader : utils.tensorloader.CustomTensorDataLoader
            The test data loader
        net : utils.models.EncoderDecoder
            The model to be trained
        learning_rate : float, optional
            The specified optimizer's learning rate
        batch_size :  int scalar, optional
            The size of a batch for the tensor data loader
        forecast_horizon : int scalar, optional
            The length of the forecast horizon in hours
        dropout_fc : float scalar, optional
            The dropout probability for the decoder
        dropout_core : float scalar, optional
            The dropout probability for the core_net
        log_df : pandas.DataFrame, optional
            A DataFrame in which the results and parameters of the training are logged
        optimizer_name : string, optional, default "adam"
            The name of the torch.optim optimizer to be used. Currently only the following
            strings are accepted as arguments: 'adagrad', 'adam', 'adamax', 'adamw', 'rmsprop', or 'sgd'
        max_epochs : int scalar, optional
            The maximum number of training epochs
        logging_tb : bool, default = True
            Specifies whether TensorBoard's SummaryWriter class should be used for logging during the training
        loss_options : dict, default={}
            Contains extra options if the loss functions mis or quantile score are used.
        exploration : bool
            todo
        trial_id : string , default "main_run"
            separate the trials per study and store all in one directory for better handling in tensorboard
        hparams : dict, default = {}, equals standard list of hyperparameters (batch_size, learning_rate)
            dict of customized hyperparameters
        config : dict, default = {}
            dict of model configurations
        Returns
        -------
        utils.models.EncoderDecoder
            The trained model
        pandas.DataFrame
            A DataFrame in which the results and parameters of the training have been logged
        float
            The minimum validation loss of the trained model
        float or torch.Tensor
            The score returned by the performance test. The data type depends on which metric was used.
            The current implementation calculates the Mean Interval Score and returns either a float, or 1d-Array with the MIS along the horizon.
            A lower score is generally better
        TODO: update
        """
        # to track the validation loss as the model trains

        config = deepcopy(self.config)
        config.update(hparams)
        temp_model_wrap: ModelWrapper = self.model_wrap.copy()
        temp_model_wrap.init_model_from_config(config)

        # tb = log_tensorboard(
        #     work_dir=self.work_dir,
        #     exploration=self.config["exploration"],
        #     trial_id=trial_id,
        # )
        training_run = TrainingRun(
            temp_model_wrap.model,
            id=trial_id,
            optimizer_name=config["optimizer_name"],
            train_data_loader=train_data_loader,
            validation_data_loader=validation_data_loader,
            earlystopping=EarlyStopping(
                patience=config.get("early_stopping_patience", 7),
                delta=config.get("early_stopping_margin",0.0)
            ),
            learning_rate=config["learning_rate"],
            loss_function=temp_model_wrap.loss_metric,
            max_epochs=config["max_epochs"],
            # log_tb=tb,
        )
        training_run.train()
        return temp_model_wrap, training_run

    # TODO dataformat currently includes targets and features which differs from sklearn
    def fit(
        self,
        train_data_loader: utils.tensorloader.CustomTensorDataLoader,
        validation_data_loader: utils.tensorloader.CustomTensorDataLoader,
        exploration: bool = None,
    ):
        if exploration is None:
            exploration = self.config.get("exploration", False)
        if exploration:
            if not self.tuning_config:
                raise AttributeError(
                    "Hyper parameters are to be explored but no config for tuning was provided."
                )
            self.tune_hyperparameters(train_data_loader, validation_data_loader)
        else:
            self.model_wrap, _ = self.run_training(
                train_data_loader,
                validation_data_loader,
                trial_id="main",
            )
        return self

    def predict(self, inputs_enc: torch.Tensor, inputs_dec: torch.Tensor):
        """
        Get the predictions for the given model and data

        Parameters
        ----------
        net : utils.models.EncoderDecoder
            The model with which to calculate the predictions
        data_loader : utils.tensorloader.CustomTensorDataLoader
            Contains the input data and targets
        horizon : int
            The horizon for the prediction
        number_of_targets : int
            The number of targets

        Returns
        -------
        torch.Tensor
            The targets (actual values)
        torch.Tensor
            The predictions from the given model
        """
        if self.model_wrap is None:
            raise RuntimeError("No model has been created to perform a prediction with")
        # TODO this array of numbers is not very readable, maybe a dict would be helpful
        return self.model_wrap.predict(inputs_enc, inputs_dec)

    def load_model(self, path: str = None):
        if path is None:
            path = os.path.join(
                self.work_dir,
                self.config.get("output_path", ""),
                f"{self.config['model_name']}.pkl",
            )
        inst = torch.load(path)
        if not isinstance(inst, ModelWrapper):
            raise RuntimeError(
                f"you tryied to load from '{path}' but the object was not a ModelWrapper"
            )
        self.model_wrap = inst
        # self.model_wrap = inst  # TODO decide on one
        return self

    def save_model(self, path: str = None):
        if path is None:
            path = os.path.join(
                self.work_dir,
                self.config.get("output_path", ""),
                f"{self.config['model_name']}.pkl",
            )
        torch.save(self.model_wrap, path)
        return self

    def make_study(
        seed=10,
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(),
    ):
        """
        TODO: Description

        Parameters
        ----------
        Returns
        -------
        TODO
        Raises
        ------
        TODO
        """
        # Set up the median stopping rule as the pruning condition.
        sampler = optuna.samplers.TPESampler(
            # seed=seed
        )  # Make the sampler behave in a deterministic way.
        study = optuna.create_study(
            sampler=sampler,
            direction=direction,
            pruner=pruner,
        )
        return study

    def tuning_objective(
        self,
        tuning_settings: dict,
        train_data_loader,
        validation_data_loader,
    ):
        """
        Implement an objective function for optimization with Optuna.

        Provide a callable for Optuna to use for optimization. The callable creates and trains a
        model with the specified features, scalers and hyperparameters. Each hyperparameter triggers a trial.

        Parameters
        ----------
        work_dir: TODO
        selected_features : pandas.DataFrame
            The data frame containing the model features, to be split into sets for training
        scalers : dict
            A dict of sklearn.preprocessing scalers with their corresponding feature group
            names (e.g."main", "add") as keywords
        hyper_param: dict
            A dictionary containing hyperparameters for the Optuna optimizer
        log_df : pandas.DataFrame
            A DataFrame in which the results and parameters of the training are logged
        config : dict, default = {}
            dict of model configurations
        args : TODO
        Returns
        -------
        Callable
            A callable that implements the objective function. Takes an optuna.trial._trial.Trial as an argument,
            and is used as the first argument of a call to optuna.study.Study.optimize()
            TODO: update
        Raises
        ------
        optuna.exceptions.TrialPruned
            If the trial was pruned
        """

        def search_params(trial: optuna.trial.Trial):
            # for more hyperparam, add settings and kwargs in a way compatible with
            # trial object(and suggest methods) of optuna
            hparams = {}
            for key, hparam in tuning_settings.items():
                print("Creating parameter: ", key)
                func_generator = getattr(trial, hparam["function"])
                hparams[key] = func_generator(**(hparam["kwargs"]))

            model_wrap, training_run = self.run_training(
                train_data_loader,
                validation_data_loader,
                hparams=hparams,
                trial_id=trial.number,
            )
            trial.set_user_attr("wrapped_model", model_wrap)
            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            return training_run.validation_loss

        return search_params


class TrainingRun:
    _next_id = 0

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer_name: str,
        learning_rate: float,
        loss_function: Callable[[torch.Tensor, torch.Tensor], float],
        max_epochs: int,
        train_data_loader: utils.tensorloader.CustomTensorDataLoader = None,
        validation_data_loader: utils.tensorloader.CustomTensorDataLoader = None,
        id: Union[str, int] = None,
        device: str = "cpu",
        log_df: pd.DataFrame = None,
        log_tb: torch.utils.tensorboard.SummaryWriter = None,
    ):
        if id is None:
            self.id = TrainingRun._next_id
            TrainingRun._next_id += 1
        else:
            self.id = id
        self.model = model
        self.train_dl = train_data_loader
        self.validation_dl = validation_data_loader
        self.validation_loss = np.inf
        self.training_loss = np.inf
        self.set_optimizer(optimizer_name, learning_rate)
        self.loss_function = loss_function
        self.step_counter = 0
        self.max_epochs = max_epochs
        self.early_stopping = EarlyStopping()
        self.log_df = log_df
        self.log_tb = log_tb
        self.training_start_time = None
        self.training_end_time = None
        self.to(device)

    def set_optimizer(self, optimizer_name: str, learning_rate: float):
        """
        Specify which optimizer to use during training.

        Initialize a torch.optim optimizer for the given model based on the specified name and learning rate.

        Parameters
        ----------
        name : string or None, default = 'adam'
            The name of the torch.optim optimizer to be used. The following
            strings are accepted as arguments: 'adagrad', 'adam', 'adamax', 'adamw', 'rmsprop', or 'sgd'

        learning_rate : float or None
            The learning rate to be used by the optimizer. If set to None, the default value as defined in
            torch.optim is used

        Returns
        -------
        torch.optim optimizer class
            A torch.optim optimizer that implements one of the following algorithms:
            Adagrad, Adam, Adamax, AdamW, RMSprop, or SGD (stochastic gradient descent)
            SGD is set to use a momentum of 0.5.

        """
        if self.model is None:
            raise AttributeError(
                "The model has to be initialized before the optimizer is set."
            )
        if optimizer_name == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        if optimizer_name == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=learning_rate, momentum=0.5
            )
        if optimizer_name == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=learning_rate
            )
        if optimizer_name == "adagrad":
            self.optimizer = torch.optim.Adagrad(
                self.model.parameters(), lr=learning_rate
            )
        if optimizer_name == "adamax":
            self.optimizer = torch.optim.Adamax(
                self.model.parameters(), lr=learning_rate
            )
        if optimizer_name == "rmsprop":
            self.optimizer = torch.optim.RMSprop(
                self.model.parameters(), lr=learning_rate
            )
        if self.optimizer is None:
            raise AttributeError(f"Could find optimizer with name {optimizer_name}.")
        return self

    def reset(self):
        self.step_counter = 0
        self.validation_loss = np.inf
        self.training_loss = np.inf

    def step(self):
        self.training_loss = 0.0

        self.model.train()
        # train step
        for (inputs1, inputs2, targets) in self.train_dl:
            prediction, _ = self.model(inputs1, inputs2)
            self.optimizer.zero_grad()
            loss = self.loss_function(targets, prediction)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            self.optimizer.step()
            self.step_counter += 1
            self.training_loss += loss.item() / len(self.train_dl)

        return self

    def validate(self):
        with torch.no_grad():
            self.model.eval()
            self.validation_loss = 0.0
            for (inputs1, inputs2, targets) in self.validation_dl:
                output, _ = self.model(inputs1, inputs2)
                self.validation_loss += self.loss_function(
                    targets, output
                ).item() / len(self.validation_dl)
        return self

    def train(self):
        if not self.train_dl:
            raise AttributeError("No training data provided")
        if self.log_tb:
            # is this actually for every run or model specific
            inputs_enc, inputs_dec, targets = next(iter(self.train_dl))
            self.log_tb.add_graph(self.model, [inputs_enc, inputs_dec])
        self.training_start_time = perf_counter()
        print("Begin training...")
        self.model.train()
        for epoch in range(self.max_epochs):
            t1_start = perf_counter()
            self.step()
            t1_stop = perf_counter()
            if self.validation_dl:
                self.validate()
                self.early_stopping(self.validation_loss, self.model)
            else:
                print(
                    "No validation data was provided, thus no validation was performed"
                )
            print(
                "Epoch {}/{}\t train_loss {:.2e}\t val_loss {:.2e}\t elapsed_time {:.2e}".format(
                    epoch + 1,
                    self.max_epochs,
                    self.training_loss,
                    self.validation_loss,
                    t1_stop - t1_start,
                )
            )
            if self.log_tb:
                add_tb_element(
                    net=self.model,
                    tb=self.log_tb,
                    epoch_loss=self.training_loss,
                    validation_loss=self.validation_loss,
                    t0_start=self.training_start_time,
                    t1_stop=t1_stop,
                    t1_start=t1_start,
                    next_epoch=epoch + 1,
                    step_counter=self.step_counter,
                )

            if self.early_stopping.early_stop:
                print(
                    f"No improvement has been achieved in the last {self.early_stopping.patience} epochs. Aborting training and loading best model."
                )
                # load the last checkpoint with the best model
                self.model.load_state_dict(
                    torch.load(self.early_stopping.temp_dir + "checkpoint.pt")
                )
                self.validation_loss = self.early_stopping.val_loss_min
                break
        self.model.eval()
        self.training_end_time = t1_stop
        return self

    def to(self, device: str):
        self.device = device
        if self.model:
            self.model.to(device)
        if self.train_dl:
            self.train_dl.to(device)
        if self.validation_dl:
            self.validation_dl.to(device)
        return self


###################################################### REFACTORED ABOVE #############################################################################


def update(
    model_name,
    achieved_score,
    config,
    loss,
    exploration=True,
    study=None,
    interactive=False,
):
    config["best_score"] = achieved_score
    config["best_loss"] = loss

    if exploration:
        if interactive:
            if query_true_false("Overwrite config with new parameters?"):
                print("study best value: ", study.best_value)
                print("current loss: ", loss)
                config.update(study.best_trial.params)
                config["exploration"] = not query_true_false(
                    "Parameters tuned and updated. Do you wish to turn off hyperparameter tuning for future training?"
                )

    print(
        "Model improvement achieved. Save {}-file in {}.".format(
            model_name, config["output_path"]
        )
    )


def save(
    work_dir,
    model_name,
    out_model,
    old_score,
    achieved_score,
    achieved_loss,
    achieved_net,
    config={},
    config_path="",
    study=None,
    trial=None,
    interactive=False,
):
    """
    description

    long_description

    Notes
    -----


    Parameters
    ----------
    TODO: complete this list
    work_dir : string
        TODO
    config : dict, default = {}
        dict of model configurations
    config_path : string, default = " "
        TODO
    Returns
    -------
    min_net
    TODO: complete
    """
    min_net = None
    if old_score > achieved_score:
        if config["exploration"]:
            update(
                model_name=model_name,
                achieved_score=achieved_score,
                config=config,
                exploration=config["exploration"],
                study=study,
                interactive=interactive,
                loss=achieved_loss,
            )
        min_net = achieved_net
    else:
        if query_true_false(
            "Existing model for this target did not improve in current run. "
            "Do you still want to overwrite the existing model?"
        ):
            # user has selected to overwrite existing model file
            min_net = achieved_net

    if min_net is not None:
        print("saving model")
        if not os.path.exists(
            os.path.join(work_dir, config["output_path"])
        ):  # make output folder if it doesn't exist
            os.makedirs(os.path.join(work_dir, config["output_path"]))
        torch.save(min_net, out_model)

        # drop unnecessary helper vars before using PAR to save config
        config.pop("hyper_params", None)
        config.pop("trial_id", None)
        config.pop("n_trials", None)

        write_config(
            config,
            model_name=model_name,
            config_path=config_path,
            main_path=work_dir,
        )

    return min_net
