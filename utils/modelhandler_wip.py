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
import os
import shutil
import tempfile
import optuna
import torch
from functools import partial
from typing import Callable, Union
import utils

import utils.datahandler as dh

from datetime import datetime
from time import perf_counter  # ,localtime
from utils import models_wip as models
from utils import metrics
from utils.loghandler import (
    log_data,
    clean_up_tensorboard_dir,
    log_tensorboard,
    add_tb_element,
    end_tensorboard,
)
from utils.cli import query_true_false
from utils.confighandler import read_config, write_config


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
    def __init__(self, patience=7, verbose=False, delta=0, log_df=None, log_tb=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.temp_dir = ""
        self.log_df = log_df
        self.log_tb = False

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
            The validation loss, as calculated by one of the metrics in utils.eval_metrics
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
        work_dir: str,
        config: dict,
        model_name: str,
        model=None,
        tuning_config: dict = None,
        scalers=None,
        loss="nll_gaus",
        loss_kwargs: dict = {},
        device="cpu",
    ):
        self.model_name = model_name
        self._model = model
        self._config = config
        self._scalers = scalers
        self.tuning_config = tuning_config
        self.optimizer = None
        self._device = device
        self.set_loss(loss, **loss_kwargs)
        self.work_dir = work_dir
        self.hparams = None

        if config.get("exploration", False):
            if not self.tuning_config:
                raise AttributeError(
                    "Exploration was requested but no configuration for it was provided. Define the relative path to the hyper parameter tuning config as 'exploration_path' in the main config or provide a dict to the modelhandler"
                )

    def to(self, device):
        self._device = device
        self._model.to(device)
        return self

    @property
    def config(self):
        return self._config

    def set_loss(self, metric_id: str = "nll_gauss", **kwargs):
        """
        Set the meric which is to be used as loss for training

        Parameters
        ----------
        metric_id : string, default = 'mis'
            The name of the metric to use for scoring. See functions in utils.eval_metrics for
            possible values.
        **kwargs:
            arguments provided to the chosen metric.
        Returns
        -------
        self: ModelHandler
        """
        if metric_id == "mis":
            self._loss = partial(metrics.mis, **kwargs)
        elif metric_id == "nll_gauss":
            self._loss = partial(metrics.nll_gauss, **kwargs)
        elif metric_id == "quant":
            self._loss = partial(metrics.quantile_score, **kwargs)
        elif metric_id == "crps":
            self._loss = partial(metrics.crps_gaussion, **kwargs)
        elif metric_id == "mse":
            self._loss = partial(metrics.mse, **kwargs)
        elif metric_id == "rmse":
            self._loss = partial(metrics.rmse, **kwargs)
        elif metric_id == "mape":
            self._loss = partial(metrics.mape, **kwargs)
        else:
            # TODO: catch exception here if performance self._loss is undefined
            self._loss = None
        return self

    def tune_hyperparameters(
        self,
        previous_min_val_loss=float("inf"),
        interactive=False,
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
            self.tuning_objective(self.tuning_config["settings"]),
            n_trials=self.tuning_config["number_of_tests"],
            n_jobs=self.tuning_config.get("parallel_jobs", -1),
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
        if previous_min_val_loss is None:
            previous_min_val_loss = float("inf")
        if previous_min_val_loss < study.best_value:
            print("None of the tested setups performed better than the loss provided.")
        else:
            if interactive:
                if query_true_false(
                    "Found a better performing setup.\nAccept changes?"
                ):
                    self.config.update(trial.params)
                    print(
                        "Saved ... do not forget to save the in memory changes to disc."
                    )
            else:
                self.config.update(trial.params)
        self.model = trial.user_attrs["training_details"].model
        return self

    def train(
        self,  # Maybe use the datahandler as "data" which than provides all the data_loaders,for unifying the interface.
        train_data_loader,
        validation_data_loader,
        model=None,
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
        config = self.config.deepcopy()
        config.update(hparams)
        if model is None:
            model = self.model
            use_self_model = True
        training_run = TrainingRun(
            model,
            id=trial_id,
            optimizer_name=config["optimizer_name"],
            train_data_loader=train_data_loader,
            validation_data_loader=validation_data_loader,
            learning_rate=config["learning_rate"],
            loss_function="TODO",
            max_epochs=config["max_epochs"],
        )
        training_run.train()
        if use_self_model:
            self.model = training_run.model
            return self
        return model

    # TODO dataformat curretnly includes targets and features which differs from sklearn
    def fit(
        self,
        train_data_loader: utils.tensorloader.CustomTensorDataLoader,
        validation_data_loader: utils.tensorloader.CustomTensorDataLoader,
        tuning_config=None,
    ):
        if tuning_config is None:
            self.make_model(**self.config)
            self.train(train_data_loader, validation_data_loader, trial_id="main")
        else:
            self.tune_hyperparameters(
                previous_min_val_loss=self.config.get("best_loss", float("inf"))
            )
        return self

    def predict(self, data: utils.tensorloader.CustomTensorDataLoader):
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
        return [
            self.model(inputs_enc, inputs_dec) for inputs_enc, inputs_dec, _ in data
        ]

    def load(self, path: str):
        raise NotImplemented()

    def save(self, path: str):
        raise NotImplemented()

    # TODO adjust function inputs
    # def score(
    #     self,
    #     data,
    #     score_type="mis",
    #     option=0.05,
    #     avg_on_horizon=True,
    #     horizon=1,
    #     number_of_targets=1,
    # ):
    #     """
    #     Determine the score of the given model using the specified metrics in utils.eval_metrics

    #     Return a single float value (total score) or a 1-D array (score over horizon) based on
    #     the value of avg_on_horizon

    #     Parameters
    #     ----------
    #     data : utils.tensorloader.CustomTensorDataLoader
    #         Contains the input data and targets
    #     score_type : string, default = 'mis'
    #         The name of the metric to use for scoring. See functions in utils.eval_metrics for
    #         possible values.
    #     option : float or list, default = 0.05
    #         An optional parameter in case the 'mis' or 'quantile_score' functions are used. In the
    #         case of 'mis' it should be given the value for 'alpha'. In the case of QS, it should
    #         contain a list with the quantiles.
    #     avg_on_horizon : bool, default = true
    #         Determines whether the return value is a float (total score, when true) or a 1-D array
    #         (score over the horizon, when false).
    #     horizon : int, default = 1
    #         The horizon for the prediction
    #     number_of_targets : int, default = 1
    #         The number of targets for the prediction.

    #     Returns
    #     -------
    #     float or 1-D array (torch.Tensor)
    #         Either the total score (float) or the score over the horizon (array), depending on the
    #         value of avg_on_horizon
    #     """
    #     # check performance
    #     targets, raw_output = get_prediction(
    #         self._model, data, horizon, number_of_targets
    #     )  ##should be test data loader
    #     [y_pred_upper, y_pred_lower, expected_values] = get_pred_interval(
    #         raw_output, self.criterion, targets
    #     )
    #     # get upper and lower prediction interval, depending on loss function used for training
    #     # TODO give feedback on invalid score type
    #     if "mis" in str(score_type):
    #         output = [y_pred_upper, y_pred_lower]
    #         score = metrics.mis(targets, output, alpha=option, total=avg_on_horizon)
    #     elif "nll_gauss" in str(score_type):
    #         output = raw_output  ##this is only valid if self._model was trained with gaussin nll or crps
    #         score = metrics.nll_gauss(targets, output, total=avg_on_horizon)
    #     elif "quant" in str(score_type):
    #         output = expected_values
    #         score = metrics.quantile_score(
    #             targets, output, quantiles=option, total=avg_on_horizon
    #         )
    #     elif "crps" in str(score_type):
    #         output = raw_output  ##this is only valid if self._model was trained with gaussin nll or crps
    #         score = metrics.crps_gaussian(targets, output, total=avg_on_horizon)
    #     elif score_type == "mse":
    #         output = expected_values
    #         score = metrics.mse(targets, output.unsqueeze(0), total=avg_on_horizon)
    #     elif score_type == "rmse":
    #         output = expected_values
    #         score = metrics.rmse(targets, output.unsqueeze(0), total=avg_on_horizon)
    #     elif "mape" in str(score_type):
    #         output = expected_values
    #         score = metrics.mape(targets, output.unsqueeze(0), total=avg_on_horizon)
    #     else:
    #         # TODO: catch exception here if performance score is undefined
    #         score = None
    #     return score

    # TODO arguments are not correct yet
    def make_model(
        self,
        enc_size,
        dec_size,
        num_pred,
        loss,
        core_net,
        relu_leak,
        dropout_fc,
        dropout_core,
        rel_linear_hidden_size,
        rel_core_hidden_size,
        core_layers,
        **_,
    ):
        """
        Provide a model that has been prepared for training.

        Split the given data frame into training, validation and testing sets and specify
        encoder and decoder features. Use specified device for CUDA tensors if available.

        Parameters
        ----------
        scalers : dict
            A dict of sklearn.preprocessing scalers with their corresponding feature group
            names (e.g."main", "add") as keywords
        enc_size : int
            TODO
        dec_size : int
            TODO
        num_pred : int
            The number of prediction outputs, depending on loss function
        loss : string
            The string to specify the loss function
        core_net : string
            The name of the core_net, for example 'torch.nn.GRU'
        relu_leak : float scalar
            The value of the negative slope for the LeakyReLU
        dropout_fc : float scalar
            The dropout probability for the decoder
        dropout_core : float scalar
            The dropout probability for the core_net
        rel_linear_hidden_size : float scalar
            The relative linear hidden size, as a fraction of the total number of features in the training data
        rel_core_hidden_size : float scalar
            The relative core hidden size, as a fraction of the total number of features in the training data
        core_layers : int scalar
            The number of layers of the core_net
        target_id : string
            The name of feature to forecast, e.g. "demand"

        Returns
        -------
        utils.models.EncoderDecoder
            The prepared model with desired encoder/decoder features
        """
        model = models.EncoderDecoder(
            input_size1=enc_size,
            input_size2=dec_size,
            out_size=num_pred,
            dropout_fc=dropout_fc,
            dropout_core=dropout_core,
            rel_linear_hidden_size=rel_linear_hidden_size,
            rel_core_hidden_size=rel_core_hidden_size,
            core_net=core_net,
            relu_leak=relu_leak,
            core_layers=core_layers,
            criterion=loss,
        )
        for param in model.parameters():
            torch.nn.init.uniform_(param.data, -0.08, 0.08)
        return model

    @property
    def model(self):
        return self._model

    @property
    def scalers(self):
        return self._scalers

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
            seed=seed
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
        train_data_loader: utils.datahandler.CustomTensorDataLoader,
        validation_data_loader: utils.datahandler.CustomTensorDataLoader,
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

            config = self.config.deepcopy()
            config.update(hparams)

            model = self.make_model(**config)
            training_run = TrainingRun(
                model,
                optimizer_name=config["optimizer_name"],
                train_data_loader=train_data_loader,
                validation_data_loader=validation_data_loader,
                learning_rate=config["learning_rate"],
                loss_function="TODO",
                max_epochs=config["max_epochs"],
            )
            training_run.train()
            trial.set_user_attr("training_details", training_run)
            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            return training_run.validation_loss

        return search_params


class TrainingRun:
    _next_id = 1

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer_name: str,
        train_data_loader: utils.datahandler.CustomTensorDataLoader,
        validation_data_loader: utils.datahandler.CustomTensorDataLoader,
        learning_rate: float,
        loss_function: Callable[[torch.Tensor, torch.Tensor], float],
        max_epochs: int,
        id: Union[str, int] = None,
        device: str = "cpu",
    ):
        if id is None:
            self.id = TrainingRun._next_id
            TrainingRun._next_id += 1
        else:
            self.id = id
        self.model = model
        self.train_dl = train_data_loader
        self.validation_dl = validation_data_loader
        self.validation_loss = float("inf")
        self.training_loss = float("inf")
        self.set_optimizer(optimizer_name, learning_rate)
        self.loss_function = loss_function
        self.step_counter = 0
        self.max_epochs = max_epochs
        self.early_stopping = EarlyStopping()
        self.device = device

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
        self.validation_loss = float("inf")
        self.training_loss = float("inf")

    def step(self):
        epoch_loss = 0.0
        t1_start = perf_counter()
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
            epoch_loss += loss.item() / len(self.train_dl)
            final_epoch_loss = epoch_loss
        t1_stop = perf_counter()
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
        self.model.train()
        for epoch in range(self.max_epochs):
            self.step()
            self.validate()
            # TODO Here should be some logging
            self.early_stopping(self.validation_loss, self.model)
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
        return self

    def to(self, device: str):
        self.device = device
        self.model.to(device)
        self.train_dl.to(device)
        self.validation_dl.to(device)
        return self


###################################################### REFACTORED ABOVE #############################################################################


def get_pred_interval(predictions, criterion, targets):
    """
    Get the upper and lower limits and expected values of predictions

    Parameters
    ----------
    predictions : torch.Tensor
        The predicted values
    criterion : callable
        One of the metrics from utils.eval_metrics
    targets : torch.Tensor
        The the actual (not predicted) values

    Returns
    -------
    torch.Tensor
        The upper limit of the predictions
    torch.Tensor
        The lower limit of the predictions
    torch.Tensor
        The expected values of the predictions
    """
    # Better solution for future: save criterion as class object of 'loss' with 'name' attribute
    # detect criterion
    if ("nll_gaus" in str(criterion)) or ("crps" in str(criterion)):
        # loss_type = 'nll_gaus'
        expected_values = predictions[:, :, 0:1]  # expected_values:mu
        sigma = torch.sqrt(predictions[:, :, -1:].exp())
        # TODO: make 95% prediction interval changeable
        y_pred_upper = expected_values + 1.96 * sigma
        y_pred_lower = expected_values - 1.96 * sigma
    elif "quantile" in str(criterion):
        # loss_type = 'pinball'
        y_pred_lower = predictions[:, :, 0:1]
        y_pred_upper = predictions[:, :, 1:2]
        expected_values = predictions[:, :, -1:]
    elif "rmse" in str(criterion):
        expected_values = predictions
        rmse = metrics.rmse(targets, expected_values.unsqueeze(0))
        # In order to produce an interval covering roughly 95% of the error magnitudes,
        # the prediction interval is usually calculated using the model output ± 2 × RMSE.
        y_pred_lower = expected_values - 2 * rmse
        y_pred_upper = expected_values + 2 * rmse
    elif ("mse" in str(criterion)) or ("mape" in str(criterion)):
        # loss_type = 'mis'
        expected_values = predictions
        y_pred_lower = 0
        y_pred_upper = 1

        # TODO: add all criterion possibilities
    else:
        print("invalid criterion")

    return y_pred_upper, y_pred_lower, expected_values


# ToDo: refactor best score, refactor relative_metric
def calculate_relative_metric(curr_score, best_score):
    """
    Calculate the difference between the best score and the current score, as a percentage
    relative to the best score.

    A negative result indicates that the current score is lower (=better) than the best score

    Parameters
    ----------
    curr_score : float
        The current score
    best_score : float
        The best score

    Returns
    -------
    float
        The relative score in percent
    """
    return (100 / best_score) * (curr_score - best_score)


class loss:
    """
    Stores loss functions and how many parameters they have.

    Parameters
    ----------
    func : callable
        A callable loss function
    num_params : int
        Number of parameters
    """

    def __init__(self, func, num_params):
        self._func = func
        self.num_params = num_params

    def __call__(self, *args, **kwargs):
        return self._func(*args, **kwargs)


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
