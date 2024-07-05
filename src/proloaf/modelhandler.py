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
from __future__ import annotations
import numpy as np
import pandas as pd
import sklearn
import os
import sys
import tempfile
import optuna
import torch
from typing import Any, Callable, Iterable, Union, List, Dict, Literal
from copy import deepcopy

import proloaf

from time import perf_counter
from proloaf import models
from proloaf import metrics
from proloaf.loghandler import (
    log_tensorboard,
    add_tb_element,
    end_tensorboard,
)
from proloaf.event_logging import create_event_logger
from proloaf.event_logging import timer

logger = create_event_logger(__name__)


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

    def __init__(self, patience: int = 7, verbose: bool = False, delta: float = 0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.temp_dir = ""

    def __call__(self, val_loss: float, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score - self.delta:
            self.counter += 1
            logger.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss: float, model):
        """
        Save model when validation loss decreases

        Parameters
        ----------
        val_loss : float
            The validation loss, as calculated by one of the metrics in proloaf.metrics
        model : proloaf.models.EncoderDecoder
            The model being trained
        """
        if self.verbose:
            logger.info(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        temp_dir = "./"  # tempfile.mktemp()
        torch.save(model.state_dict(), temp_dir + "checkpoint.pt")
        self.val_loss_min = val_loss
        self.temp_dir = temp_dir


class ModelWrapper:
    """Bundles a forcasting Model with the information needed to train it. It exposes methods similar to a SciKit-Learn regressor,
    due to different input types they also differ in name to avoid missuse.

    Parameters
    ----------
    name: str, default = "model"
        Any name for the model
    target_id: Union[str, int], default = "target",
        Name of the data column that should be predicted
    encoder_features: List[str], default = None
        Features the encoder of the model needs, for documentation only (will change in the future)
    decoder_features: List[str] = None,
        Features the encoder of the model needs, for documentation only (will change in the future)
    aux_features: List[str] = None,
        Auxillary features both encoder and decoder use, for documentation only (will change in the future)
    scalers: sklearn.transformer, default = None,
        scaler(s) that has/have been used on the data, for documentation only (will change in the future)
    training_metric: str, default = "nllgaus",
        Name of the Metric used for training
    metric_options: Dict[str, Any], default = None
        Arguments used for initialization of the training metric
    optimizer_name: str: default = "adam"
        Name of the optimizer used in in Training. Names correspond to lowercase class names of optimizers in [torch.optim](https://pytorch.org/docs/stable/optim.html)
    early_stopping_patience: int, default = 7
        Number of optimization steps without improvement until training is terminated
    early_stopping_margin: float, default = 0.0
        Minimum difference of performance to be regarded an improvement.
    learning_rate: float, default = 1e-4,
        Learning rate used during training
    max_epochs: int, default = 100
        Maximum number of epochs done during training. One epoch corresponds to one go thorugh all the training data.
    forecast_horizon: int, default = 24
        Number of timesteps in the future that are to be predicted.
    model_class: str, default = "recurrent",
        Identifier of which class is to be used as model
    model_parameters: Dict[str, Any], default = None,
        Parameters used as keyword arguments during model creation
    batch_size: int = 1,
        Number of samples in each batch during Training
    history_horizon: int = 24,
        Number of timesteps in the past that are considered when making a prediciton.
    """

    def __init__(
        self,
        name: str = "model",
        target_id: Union[str, int] = None,
        encoder_features: List[str] = None,
        decoder_features: List[str] = None,
        aux_features: List[str] = None,
        scalers=None,
        training_metric: str = None,
        metric_options: Dict[str, Any] = None,
        optimizer_name: str = "adam",
        early_stopping_patience: int = 7,
        early_stopping_margin: float = 0.0,
        learning_rate: float = 1e-4,
        max_epochs: int = 100,
        forecast_horizon: int = 24,
        model_class: str = "recurrent",
        model_parameters: Dict[str, Any] = None,
        batch_size: int = 1,
        history_horizon: int = 24,
    ):
        self.initialzed = False
        self.last_training = None
        self.model = None
        self.name = "model"
        self.target_id = ["target"]
        self.encoder_features = []
        self.decoder_features = []
        self.aux_features = []
        self.scalers = None
        self.set_loss(loss="nllgauss", loss_options={})

        self.optimizer_name = "adam"
        self.early_stopping_patience = 7
        self.early_stopping_margin = 0.0
        self.learning_rate = 1e-4
        self.max_epochs = 100
        self.forecast_horizon = 24
        self.model_class = "recurrent"
        self.model_parameters = {}
        self.batch_size = 1
        self.history_horizon = 24

        self.update(
            name=name,
            target_id=target_id,
            encoder_features=encoder_features,
            decoder_features=decoder_features,
            aux_features=aux_features,
            scalers=scalers,
            training_metric=training_metric,
            metric_options=metric_options,
            optimizer_name=optimizer_name,
            early_stopping_patience=early_stopping_patience,
            early_stopping_margin=early_stopping_margin,
            learning_rate=learning_rate,
            max_epochs=max_epochs,
            forecast_horizon=forecast_horizon,
            model_class=model_class,
            model_parameters=model_parameters,
            batch_size=batch_size,
            history_horizon=history_horizon,
        )

    def get_model_config(self):
        """Get the model specfic attributes of the wrapper.

        Returns
        -------
        Dict[str, Union[str,Dict]]
            Parameters corresponding to the wrapped model.
        """
        return {
            "model_class": self.model_class,
            "model_parameters": self.model_parameters,
        }

    def get_training_config(self):
        """Get the training specfic attributes of the wrapper.

        Returns
        -------
        Dict[str, Union[str,int,float]]
            Parameters corresponding to the traingin setup.
        """
        return {
            "optimizer_name": self.optimizer_name,
            "early_stopping_patience": self.early_stopping_patience,
            "early_stopping_margin": self.early_stopping_margin,
            "learning_rate": self.learning_rate,
            "max_epochs": self.max_epochs,
            "batch_size": self.batch_size,
            "history_horizon": self.history_horizon,
        }

    def update(
        self,
        name: str = None,
        target_id: Union[str, int] = None,
        encoder_features: List[str] = None,
        decoder_features: List[str] = None,
        aux_features: List[str] = None,
        scalers=None,
        training_metric: str = None,
        metric_options: Dict[str, Any] = None,
        optimizer_name: str = None,
        early_stopping_patience: int = None,
        early_stopping_margin: float = None,
        learning_rate: float = None,
        max_epochs: int = None,
        forecast_horizon: int = None,
        model_class: str = None,
        model_parameters: Dict[str, Any] = None,
        batch_size: int = None,
        history_horizon: int = None,
        **_,
    ):
        """Change any attribute of the modelwrapper.
        Unknown keyword arguments are silently discarded.
        It is not possible to set attributes to None.

        Paramers
        --------
        See. Class description

        """
        if name is not None:
            self.name = name
        if target_id is not None:
            self.target_id = target_id
        if encoder_features is not None:
            self.encoder_features = deepcopy(encoder_features)
        if decoder_features is not None:
            self.decoder_features = deepcopy(decoder_features)
        if aux_features is not None:
            self.aux_features = deepcopy(aux_features)
        if scalers is not None:
            self.scalers = scalers
        if training_metric is not None:
            self.set_loss(loss=training_metric, loss_options=metric_options)
        if optimizer_name is not None:
            self.optimizer_name = optimizer_name
        if early_stopping_patience is not None:
            self.early_stopping_patience = early_stopping_patience
        if early_stopping_margin is not None:
            self.early_stopping_margin = early_stopping_margin
        if learning_rate is not None:
            self.learning_rate = learning_rate
        if max_epochs is not None:
            self.max_epochs = max_epochs
        if forecast_horizon is not None:
            self.forecast_horizon = forecast_horizon
        if model_class is not None:
            self.model_class = model_class
        if model_parameters is not None:
            if self.model_parameters == {}:
                self.model_parameters = model_parameters
            else:
                self.model_parameters[self.model_class].update(
                    model_parameters[self.model_class]
                )
        if batch_size is not None:
            self.batch_size = batch_size
        if history_horizon is not None:
            self.history_horizon = history_horizon

        self.initialzed = False
        return self

    @property
    def loss_metric(self) -> metrics.Metric:
        return metrics.get_metric(self._loss, **self._loss_options)

    @loss_metric.setter
    def loss_metric(self, var):
        raise AttributeError("Can't set loss manually, use .set_loss() instead")

    def set_loss(self, loss: str, loss_options=None):
        """Change the Metric used for training.
        If you do not retrain afterwards this will lead to unexpected behaviour,
        as model output and metric input do not represent the same thing.

        Parameters
        ----------
        loss: str
            String identifier of the desired metric
        loss_options: Dict[str,Any]
            Keyword arguments for intializing the metric
        """
        if not isinstance(loss, str):
            raise AttributeError(
                "Set the loss using the string identifier of the metric."
            )
        if loss is None:
            self._loss = None
            self._loss_options = None
            self.output_labels = None
            return self
        if loss_options is None:
            loss_options = {}
        self._loss = loss
        self._loss_options = deepcopy(loss_options)
        self.output_labels = deepcopy(self.loss_metric.input_labels)
        return self

    def copy(self):
        temp_mh = ModelWrapper(
            name=self.name,
            target_id=self.target_id,
            encoder_features=self.encoder_features,
            decoder_features=self.decoder_features,
            aux_features=self.aux_features,
            scalers=self.scalers,
            training_metric=self._loss,
            metric_options=self._loss_options,
            optimizer_name=self.optimizer_name,
            early_stopping_patience=self.early_stopping_patience,
            early_stopping_margin=self.early_stopping_margin,
            learning_rate=self.learning_rate,
            max_epochs=self.max_epochs,
            forecast_horizon=self.forecast_horizon,
            model_class=self.model_class,
            model_parameters=self.model_parameters,
            batch_size=self.batch_size,
            history_horizon=self.history_horizon,
        )
        # TODO include trainingparameters
        temp_mh.init_model()
        if self.model is not None:
            temp_mh.model.load_state_dict(self.model.state_dict())
            temp_mh.initialzed = self.initialzed
        return temp_mh

    def init_model(self):
        """Initializes the wrapped model."""
        logger.debug(f"{self.model_class = }")
        logger.debug(f"{self.model_parameters = }")

        # TODO adjust models according to "None means default", then replace
        model_parameters = self.model_parameters[self.model_class]
        # model_parameters = self.model_parameters.get(self.model_class)
        # if model_parameters is None:
        #     logger.warning(f"No configuration for model class {self.model_class}. Using default")
        #     model_parameters = {}

        if self.model_class == "simple_transformer":
            if len(self.target_id) > 1:
                raise NotImplementedError(
                    "simpletransformer does not support multiple target_ids yet."
                )
            n_heads = model_parameters.get("max_n_heads", 1)
            f_size = max(len(self.encoder_features), len(self.decoder_features)) + len(
                self.aux_features
            )

            # Count down until f_size is a multiple of n_heads
            while (f_size // n_heads) * n_heads < f_size:
                n_heads -= 1
            if n_heads != model_parameters.get("max_n_heads", 1):
                logger.info(
                    f"Number of heads was not a divisor of number of features. {n_heads = }"
                )
            self.model = models.Transformer(
                feature_size=f_size,
                num_layers=model_parameters.get("num_layers"),
                dropout=model_parameters.get("dropout"),
                n_heads=n_heads,
                out_size=len(self.output_labels),
                dim_feedforward=model_parameters.get("dim_feedforward", 1024),
                # n_target_features=len(self.target_id),
                encoder_features=self.encoder_features,
                decoder_features=self.decoder_features,
            )
        elif self.model_class == "recurrent":
            self.model = models.EncoderDecoder(
                enc_size=len(self.encoder_features) + len(self.aux_features),
                dec_size=len(self.decoder_features) + len(self.aux_features),
                out_size=len(self.output_labels),
                n_target_features=len(self.target_id),
                dropout_fc=model_parameters.get("dropout_fc"),
                dropout_core=model_parameters.get("dropout_core"),
                rel_linear_hidden_size=model_parameters.get("rel_linear_hidden_size"),
                rel_core_hidden_size=model_parameters.get("rel_core_hidden_size"),
                core_net=model_parameters.get("core_net"),
                relu_leak=model_parameters.get("relu_leak"),
                core_layers=model_parameters.get("core_layers"),
            )
        elif self.model_class == "autoencoder":
            self.model = models.AutoEncoder(
                enc_size=len(self.encoder_features) + len(self.aux_features),
                dec_size=0,  # len(self.aux_features),  # TODO should the autoencoder have access to aux features
                # Aux_features will be reconstructed to which we might not want but is necessary to compare to dual model
                out_size=len(self.output_labels),
                dropout_fc=model_parameters.get("dropout_fc"),
                dropout_core=model_parameters.get("dropout_core"),
                rel_linear_hidden_size=model_parameters.get("rel_linear_hidden_size"),
                rel_core_hidden_size=model_parameters.get("rel_core_hidden_size"),
                core_net=model_parameters.get("core_net"),
                relu_leak=model_parameters.get("relu_leak"),
                core_layers=model_parameters.get("core_layers"),
            )
            self.set_loss("autoencoderloss", {"loss_metric": self.loss_metric})

        elif self.model_class == "dualmodel":
            metr = self.loss_metric
            if not isinstance(metr, metrics.AutoEncoderLoss):
                self.set_loss(
                    "dualmodelloss",
                    {"forecast_loss": metr, "reconstruction_loss": metr},
                )
            self.model = models.DualModel(
                enc_size=len(self.encoder_features) + len(self.aux_features),
                dec_size=len(self.decoder_features)
                + len(
                    self.aux_features
                ),  # TODO should the autoencoder have access to aux features
                out_size=(len(self.output_labels[0]), len(self.output_labels[1])),
                n_target_features=len(self.target_id),
                dropout_fc=model_parameters.get("dropout_fc"),
                dropout_core=model_parameters.get("dropout_core"),
                rel_linear_hidden_size=model_parameters.get("rel_linear_hidden_size"),
                rel_core_hidden_size=model_parameters.get("rel_core_hidden_size"),
                core_net=model_parameters.get("core_net"),
                relu_leak=model_parameters.get("relu_leak"),
                core_layers=model_parameters.get("core_layers"),
            )
            # Use the requested loss for forecast in dual model

        for param in self.model.parameters():
            torch.nn.init.uniform_(param.data, -0.08, 0.08)
        self.initialzed = True
        return self

    def to(self, device: Union[str, int]):
        """Moves the model to a specific device.
        Parameters
        ----------
        device: Union[str,int]
            Name or id of the device where the model should be moved.
        """
        self._device = device
        if self.model:
            self.model.to(device)
        return self

    def run_training(
        self,  # Maybe use the datahandler as "data" which than provides all the data_loaders,for unifying the interface.
        train_data,
        validation_data,
        trial_id=None,
        log_tb=None,
        batch_size: int = None,
    ):
        """
        Train the wrapped model using the given parameters specified in the ModelWrapper.
        Log the training data optionally using TensorBoard's SummaryWriter

        Parameters
        ----------
        train_data : proloaf.tensorloader.TimeSeriesData
            The training data
        validation_data : proloaf.tensorloader.TimeSeriesData
            The validation data
        logging_tb : SummaryWriter, default = None
            If specied this TensorBoard SummaryWriter will be used for logging during the training
        trial_id : Any , default = None
            Identifier for a specific training run. Will be consecutive number if not specified
        Returns
        -------
        self
        """
        if self.model is None:
            raise AttributeError("Model was not initialized")
        training_run = TrainingRun(
            self.model,
            id=trial_id,
            optimizer_name=self.optimizer_name,
            train_data=train_data,
            validation_data=validation_data,
            early_stopping_patience=self.early_stopping_patience,
            early_stopping_margin=self.early_stopping_margin,
            learning_rate=self.learning_rate,
            batch_size=batch_size,
            loss_function=self.loss_metric,
            max_epochs=self.max_epochs,
            history_horizon=self.history_horizon,
            log_tb=log_tb,
            device=self._device,
        )
        training_run.train()
        values = {
            "hparam/hp_total_time": training_run.training_start_time
            - training_run.training_end_time,
            "hparam/score": training_run.validation_loss,
        }
        training_run.remove_data()
        self.last_training = training_run
        return self

    def predict(
        self,
        inputs_enc: torch.Tensor,
        inputs_enc_aux: torch.Tensor,
        inputs_dec: torch.Tensor,
        inputs_dec_aux: torch.Tensor,
        last_value: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get the predictions for the wrapped model

        Parameters
        ----------
        inputs_enc : torch.Tensor
            Contains the input data for the encoder
        inputs_enc_aux : torch.Tensor
            Contains auxillary input data for the encoder
        inputs_dec : torch.Tensor
            Contains the input data for the decoder
        inputs_dec_aux : torch.Tensor
            Contains auxillary input data for the decoder
        last_value : torch.Tensor
            Last known target value(s) in history horizon

        Returns
        -------
        torch.Tensor
            Results of the prediction, 3D-Tensor of shape (batch_size, timesteps, predicted features)
        """
        if not self.initialzed:
            raise RuntimeError(
                "The model has not been initialized. Use .init_model() to do that"
            )

        self.to(inputs_enc.device)
        try:
            val, _ = self.model(
                inputs_enc, inputs_enc_aux, inputs_dec, inputs_dec_aux, last_value
            )
        except RuntimeError as err:
            raise RuntimeError(
                "The prediction could not be executed likely because to much memory is needed. Try to reduce batch size of the input."
            ) from err
        return val


class ModelHandler:
    """
    Utility wrapper for the model, implementing sklearns Predictor interface
    in addition to providing methods for initializing and handling the
    forecasting model.

    Parameters
    ----------
    config: Dict[str,Any]
        Dictonary with definitions for model and training, keys are see example config files for possible setup.
    work_dir: str, default = directory of excuted script (argv[0]).
        Folder that is used as reference for relative pathes in the config.
    tuning_config: Dict[str,Any], default = None
        Dict defining what parameters should be tunned and which sampling functions
        are to be used. If none is provided it can be pointed to from the basic config.
        See example file for syntax.
    scalers : proloaf.datahandler.MultiScaler, default = None
        Scalers with which the dataset is scaled before training. Currently this is for storage only,
        the scaler will not automatically be applied to the data during prediciton.
    loss: str, default = nllgauss
        Lowercase name of the metric used as loss during training. Available metrics can be found in `proloaf.metrics`
    loss_kwargs: Dict[str,Any], default = {}
        Keyword arguments that are provided to the metric during its initialization. Depends on the chosen loss.
    device: Union[str,int], default = "cpu"
        Device on which the model should be trained. Values are equivalent to the ones used in PyTorch.

    Notes
    -----
    Reference: https://scikit-learn.org/stable/developers/develop.html
    """

    def __init__(
        self,
        config: dict,
        work_dir: str = None,
        tuning_config: dict = None,
        scalers: sklearn.transformer = None,
        loss: str = "nllgauss",
        loss_kwargs: dict = {},
        device: str = "cpu",
    ):
        self.work_dir = (
            work_dir
            if work_dir is not None
            else os.path.dirname(os.path.abspath(sys.argv[0]))
        )
        self.config = deepcopy(config)
        self.tuning_config = deepcopy(tuning_config)

        self._model_wrap: ModelWrapper = ModelWrapper(
            name=config.get("model_name"),
            target_id=config.get("target_id"),
            encoder_features=config.get("encoder_features"),
            decoder_features=config.get("decoder_features"),
            aux_features=config.get("aux_features"),
            scalers=scalers,
            training_metric=loss,
            metric_options=loss_kwargs,
            optimizer_name=config.get("optimizer_name", "adam"),
            early_stopping_patience=int(config.get("early_stopping_patience", 7)),
            early_stopping_margin=float(config.get("early_stopping_margin", 0.0)),
            learning_rate=float(config.get("learning_rate", 1e-4)),
            max_epochs=int(config.get("max_epochs", 100)),
            forecast_horizon=config.get("forecast_horizon", 168),
            model_class=config.get("model_class", "recurrent"),
            model_parameters=config.get("model_parameters"),
            batch_size=int(config.get("batch_size", 1)),
            history_horizon=int(config.get("history_horizon", 1)),
        )
        self.to(device)

        if config.get("exploration", False):
            if not self.tuning_config:
                raise AttributeError(
                    "Exploration was requested but no configuration for it was provided. Define the relative path to the hyper parameter tuning config as 'exploration_path' in the main config or provide a dict to the modelhandler"
                )

    def get_config(self):
        """Get full updated config of this Modelhandler.

        Returns
        -------
        Dict[str,Any]
            Configuration of this Modelhandler
        """
        config = deepcopy(self.config)
        config.update(self.model_wrap.get_model_config())
        config.update(self.model_wrap.get_training_config())
        return config

    @property
    def model_wrap(self):
        return self._model_wrap

    @model_wrap.setter
    def model_wrap(self, model_wrap: torch.nn.Module):
        self._model_wrap = model_wrap
        if self._model_wrap:
            self._model_wrap.to(self._device)

    def to(self, device):
        """Move the handled model to the specified device.

        Parameters
        ----------
        device: Union[str,int]
            Device on which the model should be trained. Values are equivalent to the ones used in PyTorch.

        """
        self._device = device
        if self.model_wrap:
            self.model_wrap.to(device)
        return self

    def tune_hyperparameters(
        self,
        train_data: proloaf.tensorloader.TimeSeriesData,
        validation_data: proloaf.tensorloader.TimeSeriesData,
    ):
        """
        Explores the searchspace of hyperparameters defined during initialization by training
        multiple models and comparing them. The best configuration is then used to update the config.


        Notes
        -----
        Hyperparameter exploration
        - Possible hyperparameters are: target_id, encoder_features, decoder_features,
          optimizer_name, early_stopping_patience,early_stopping_margin,learning_rate
          max_epochs,model_class,forecast_horizon, batch_size, and every model_parameters

        Parameters
        ----------
        train_data: proloaf.tensorloader.TimeSeriesData
            Prepared data for training.
        validation_data: proloaf.tensorloader.TimeSeriesData
            Prepared data for validation after each epoch.

        Returns
        -------
        self
        """
        logger.info(
            "Max. number of iteration trials for hyperparameter tuning: {!s}".format(
                self.tuning_config["number_of_tests"]
            )
        )
        study = self.make_study()
        study.optimize(
            self.tuning_objective(
                self.tuning_config["settings"],
                train_data=train_data,
                validation_data=validation_data,
            ),
            n_trials=self.tuning_config.get("number_of_tests", None),
            timeout=self.tuning_config.get("timeout", None),
        )

        logger.info("Number of finished trials: {!s}".format(len(study.trials)))
        trials_df = study.trials_dataframe()

        if not os.path.exists(os.path.join(self.work_dir, self.config["log_path"])):
            os.mkdir(os.path.join(self.work_dir, self.config["log_path"]))
        if not os.path.exists(os.path.join(self.work_dir, self.config["log_path"])):
            os.mkdir(
                os.path.join(
                    self.work_dir,
                    self.config["log_path"],
                    f"tuning-results_{self.model_wrap.name}",
                )
            )
        trials_df.to_csv(
            os.path.join(
                self.work_dir,
                self.config["log_path"],
                f"tuning-results_{self.model_wrap.name}",
            ),
            index=False,
        )

        trial = study.best_trial
        logger.info("Best trial")
        logger.info(" Value: {!s}\n".format(trial.value))
        logger.info("Params:")
        for key, value in trial.params.items():
            logger.info("    {}: {}".format(key, value))

        # The order here is important as .get_config() pulls parameters directly from model_wrap
        self.model_wrap = trial.user_attrs["wrapped_model"]
        self.config.update(self.get_config())

        return self

    def select_model(
        self,
        data: proloaf.tensorloader.TimeSeriesData,
        models: List[ModelWrapper],
        loss: metrics.Metric,
    ):
        """Select the best model from a list of models.

        Parameters
        ----------
        data: proloaf.tensorloader.TimeSeriesData
            Data on which the Models are benchmarked
        models: List[ModelWrapper]
            List of models from which the best is selected
        loss: proloaf.metrics.Metric
            Loss metric used as perfomance criterion

        Returns
        -------
        ModelWrapper
            The most performant model
        """
        for model in models:  # check if all models have same target
            if model.target_id != models[0].target_id:
                raise Exception(
                    "At least one model has a different target_id than the other ones. "
                    "You can only compare models with the same target_id"
                )
        perf_df = self.benchmark(data, models, [loss], avg_over="all")
        logger.info(f"Performance was:\n {perf_df}")
        idx = perf_df.iloc[0].to_numpy().argmin()
        self.model_wrap = models[idx]
        logger.info(f"selected {self.model_wrap.name}")
        return self.model_wrap

    @staticmethod
    def benchmark(
        data: proloaf.tensorloader.TimeSeriesData,
        models: List[ModelWrapper],
        test_metrics: List[metrics.Metric],
        avg_over: Union[Literal["time"], Literal["sample"], Literal["all"]] = "all",
    ) -> pd.DataFrame:
        """Benchmark a list of models. Calculates all the metrics for all models.

        Parameters
        ----------
        data: proloaf.tensorloader.TimeSeriesData
            Data on which the Models are benchmarked
        models: List[ModelWrapper]
            List of models from which the best is selected
        test_metrics: List[proloaf.metrics.Metric]
            All the metrics that should be calculated
        avg_over: Union["time","sample","all"], default = "all"
            Axis over which the metrics should be averaged
            - "time": Metric will be calculated for each sample separately
            - "sample": Metric will be calculated for each timestep separately
            - "all": Metric will averaged over the whole dataset

        Returns
        -------
        pandas.DataFrame
            Perfromances of the models
        """

        with torch.no_grad():
            bench = {}
            for model in models:
                dataloader = data.make_data_loader(
                    encoder_features=model.encoder_features,
                    decoder_features=model.decoder_features,
                    batch_size=None,
                    shuffle=False,
                )
                (
                    inputs_enc,
                    inputs_enc_aux,
                    inputs_dec,
                    inputs_dec_aux,
                    last_value,
                    targets,
                ) = next(iter(dataloader))
                # Dataloader is setup to be only one batch

                # Dataloader is setup to be only one batch

                logger.info(f"benchmarking {model.name}")
                quantiles = model.loss_metric.get_quantile_prediction(
                    predictions=model.predict(
                        inputs_enc=inputs_enc,
                        inputs_enc_aux=inputs_enc_aux,
                        inputs_dec=inputs_dec,
                        inputs_dec_aux=inputs_dec_aux,
                        last_value=last_value,
                    ),
                    target=targets,
                    inputs_enc=inputs_enc,
                    inputs_enc_aux=inputs_enc_aux,
                )
                performance = list()

                for metric in test_metrics:
                    if isinstance(model.loss_metric, metrics.AutoEncoderLoss):
                        metric = metrics.AutoEncoderLoss(metric)
                    if isinstance(
                        model.loss_metric, metrics.DualModelLoss
                    ) and not isinstance(metric, metrics.DualModelLoss):
                        quants = quantiles[0]
                    else:
                        quants = quantiles
                    performance.append(
                        metric.from_quantiles(
                            target=targets,
                            quantile_prediction=quants,
                            avg_over=avg_over,
                            inputs_enc=inputs_enc,
                            inputs_enc_aux=inputs_enc_aux,
                        )
                        .squeeze()
                        .cpu()
                        .numpy()
                    )
                performance = np.array(performance)
                if len(performance.shape) == 1:
                    performance = performance[np.newaxis, ...]
                else:
                    performance = performance.T

                df = pd.DataFrame(
                    data=performance, columns=[met.id for met in test_metrics]
                )
                name = model.name
                i = 1
                while name in bench:
                    name = model.name + f"({i})"
                    i = i + 1
                bench[name] = df
        return pd.concat(bench.values(), keys=bench.keys(), axis=1)

    def run_training(
        self,  # Maybe use the datahandler as "data" which than provides all the data_loaders,for unifying the interface.
        train_data: proloaf.tensorloader.TimeSeriesData,
        validation_data: proloaf.tensorloader.TimeSeriesData,
        trial_id=None,
        hparams={},
    ):
        """
        Train the internal model using the given data.

        Parameters
        ----------
        train_data : proloaf.tensorloader.TimeSeriesData
            The training data
        validation_data : proloaf.tensorloader.TimeSeriesData
            The validation data
        hparams : Dict[str,Any], default = {}
            Parameter updates for model and training
        trial_id : Any , default = None
            Identifier for a specific training run. Will be consecutive number if not specified

        Returns
        -------
        ModelWrapper
            The trained model
        """
        # to track the validation loss as the model trains

        config = deepcopy(self.config)
        # XXX this is a quick and dirty solution to enable continuing the training of a loaded model, should be refactored
        if hparams:
            model_parameters = hparams.pop("model_parameters", None)
            config.update(hparams)

            if model_parameters is not None:
                hparams["model_parameters"] = model_parameters
                for model_class, params in model_parameters.items():
                    config["model_parameters"][model_class].update(params)

            temp_model_wrap: ModelWrapper = (
                self.model_wrap.copy().update(**hparams).init_model()
            ).to(self._device)
        else:
            # XXX if no hparams the model is copied
            temp_model_wrap = self.model_wrap.copy().to(self._device)
        tb = log_tensorboard(
            work_dir=self.work_dir,
            exploration=self.config["exploration"],
            trial_id=trial_id,
        )
        temp_model_wrap.run_training(
            train_data, validation_data, trial_id, tb, config.get("batch_size")
        )

        values = {
            "hparam/hp_total_time": temp_model_wrap.last_training.training_end_time
            - temp_model_wrap.last_training.training_start_time,
            "hparam/score": temp_model_wrap.last_training.validation_loss,
        }
        end_tensorboard(tb, hparams, values)
        return temp_model_wrap

    @timer(logger)
    def fit(
        self,
        train_data: proloaf.tensorloader.TimeSeriesData,
        validation_data: proloaf.tensorloader.TimeSeriesData,
        exploration: bool = None,
    ):
        """
        Train a prediciton model with or without hyperparameter tuning

        Parameters
        ----------
        train_data: proloaf.tensorloader.TimeSeriesData
            The training data
        validation_data: proloaf.tensorloader.TimeSeriesData
            The validation data
        exploration: bool , default = False
            If True hyperparameters are optimized

        Returns
        -------
        self
        """
        if exploration is None:
            exploration = self.config.get("exploration", False)
        logger.debug(f"{exploration = }")
        if exploration:
            if not self.tuning_config:
                raise AttributeError(
                    "Hyper parameters are to be explored but no config for tuning was provided."
                )
            self.tune_hyperparameters(train_data, validation_data)
        else:
            self.model_wrap = self.run_training(
                train_data,
                validation_data,
                trial_id="main",
            )
        return self

    @timer(logger)
    def predict(
        self,
        inputs_enc: torch.Tensor,
        inputs_enc_aux: torch.Tensor,
        inputs_dec: torch.Tensor,
        inputs_dec_aux: torch.Tensor,
        last_value: torch.Tensor,
    ):
        """
        Get the predictions for the given model and data

        Parameters
        ----------
        inputs_enc : torch.Tensor
            Contains the input data for the encoder
        inputs_enc_aux : torch.Tensor
            Contains auxillary input data for the encoder
        inputs_dec : torch.Tensor
            Contains the input data for the decoder
        inputs_dec_aux : torch.Tensor
            Contains auxillary input data for the decoder

        Returns
        -------
        torch.Tensor
            Results of the prediction, 3D-Tensor of shape (batch_size, timesteps, predicted features)

        """
        if self.model_wrap is None:
            raise RuntimeError("No model has been created to perform a prediction with")
        return self.model_wrap.predict(
            inputs_enc=inputs_enc,
            inputs_enc_aux=inputs_enc_aux,
            inputs_dec=inputs_dec,
            inputs_dec_aux=inputs_dec_aux,
            last_value=last_value,
        )

    @staticmethod
    def load_model(path: str = None, locate: str = "cpu") -> ModelWrapper:
        """Loads model from given path.

        Parameters
        ----------
        path: str
            path to the saved model

        """
        inst = torch.load(path, map_location=torch.device(locate))
        if not isinstance(inst, ModelWrapper):
            raise RuntimeError(
                f"you tryied to load from '{path}' but the object was not a ModelWrapper"
            )
        return inst

    @staticmethod
    def save_model(model: ModelWrapper, path: str):
        """Saves model to file. Same as `torch.save(...)`.

        Parameters
        ----------
        model: ModelWrapper
            The model to be saved
        path: str
            Path to the file where the model is saved
        """
        torch.save(model, path)

    def save_current_model(self, path: str):
        """Saves internal model to file`.

        Parameters
        ----------
        path: str
            Path to the file where the model is saved
        """
        if self.model_wrap.model is None:
            raise RuntimeError(
                "The Model is not initialized and can thus not be saved."
            )
        torch.save(self.model_wrap, path)
        return self

    @staticmethod
    def make_study(
        direction: Union[Literal["minimize"], Literal["maximize"]] = "minimize",
        pruner: optuna.pruners.BasePruner = optuna.pruners.MedianPruner(),
    ):
        """
        Creates a optuna study for hyperparameter tuning.

        Parameters
        ----------
        direction: Union["minimize", "maximize"], default = "minimize"
            Defines if the study should minimize odr maximize the objective
        pruner: optuna.pruners.BasePruner, default = optuna.pruners.MedianPruner()
            Strategy to terminate less promissing trials ahead of time

        Returns
        -------
        optuna.study.Study
            The study object for optimizing hyperparameters
        Raises
        ------
        ValueError
            If direction is not parseable
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
        tuning_settings: Dict[str, Any],
        train_data: proloaf.tensorloader.TimeSeriesData,
        validation_data: proloaf.tensorloader.TimeSeriesData,
    ):
        """
        Creates a callable to evaluate a parameter configuration.
        The callable initializes the trial depending on the tuning_config.
        It then trains a model with the created parameters and returns the validation error.

        Parameters
        ----------
        tuning_config: Dict[str,Any]
            Dict defining what parameters should be tunned and which sampling functions
            are to be used. See example file for syntax.
        train_data : proloaf.tensorloader.TimeSeriesData
            The training data
        validation_data : proloaf.tensorloader.TimeSeriesData
            The validation data
        Returns
        -------
        Callable
            Function that runs a trial
        Raises
        ------
        optuna.exceptions.TrialPruned
            If the trial was pruned
        """

        def search_params(trial: optuna.trial.Trial):
            hparams = {}
            for key, hparam in tuning_settings.items():
                if key == "model_parameters":
                    continue

                logger.info("Creating parameter: {!s}".format(key))
                func_generator = getattr(trial, hparam["function"])
                hparams[key] = func_generator(**(hparam["kwargs"]))

            model_class = self.config.get("model_class")

            if "model_parameters" in tuning_settings:
                hparams["model_parameters"] = {model_class: {}}
                for key, hparam in (
                    tuning_settings["model_parameters"].get(model_class, {}).items()
                ):
                    logger.info("Creating parameter: {!s}".format(key))
                    func_generator = getattr(trial, hparam["function"])
                    hparams["model_parameters"][model_class][key] = func_generator(
                        **(hparam["kwargs"])
                    )

            model_wrap = self.run_training(
                train_data,
                validation_data,
                hparams=hparams,
                trial_id=trial.number,
            )
            model_wrap.last_training.remove_data()
            trial.set_user_attr("wrapped_model", model_wrap)
            trial.set_user_attr("training_run", model_wrap.last_training)
            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            return model_wrap.last_training.validation_loss

        return search_params


class TrainingRun:
    """Class representing a single training of a PyTorch model.

    Parameters
    ----------
    model: torch.nn.Module,
        PyTorch model to be trained.
    loss_function: Callable[[torch.Tensor, torch.Tensor], float],
        Callable that returns the loss from a prediction and real values. Lower is better.
    train_data: proloaf.tensorloader.TimeSeriesData,
        Data used for training the model
    validation_data: proloaf.tensorloader.TimeSeriesData, default = None
        Data used for validating model performance. If None is provided no validatin will be performed.
    optimizer_name: str, default = "adam"
        Lowercase class name of the `torch.optim` class to be used as optimizer.
    learning_rate: float, default = 1e-4
        Multiplier for the stepsize in the gradient descend.
    max_epochs: int = 100,
        maximum number of of iterations over the whole dataset.
    early_stopping_patience: int = 7,
        Number of epochs without improvement before stopping the training.
    early_stopping_margin: float = 0,
        Minimum improvement to be considered in early stopping.
    batch_size: int, default = 100,
        Number of samples per batch. Defaults to all samples in a single batch.
    history_horizon: int, default = 24
        Timesteps of data given to the encoder.
    forecast_horizon: int = None,
        Timesteps of data given to the decoder.
    id: Union[str, int] = None,
        identifier for a training run, consecutive number by default, only stored in memory (reset on each restart).
    device: str = "cpu",
        Device the training is run on. See `torch.Tensor.to(...)` for additional inforamtion.
    log_tb: torch.utils.tensorboard.SummaryWriter, default = None,
        Tensorboard writer to log the training. If none Training will not be logged.

    Attributes
    ----------
    optimizer: torch.optim.Optimizer
        Optimizer object that is/was used for training.
    step_counter: int
        Number of optimization steps performed.
    training_start_time: datetime.datetime
        Time the training was started or None if not trained.
    training_end_time: datetime.datetime
        Time the training finished or None if no training finished.
    validation_loss: float
        Error on the training dataset.
    training_loss:float
        Error during training.
    n_epochs: int || None
        Number of epochs run in the training.



    All the parameters are saved in attributes of the same name.
    """

    _next_id = 0

    def __init__(
        self,
        model: torch.nn.Module,
        loss_function: Callable[[torch.Tensor, torch.Tensor], float],
        train_data: proloaf.tensorloader.TimeSeriesData,
        validation_data: proloaf.tensorloader.TimeSeriesData = None,
        optimizer_name: str = "adam",
        learning_rate: float = 1e-4,
        max_epochs: int = 100,
        early_stopping_patience: int = 7,
        early_stopping_margin: float = 0.0,
        batch_size: int = None,
        history_horizon: int = 24,
        forecast_horizon: int = 24,
        id: Union[str, int] = None,
        device: str = "cpu",
        log_tb: torch.utils.tensorboard.SummaryWriter = None,
    ):
        if id is None:
            self.id = TrainingRun._next_id
            TrainingRun._next_id += 1
        else:
            self.id = id
        self.model = model
        self.train_ds = train_data
        self.train_dl = None
        self.validation_ds = validation_data
        self.validation_dl = None
        self.history_horizon = history_horizon
        self.forecast_horizon = forecast_horizon
        self.validation_loss = np.inf
        self.training_loss = np.inf
        self.optimizer_name = optimizer_name
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.set_optimizer(optimizer_name, learning_rate)
        self.loss_function = loss_function
        self.step_counter = 0
        self.max_epochs = max_epochs
        self.n_epochs = None
        self.early_stopping = EarlyStopping(
            patience=early_stopping_patience, delta=early_stopping_margin
        )
        self.log_tb = log_tb
        self.training_start_time = None
        self.training_end_time = None
        self.to(device)

    def get_config(self):
        """Get a dict with all relevant attributes from the config.

        Returns
        -------
        Dict[str,Any]
            Part of the config related to the training.
        """
        return {
            "optimizer_name": self.optimizer_name,
            "max_epochs": self.max_epochs,
            "batch_size": self.batch_size,
            "history_horizon": self.history_horizon,
            "forecast_horizon": self.forecast_horizon,
            "early_stopping_patience": self.early_stopping.patience,
            "early_stopping_margin": self.early_stopping.delta,
            "learning_rate": self.learning_rate,
        }

    def remove_data(self):
        """Remove the data from this TrainingRun.
        This saves memory and disk space when saving, it can not be rerun afterwards without setting new data.
        """
        self.train_ds = None
        self.train_dl = None
        self.validation_ds = None
        self.validation_dl = None
        return self

    def set_optimizer(self, optimizer_name: str, learning_rate: float):
        """
        Specify which optimizer to use during training.

        Initialize a torch.optim optimizer for the given model based on the specified name and learning rate.

        Parameters
        ----------
        optimizer_name : string or None, default = 'adam'
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
        #  params = filter(lambda p: p.requires_grad, self.model.parameters())
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
        """Reset the Training. The model state will not be reset."""
        self.step_counter = 0
        self.validation_loss = np.inf
        self.training_loss = np.inf

    def step(self):
        """Perform optimization of a single run through all batches."""
        self.training_loss = 0.0

        self.model.train()
        # train step
        for (
            inputs_enc,
            inputs_enc_aux,
            inputs_dec,
            inputs_dec_aux,
            last_value,
            targets,
        ) in self.train_dl:
            prediction, _ = self.model(
                inputs_enc=inputs_enc,
                inputs_enc_aux=inputs_enc_aux,
                inputs_dec=inputs_dec,
                inputs_dec_aux=inputs_dec_aux,
                last_value=last_value,
            )
            self.optimizer.zero_grad()
            loss = self.loss_function(
                target=targets,
                predictions=prediction,
                inputs_enc=inputs_enc,
                inputs_enc_aux=inputs_enc_aux,
                inputs_dec=inputs_dec,
                inputs_dec_aux=inputs_dec_aux,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            self.optimizer.step()
            self.step_counter += 1
            self.training_loss += loss.item() / len(self.train_dl)

        return self

    def validate(self):
        """Validate model performance on an unseen dataset."""
        with torch.no_grad():
            self.model.eval()
            self.validation_loss = 0.0
            for (
                inputs_enc,
                inputs_enc_aux,
                inputs_dec,
                inputs_dec_aux,
                last_value,
                targets,
            ) in self.validation_dl:
                prediction, _ = self.model(
                    inputs_enc=inputs_enc,
                    inputs_enc_aux=inputs_enc_aux,
                    inputs_dec=inputs_dec,
                    inputs_dec_aux=inputs_dec_aux,
                    last_value=last_value,
                )
                self.validation_loss += self.loss_function(
                    target=targets,
                    predictions=prediction,
                    inputs_enc=inputs_enc,
                    inputs_enc_aux=inputs_enc_aux,
                ).item() / len(self.validation_dl)
        return self

    def train(self):
        """Run the whole training process including validation if validation data was provided.
        The training will be stopped early if the validation loss stops improving.
        """
        if not self.train_dl:
            if not self.train_ds:
                raise AttributeError("No training data provided")
            self.train_dl = self.train_ds.make_data_loader(
                history_horizon=self.history_horizon,
                forecast_horizon=self.forecast_horizon,
                batch_size=self.batch_size,
            )

        if not self.validation_dl:
            if self.validation_ds:
                self.validation_dl = self.validation_ds.make_data_loader(
                    history_horizon=self.history_horizon,
                    forecast_horizon=self.forecast_horizon,
                    batch_size=self.batch_size,
                )
        if self.log_tb:
            # is this actually for every run or model specific
            (
                inputs_enc,
                inputs_enc_aux,
                inputs_dec,
                inputs_dec_aux,
                last_value,
                targets,
            ) = next(iter(self.train_dl))

            self.log_tb.add_graph(
                self.model,
                [inputs_enc, inputs_enc_aux, inputs_dec, inputs_dec_aux, last_value],
            )
        self.training_start_time = perf_counter()
        logger.info("Begin training...")
        self.model.train()
        for epoch in range(self.max_epochs):
            t1_start = perf_counter()
            self.step()
            t1_stop = perf_counter()
            if self.validation_dl:
                self.validate()
                self.early_stopping(self.validation_loss, self.model)
            else:
                logger.warning(
                    "No validation data was provided, thus no validation was performed"
                )
            logger.info(
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
                    training_start=self.training_start_time,
                    epoch_stop=t1_stop,
                    epoch_start=t1_start,
                    next_epoch=epoch + 1,
                    step_counter=self.step_counter,
                )
            self.n_epochs = epoch + 1
            if self.early_stopping.early_stop:
                logger.info(
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
        """Model and data to specified device. See `torch.tensor.to(...)`
        for more details.
        """
        self.device = device
        if self.model:
            self.model.to(device)
        if self.train_ds:
            self.train_ds.to(device)
        if self.validation_ds:
            self.validation_ds.to(device)
        return self
