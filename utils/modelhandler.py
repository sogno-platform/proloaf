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

import utils.datahandler as dh

from datetime import datetime
from time import perf_counter  # ,localtime
from utils import models
from utils import metrics
from utils.loghandler import log_data, clean_up_tensorboard_dir, log_tensorboard, add_tb_element, end_tensorboard
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
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.temp_dir = ""

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score - self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
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
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        temp_dir = tempfile.mktemp()
        torch.save(model.state_dict(), temp_dir+'checkpoint.pt')
        self.val_loss_min = val_loss
        self.temp_dir=temp_dir


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
    def __init__(self, model = None, scalers = None, hparam_tuning = False, device = 'cpu'):
        self._model = model
        self._scalers = scalers
        self.hparam_tuning = hparam_tuning
        self.optimizer = None
        self._device = device

    def to(self,device):
        self._device = device
        self._model.to(device)
        return self
    
    def fit(self, data):
        raise NotImplemented()

    def predict(self, data):
        raise NotImplemented()

    def load(self, path:str):
        raise NotImplemented()
    
    def save(self, path:str):
        raise NotImplemented()

    def score(self,data):
        raise NotImplemented()

    def initialize_model(self):
        raise NotImplemented()
    
    @property
    def model(self):
        return self._model

    @property
    def scalers(self):
        return self._scalers
    

    def set_optimizer(self, name, learning_rate):
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
            raise AttributeError("The model has to be initialized before the optimizer is set.")

        if name == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        if name == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.5)
        if name == "adamw":
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        if name == "adagrad":
            self.optimizer = torch.optim.Adagrad(self.model.parameters(), lr=learning_rate)
        if name == "adamax":
            self.optimizer = torch.optim.Adamax(self.model.parameters(), lr=learning_rate)
        if name == "rmsprop":
            self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=learning_rate)
        return self



###################################################### REFACTOR ABOVE #############################################################################


def get_prediction(net, data_loader, horizon, number_of_targets):
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
    record_targets = torch.zeros((len(data_loader), horizon, number_of_targets))
    #TODO: try to remove loop here to make less time-consuming,
    # was inspired from original numpy version of code but torches should be more intuitive
    for i, (inputs1, inputs2, targets) in enumerate(data_loader):
        record_targets[i,:,:] = targets
        output,_ = net(inputs1, inputs2)
        if i==0:
            record_output = torch.zeros((len(data_loader), horizon, len(output)))
        for j, elementwise_output in enumerate(output):
            record_output[i, :, j] = elementwise_output.squeeze()

    return record_targets, record_output

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
    #detect criterion
    if ('nll_gaus' in str(criterion)) or ('crps' in str(criterion)):
        #loss_type = 'nll_gaus'
        expected_values = predictions[:,:,0:1] # expected_values:mu
        sigma = torch.sqrt(predictions[:,:,-1:].exp())
        #TODO: make 95% prediction interval changeable
        y_pred_upper = expected_values + 1.96 * sigma
        y_pred_lower = expected_values - 1.96 * sigma
    elif 'quantile' in str(criterion):
        #loss_type = 'pinball'
        y_pred_lower = predictions[:,:,0:1]
        y_pred_upper = predictions[:,:,1:2]
        expected_values = predictions[:,:,-1:]
    elif 'rmse' in str(criterion):
        expected_values = predictions
        rmse = metrics.rmse(targets, expected_values.unsqueeze(0))
        # In order to produce an interval covering roughly 95% of the error magnitudes,
        # the prediction interval is usually calculated using the model output ± 2 × RMSE.
        y_pred_lower = expected_values-2*rmse
        y_pred_upper = expected_values+2*rmse
    elif ('mse' in str(criterion)) or ('mape' in str(criterion)):
        # loss_type = 'mis'
        expected_values = predictions
        y_pred_lower = 0
        y_pred_upper = 1

        #TODO: add all criterion possibilities
    else:
        print('invalid criterion')

    return y_pred_upper, y_pred_lower, expected_values

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the given Pytorch object as a checkpoint.

    If is_best is true, create a copy of the object called 'model_best.pth.tar'

    Parameters
    ----------
    state : PyTorch object
        The object to save
    is_best : bool
        If true, create a copy of the saved object.
    filename : string, default = 'checkpoint.pth.tar'
        The name to give the saved object
    """
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')




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
    return (100 / best_score) * (curr_score-best_score)

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

    def __call__(self, *args,**kwargs):
        return self._func(*args,**kwargs)

def performance_test(net, data_loader, score_type='mis', option=0.05, avg_on_horizon=True, horizon=1, number_of_targets=1):
    """
    Determine the score of the given model using the specified metrics in utils.eval_metrics

    Return a single float value (total score) or a 1-D array (score over horizon) based on
    the value of avg_on_horizon

    Parameters
    ----------
    net : utils.models.EncoderDecoder
        The model for which the score is to be determined
    data_loader : utils.tensorloader.CustomTensorDataLoader
        Contains the input data and targets
    score_type : string, default = 'mis'
        The name of the metric to use for scoring. See functions in utils.eval_metrics for
        possible values.
    option : float or list, default = 0.05
        An optional parameter in case the 'mis' or 'quantile_score' functions are used. In the
        case of 'mis' it should be given the value for 'alpha'. In the case of QS, it should
        contain a list with the quantiles.
    avg_on_horizon : bool, default = true
        Determines whether the return value is a float (total score, when true) or a 1-D array
        (score over the horizon, when false).
    horizon : int, default = 1
        The horizon for the prediction
    number_of_targets : int, default = 1
        The number of targets for the prediction.

    Returns
    -------
    float or 1-D array (torch.Tensor)
        Either the total score (float) or the score over the horizon (array), depending on the
        value of avg_on_horizon
    """
    # check performance
    targets, raw_output = get_prediction(net, data_loader,horizon, number_of_targets) ##should be test data loader
    [y_pred_upper, y_pred_lower, expected_values] = get_pred_interval(raw_output, net.criterion, targets)
    #get upper and lower prediction interval, depending on loss function used for training
    if ('mis' in str(score_type)):
        output = [y_pred_upper, y_pred_lower]
        score = metrics.mis(targets, output, alpha=option, total=avg_on_horizon)
    elif ('nll_gauss' in str(score_type)):
        output = raw_output ##this is only valid if net was trained with gaussin nll or crps
        score = metrics.nll_gauss(targets, output, total=avg_on_horizon)
    elif ('quant' in str(score_type)):
        output = expected_values
        score = metrics.quantile_score(targets, output, quantiles=option, total=avg_on_horizon)
    elif ('crps' in str(score_type)):
        output = raw_output  ##this is only valid if net was trained with gaussin nll or crps
        score = metrics.crps_gaussian(targets, output, total=avg_on_horizon)
    elif (score_type == 'mse'):
        output = expected_values
        score = metrics.mse(targets, output.unsqueeze(0), total=avg_on_horizon)
    elif (score_type == 'rmse'):
        output = expected_values
        score = metrics.rmse(targets, output.unsqueeze(0), total=avg_on_horizon)
    elif ('mape' in str(score_type)):
        output = expected_values
        score = metrics.mape(targets, output.unsqueeze(0), total=avg_on_horizon)
    else:
        #TODO: catch exception here if performance score is undefined
        score=None
    return score


def set_optimizer(name, model, learning_rate):
    """
    Specify which optimizer to use during training.

    Initialize a torch.optim optimizer for the given model based on the specified name and learning rate.

    Parameters
    ----------
    name : string or None, default = 'adam'
        The name of the torch.optim optimizer to be used. The following
        strings are accepted as arguments: 'adagrad', 'adam', 'adamax', 'adamw', 'rmsprop', or 'sgd'
    model : utils.models.EncoderDecoder
        The model which is to be optimized
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

    if name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.5)
    if name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    if name == "adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)
    if name == "adamax":
        optimizer = torch.optim.Adamax(model.parameters(), lr=learning_rate)
    if name == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    return optimizer


def make_model(
    scalers,
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
    net = models.EncoderDecoder(
        input_size1=enc_size,
        input_size2=dec_size,
        out_size=num_pred,
        dropout_fc=dropout_fc,
        dropout_core=dropout_core,
        rel_linear_hidden_size=rel_linear_hidden_size,
        rel_core_hidden_size=rel_core_hidden_size,
        scalers=scalers,
        core_net=core_net,
        relu_leak=relu_leak,
        core_layers=core_layers,
        criterion=loss,
    )

    return net

def train(
        work_dir,
        train_data_loader,
        validation_data_loader,
        test_data_loader,
        net,
        device,
        args,
        loss_options={},
        best_loss=None,
        best_score=None,
        exploration=False,
        learning_rate=None,
        batch_size=None,
        forecast_horizon=None,
        dropout_fc=None,
        log_df=None,
        optimizer_name="adam",
        max_epochs=None,
        logging_tb=False,
        trial_id="main_run",
        hparams={},
        config={},
        **_,
):
    """
    Train the given model.

    Train the provided model using the given parameters for up to the specified number of epochs, with early stopping.
    Log the training data (optionally using TensorBoard's SummaryWriter)
    Finally, determine the score of the resulting best net.

    Parameters
    ----------
    work_dir : string
        TODO
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
    args : dict, equals default setting, with default loss and default config path, etc.
        dict of customized arguments of model call
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
    net.to(device)
    criterion = args.loss
    # to track the validation loss as the model trains
    valid_losses = []

    early_stopping = EarlyStopping()
    optimizer = set_optimizer(optimizer_name, net, learning_rate)

    inputs1, inputs2, targets = next(iter(train_data_loader))

    #TODO: solve this ARGS.ci issue more elegantly
    #always have logging disabled for ci to avoid failed jobs because of tensorboard
    #if ARGS.ci:
    #    logging_tb = False
    if logging_tb:
        tb = log_tensorboard(
            work_dir=work_dir,
            exploration=exploration,
            trial_id=trial_id
        )
        tb.add_graph(net, [inputs1, inputs2])
    else:
        print("Begin training...")

    t0_start = perf_counter()
    step_counter = 0
    final_epoch_loss = 0.0

    for name, param in net.named_parameters():
        torch.nn.init.uniform_(param.data, -0.08, 0.08)

    for epoch in range(max_epochs):
        epoch_loss = 0.0
        t1_start = perf_counter()
        net.train()
        #train step
        for (inputs1, inputs2, targets) in train_data_loader:
            prediction, _ = net(inputs1, inputs2)
            optimizer.zero_grad()
            loss = criterion(targets, prediction, **(loss_options))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
            optimizer.step()
            step_counter += 1
            epoch_loss += loss.item() / len(train_data_loader)
            final_epoch_loss = epoch_loss
        t1_stop = perf_counter()

        #validation_step
        with torch.no_grad():
            net.eval()
            validation_loss = 0.0
            for (inputs1, inputs2, targets) in validation_data_loader:
                output, _ = net(inputs1, inputs2)
                validation_loss += criterion(
                    targets, output, **(loss_options)
                ).item() / len(validation_data_loader)
                valid_losses.append(validation_loss)

        print(
            "Epoch {}/{}\t train_loss {:.2e}\t val_loss {:.2e}\t elapsed_time {:.2e}".format(
                epoch + 1, max_epochs, epoch_loss, validation_loss, t1_stop - t1_start
            )
        )

        if logging_tb:
            add_tb_element(
                net=net,
                tb=tb,
                epoch_loss=epoch_loss,
                validation_loss=validation_loss,
                t0_start=t0_start,
                t1_stop=t1_stop,
                t1_start=t1_start,
                next_epoch=epoch+1,
                step_counter=step_counter
            )

        early_stopping(validation_loss, net)

        if early_stopping.early_stop:
            print("Stop in earlier epoch. Loading best model state_dict.")
            # load the last checkpoint with the best model
            net.load_state_dict(torch.load(early_stopping.temp_dir + "checkpoint.pt"))
            break

    # Now we are working with the best net and want to save the best score with respect to new test data
    with torch.no_grad():
        net.eval()

        score = performance_test(
            net,
            data_loader=test_data_loader,
            score_type="mis",
            horizon=forecast_horizon,
        ).item()

        if best_score:
            rel_score = calculate_relative_metric(score, best_score)
        else:
            rel_score = 0

        best_loss = early_stopping.val_loss_min

    # ToDo: define logset, move to main. Log on line per script execution, fetch best trial if hp_true, python database or table (MongoDB)
    # Only log once per run so immediately if exploration is false and only on the trial_main_run if it's true
    if (not exploration) or (exploration and not isinstance(trial_id, int)):
        log_df = log_data(
            config,
            args,
            loss_options,
            t1_stop - t0_start,
            final_epoch_loss,
            best_loss,
            score,
            log_df,
        )

    if logging_tb:
        # list of hyper parameters for tensorboard, will be available for sorting in tensorboard/hparams
        # if hyperparam tuning is True, use hparams as defined in tuning.json
        if exploration and isinstance(trial_id, int):
            for items in hparams["hyper_params"]:
                config.update(
                    {
                        items: hparams["hyper_params"][items]
                    }
                )
        # metrics to evaluate the model by
        values = {
            "hparam/hp_total_time": t1_stop - t0_start,
            "hparam/score": score,
            "hparam/relative_score": rel_score,
        }
        end_tensorboard(
            tb=tb,
            params=hparams,
            values=values,
            work_dir=work_dir,
            args=args)

    return net, log_df, best_loss, score, config

def tuning_objective(
        work_dir,
        selected_features,
        scalers,
        hyper_param,
        device,
        log_df,
        config,
        args,
        loss_options,
        **_,
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
    def search_params(trial):
        if config["exploration"]:
            param = {}
            # for more hyperparam, add settings and kwargs in a way compatible with
            # trial object(and suggest methods) of optuna
            for key in hyper_param["settings"].keys():
                print("Creating parameter: ", key)
                func_generator = getattr(
                    trial, hyper_param["settings"][key]["function"]
                )
                param[key] = func_generator(**(hyper_param["settings"][key]["kwargs"]))

        config.update(param)
        hyper_param["hyper_params"] = param
        config["trial_id"] = config["trial_id"] + 1
        train_dl, validation_dl, test_dl = dh.transform(selected_features, device=device, **config)

        model = make_model(
            scalers=scalers,
            enc_size=train_dl.number_features1(),
            dec_size=train_dl.number_features2(),
            num_pred=args.num_pred,
            loss=args.loss,
            **config,
        )
        _, _, val_loss, _, config_update = train(
            work_dir=work_dir,
            train_data_loader=train_dl,
            validation_data_loader=validation_dl,
            test_data_loader=test_dl,
            log_df=log_df,
            net=model,
            device=device,
            args=args,
            loss_options=loss_options,
            logging_tb=True,
            config = config,
            hparams=hyper_param,
            **config,
        )
        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        return val_loss

    return search_params

def init_tuning(
        work_dir,
        model_name,
        config,
        seed=10,
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(),
):
    """
    TODO: Description

    Parameters
    ----------
    work_dir : string
        TODO
    config : dict, default = {}
        dict of model configurations
    model_name : string,
        TODO
    Returns
    -------
    TODO
    Raises
    ------
    TODO
    """

    timeout = None
    hyper_param = read_config(
        model_name=model_name,
        config_path=config["exploration_path"],
        main_path=work_dir,
    )
    if "number_of_tests" in hyper_param.keys():
        n_trials = hyper_param["number_of_tests"]
        config["n_trials"] = n_trials
    if "timeout" in hyper_param.keys():
        timeout = hyper_param["timeout"]

    # Set up the median stopping rule as the pruning condition.
    sampler = optuna.samplers.TPESampler(
        seed=seed
    )  # Make the sampler behave in a deterministic way.
    study = optuna.create_study(
        sampler=sampler,
        direction=direction,
        pruner=pruner,
    )
    return hyper_param, sampler, study

def hparam_tuning(
        work_dir,
        config,
        hparam,
        study,
        selected_features,
        scalers,
        log_df,
        args,
        loss_options,
        device='cpu',
        min_val_loss=float('inf'),
        interactive=False,
        **_,
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
        "Max. number of iteration trials for hyperparameter tuning: ", hparam["number_of_tests"]
    )
    if 'timeout' in hparam.keys():
        if "parallel_jobs" in hparam.keys():
            parallel_jobs = hparam["parallel_jobs"]
            study.optimize(
                tuning_objective(
                    work_dir=work_dir,
                    selected_features=selected_features,
                    scalers=scalers,
                    hyper_param=hparam,
                    device=device,
                    log_df=log_df,
                    config=config,
                    args=args,
                    loss_options=loss_options,
                    **config
                ),
                n_trials=hparam["number_of_tests"],
                n_jobs=hparam["parallel_jobs"],
                timeout=hparam["timeout"],
            )
        else:
            study.optimize(
                tuning_objective(
                    work_dir=work_dir,
                    selected_features=selected_features,
                    scalers=scalers,
                    hyper_param=hparam,
                    device=device,
                    log_df=log_df,
                    config=config,
                    args=args,
                    loss_options=loss_options,
                    **config
                ),
                n_trials=hparam["number_of_tests"],
                timeout=hparam["timeout"],
            )
    else:
        if "parallel_jobs" in hparam.keys():
            parallel_jobs = hparam["parallel_jobs"]
            study.optimize(
                tuning_objective(
                    work_dir=work_dir,
                    selected_features=selected_features,
                    scalers=scalers,
                    hyper_param=hparam,
                    device=device,
                    log_df=log_df,
                    config=config,
                    args=args,
                    loss_options=loss_options,
                    **config
                ),
                n_trials=hparam["number_of_tests"],
                n_jobs=hparam["parallel_jobs"]
            )
        else:
            study.optimize(
                tuning_objective(
                    work_dir=work_dir,
                    selected_features=selected_features,
                    scalers=scalers,
                    hyper_param=hparam,
                    device=device,
                    log_df=log_df,
                    config=config,
                    args=args,
                    loss_options=loss_options,
                    **config
                ),
                n_trials=hparam["number_of_tests"],
            )

    print("Number of finished trials: ", len(study.trials))
    # trials_df = study.trials_dataframe()

    if not os.path.exists(os.path.join(work_dir, config["log_path"])):
        os.mkdir(os.path.join(work_dir, config["log_path"]))

    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    if min_val_loss == None: min_val_loss=float('inf')
    if min_val_loss > study.best_value or interactive:
        print("Training with hyper parameters fetched from config.json")
    else:
        if query_true_false("Overwrite config with new parameters?"):
            config.update(trial.params)
            # TODO: check if one can retrieve a checkpoint/model from best trial in optuna, to omit redundant training
            #  else:
            #  print("Training with hyper parameters fetched from config.json")

    return study, trial, config

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
    min_net=None
    if old_score > achieved_score:
        if config["exploration"]:
            update(
                model_name=model_name,
                achieved_score=achieved_score,
                config=config,
                exploration=config["exploration"],
                study=study,
                interactive=interactive,
                loss=achieved_loss
            )
        min_net = achieved_net
    else:
        if query_true_false("Existing model for this target did not improve in current run. "
                            "Do you still want to overwrite the existing model?"):
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
