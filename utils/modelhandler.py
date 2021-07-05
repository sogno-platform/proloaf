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
import torch
import utils.eval_metrics as metrics
import shutil
import matplotlib
# matplotlib.use('svg') #svg can be used for easy when working on VMs
import tempfile

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
        model : utils.fc_network.EncoderDecoder
            The model being trained
        """
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        temp_dir = tempfile.mktemp()
        torch.save(model.state_dict(), temp_dir+'checkpoint.pt')
        self.val_loss_min = val_loss
        self.temp_dir=temp_dir

def get_prediction(net, data_loader, horizon, number_of_targets):
    """
    Get the predictions for the given model and data

    Parameters
    ----------
    net : utils.fc_network.EncoderDecoder
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

    A negative result indicates that the current score is higher than the best score

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
    return (100 / best_score) * (best_score - curr_score)

def performance_test(net, data_loader, score_type='mis', option=0.05, avg_on_horizon=True, horizon=1, number_of_targets=1):
    """
    Determine the score of the given model using the specified metrics in utils.eval_metrics

    Return a single float value (total score) or a 1-D array (score over horizon) based on
    the value of avg_on_horizon

    Parameters
    ----------
    net : utils.fc_network.EncoderDecoder
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
