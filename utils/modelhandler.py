'''modelhandler.py holds the functions and classes of the general model architecture'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import torch
import fc_util.fc_network as fc_net
import fc_util.eval_metrics as metrics
import shutil
import matplotlib
matplotlib.use('svg')
import matplotlib.pyplot as plt

import sklearn as sk
from sklearn import preprocessing

class EarlyStopping:
# implement early stopping
# Reference: https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

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
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss

def get_prediction(net, data_loader):
    for i, (inputs1, inputs2, targets) in enumerate(data_loader):
        output, _ = net(inputs1, inputs2)
    return targets, output

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def performance_test(net, data_loader, score_type='mis', option=0.05, avg_on_horizon=True):
    # check performance
    targets, output = get_prediction(net, data_loader)
    if (score_type=='mis'):
        score = metrics.mis(targets, output, alpha=option, total=avg_on_horizon)
    elif (score_type == 'nll_gauss'):
        score = metrics.nll_gauss(targets, output, total=avg_on_horizon)
    elif (score_type == 'pinball'):
        score = metrics.pinball_loss(targets, output, quantiles=option, total=avg_on_horizon)
    elif (score_type == 'crps'):
        score = metrics.crps_gaussian(targets, output, total=avg_on_horizon)
    elif (score_type == 'mse'):
        score = metrics.mse(targets, output, total=avg_on_horizon)
    elif (score_type == 'rmse'):
        score = metrics.rmse(targets, output, total=avg_on_horizon)
    elif (score_type == 'mape'):
        score = metrics.mape(targets, output, total=avg_on_horizon)
    else:
        #TODO: catch exception here if performance score is undefined
        score=None
    return score