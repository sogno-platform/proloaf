'''modelhandler.py holds the functions and classes of the general model architecture'''
import numpy as np
import torch
import plf_util.eval_metrics as metrics
import shutil
import matplotlib
# matplotlib.use('svg') #svg can be used for easy when working on VMs
import tempfile

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
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        temp_dir = tempfile.mktemp()
        torch.save(model.state_dict(), temp_dir+'checkpoint.pt')
        self.val_loss_min = val_loss
        self.temp_dir=temp_dir

def get_prediction(net, data_loader, horizon, number_of_targets):
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

def get_pred_interval(predictions, criterion):
    # Better solution for future: save criterion as class object of 'loss' with 'name' attribute
    #detect criterion
    if ('nll_gaus' in str(criterion)) or ('crps' in str(criterion)):
        #loss_type = 'nll_gaus'
        expected_values = predictions[:,:,0:1] # expected_values:mu
        sigma = torch.sqrt(predictions[:,:,-1:].exp())
        #TODO: make 95% prediction interval changeable
        y_pred_upper = expected_values + 1.96 * sigma
        y_pred_lower = expected_values - 1.96 * sigma
    elif 'pinball' in str(criterion):
        #loss_type = 'pinball'
        y_pred_lower = predictions[:,:,0:1]
        y_pred_upper = predictions[:,:,1:2]
        expected_values = predictions[:,:,-1:]

    elif ('mse' in str(criterion)) or ('rmse' in str(criterion)) or ('mape' in str(criterion)):
        # loss_type = 'mis'
        expected_values = predictions
        y_pred_lower = 0
        y_pred_upper = 1

        #TODO: add all criterion possibilities
    else:
        print('invalid criterion')

    return y_pred_upper, y_pred_lower, expected_values

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

# ToDo: refactor best score, refactor relative_metric
def calculate_relative_metric(curr_score, best_score):
    return (100 / best_score) * curr_score

def performance_test(net, data_loader, score_type='mis', option=0.05, avg_on_horizon=True, horizon=1, number_of_targets=1):
    # check performance
    targets, raw_output = get_prediction(net, data_loader,horizon, number_of_targets) ##should be test data loader
    [y_pred_upper, y_pred_lower, expected_values] = get_pred_interval(raw_output, net.criterion)
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
        score = metrics.mse(targets, output, total=avg_on_horizon)
    elif (score_type == 'rmse'):
        output = expected_values
        score = metrics.rmse(targets, output, total=avg_on_horizon)
    elif ('mape' in str(score_type)):
        output = expected_values
        score = metrics.mape(targets, output, total=avg_on_horizon)
    else:
        #TODO: catch exception here if performance score is undefined
        score=None
    return score
