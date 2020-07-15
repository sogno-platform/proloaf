#TODO: find workaround for PICP numpy issue
#import sys
#sys.path.append('../')
import numpy as np
import pandas as pd
import torch
import warnings
import sklearn as sk
import matplotlib.pyplot as plt
import sys
import json
import os

MAIN_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(MAIN_PATH)

import plf_util.datatuner as dt
import plf_util.tensorloader as dl
import plf_util.eval_metrics as metrics
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from plf_util.config_util import read_config, parse_basic

warnings.filterwarnings('ignore')

def shape_model_input(df, columns_p, horizon_p, horizon_f):
    # shape input data that is measured in the Past and can be fetched from UDW/LDW
    x_p = dt.extract(df[columns_p].iloc[:-horizon_f, :], horizon_p)
    # shape input data that is known for the Future, here take perfect hourly temp-forecast
    x_f = dt.extract(df.drop(columns_p, axis=1).iloc[horizon_p:, :], horizon_f)
    # shape y
    y = dt.extract(df[[target_id]].iloc[horizon_p:, :], horizon_f)
    return x_p, x_f, y

def results_table(models, mse, rmse, sharpness, coverage, mis):
    data = {
        'MSE': mse,
        'RMSE': rmse,
        'Mean sharpness': sharpness,
        'Mean PICP': coverage,
        'Mean IS': mis}
    results_df = pd.DataFrame(data, index=models)
    return results_df

# def gaussian_nll_loss2(target, expected_value, variance):
#     """y, y_pred, var_pred must have the same shape."""
#     assert target.shape == expected_value.shape
#     assert target.shape == variance.shape

#     squared_errors = (target - expected_value) ** 2
#     loss = squared_errors / (2 * variance) + variance.sqrt().log()
#     return loss


def evaluate_hours(target, pred ,y_pred_upper, y_pred_lower, hour, OUTPATH, limit, actual_hours=None):
    fig, ax = plt.subplots(1)
    ax.plot(target, '.-k', label="Truth")  # true values
    ax.plot(pred, 'b', label='Predicted')
    #insert actual time
    if(actual_hours.dt.hour.any()):
        ax.set_title(actual_hours.iloc[0].strftime("%a, %Y-%m-%d"), fontsize=20)
    else:
        ax.set_title('Forecast along horizon', fontsize=22)
    ax.fill_between(np.arange(pred.shape[0]), pred.squeeze(), y_pred_upper.squeeze(), alpha=0.1, color='g')
    ax.fill_between(np.arange(pred.shape[0]), y_pred_lower.squeeze(), pred.squeeze(), alpha=0.1, color='g')

    ax.set_title('Forecast along horizon', fontsize=22)
    ax.set_xlabel("Hour", fontsize=18)
    ax.set_ylabel("Scaled Residual Load (-1,1)", fontsize=20)
    ax.legend(fontsize=20)
    ax.grid(b=True, linestyle='-')
    if(limit):
        plt.axhline(linewidth=2, color='r', y=limit)
    ax.grid()
    positions = range(0, 40, 2)
    labels = actual_hours.dt.hour.to_numpy()
    new_labels = labels[positions]
    plt.xticks(positions, new_labels)
    ax.set_xlabel("Hour of Day", fontsize=20)
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.savefig(OUTPATH + 'eval_hour{}'.format(hour))

def plot_metrics(rmse_horizon, sharpness_horizon, coverage_horizon, mis_horizon,OUTPATH, title):
    with plt.style.context('seaborn'):
        fig = plt.figure(figsize=(16, 12))
        st = fig.suptitle(title, fontsize=25)
        plt.rc('xtick', labelsize=15)
        plt.rc('ytick', labelsize=15)

        ax_rmse = plt.subplot(2, 2, 1)
        ax_sharpness = plt.subplot(2, 2, 3)
        ax_PICP = plt.subplot(2, 2, 2)
        ax_MSIS = plt.subplot(2, 2, 4)

        ax_rmse.plot(rmse_horizon, label='rmse')
        ax_rmse.set_title('RMSE along horizon', fontsize=22)
        ax_rmse.set_xlabel("Hour", fontsize=18)
        ax_rmse.set_ylabel("RMSE", fontsize=20)
        ax_rmse.legend(fontsize=20)
        ax_rmse.grid(b=True, linestyle='-')

        ax_sharpness.plot(sharpness_horizon, label='sharpness')
        ax_sharpness.set_title('sharpness along horizon', fontsize=22)
        ax_sharpness.set_xlabel("Hour", fontsize=18)
        ax_sharpness.set_ylabel("sharpness", fontsize=20)
        ax_sharpness.legend(fontsize=20)
        ax_sharpness.grid(b=True, linestyle='-')

        ax_PICP.plot(coverage_horizon, label='coverage')
        ax_PICP.set_title('coverage along horizon', fontsize=22)
        ax_PICP.set_xlabel("Hour", fontsize=18)
        ax_PICP.set_ylabel("coverage in %", fontsize=20)
        ax_PICP.legend(fontsize=20)
        ax_PICP.grid(b=True, linestyle='-')

        ax_MSIS.plot(mis_horizon, label='MIS')
        ax_MSIS.set_title('Mean Interval score', fontsize=22)
        ax_MSIS.set_xlabel("Hour", fontsize=18)
        ax_MSIS.set_ylabel("MIS", fontsize=20)
        ax_MSIS.legend(fontsize=20)
        ax_MSIS.grid(b=True, linestyle='-')

        st.set_y(1.08)
        fig.subplots_adjust(top=0.95)
        plt.tight_layout()
        plt.savefig(OUTPATH + 'metrics_plot')
        #plt.show()


if __name__ == '__main__':
    ARGS = parse_basic()
    PAR = read_config(model_name=ARGS.station, config_path=ARGS.config, main_path=MAIN_PATH)


    # DEFINES
    torch.manual_seed(1)
    INFILE = os.path.join(MAIN_PATH, PAR['data_path'])  # input factsheet

    # after training
    INMODEL = os.path.join(MAIN_PATH, PAR['output_path'], PAR['model_name'])
    OUTDIR = os.path.join(MAIN_PATH, PAR['evaluation_path'])


    target_id = PAR['target_id']
    DEVICE = 'cpu'
    SPLIT_RATIO = PAR['validation_split']

    HISTORY_HORIZON = PAR['history_horizon']
    FORECAST_HORIZON = PAR['forecast_horizon']
    feature_groups = PAR['feature_groups']

    if not os.path.exists(OUTDIR):
        os.makedirs(OUTDIR)

    # Read load data
    df = pd.read_csv(INFILE, sep=';')
    df = dt.fill_if_missing(df)

    if ('target_list' in PAR):
        if(PAR['target_list'] != None):
            df[target_id] = df[PAR['target_list']].sum(axis=1)


    # reload trained NN
    net = torch.load(INMODEL, map_location=torch.device(DEVICE))  # mapping to CPU

    df_new, _ = dt.scale_all(df, **PAR)

    target_index = df_new.columns.get_loc(target_id)

    split_index = int(len(df_new.index) * SPLIT_RATIO)
    test_df = df_new.iloc[split_index:]

    test_data_loader = dl.make_dataloader(test_df, shuffle=False, **PAR).to(DEVICE)

    # check performance
    horizon = test_data_loader.dataset.targets.shape[1]
    number_of_targets = test_data_loader.dataset.targets.shape[2]

    record_expected_values = torch.zeros((len(test_data_loader), horizon, number_of_targets), device = DEVICE)
    record_variances = torch.zeros((len(test_data_loader), horizon, number_of_targets), device = DEVICE)
    record_targets = torch.zeros((len(test_data_loader), horizon, number_of_targets), device = DEVICE)

    net.eval()
    with torch.no_grad():
        for i, (inputs1, inputs2, targets) in enumerate(test_data_loader):
            (expected_values, log_variances), _ = net(inputs1, inputs2)
            record_expected_values = expected_values
            record_variances = log_variances.exp()
            record_targets = targets

    for group in PAR['feature_groups']:
        if group['features'] is not None and PAR['target_id'] in group['features']:
            scaler_name = group['name']
            scaler = net.scalers[scaler_name]
            scale = scaler.scale_.take(target_index)
            break # assuming target column can only be scaled once

    if type(scaler) == RobustScaler:
        # customized inversion robust_scaler
        if scaler.center_.any() == False:  # no center shift applied
            center = 0
        else:
            center = scaler.center_.take(target_index)
        scale = scaler.scale_.take(target_index)  # The (scaled) interquartile range for each feature in the training set
    elif type(scaler) == StandardScaler:
        if scaler.mean_.take(target_index) == None:
            center = 0
        else:
            center = scaler.mean_.take(target_index)
        scale = scaler.scale_.take(target_index)
    elif type(scaler) == MinMaxScaler:
        range_min = scaler.feature_range[0]
        range_max = scaler.feature_range[1]
        data_max = scaler.data_max_.take(target_index)
        data_min = scaler.data_min_.take(target_index)
        scale = (data_max - data_min) / (range_max - range_min)
        center = data_min - range_min * scale
    #TODO: use get_prediction from mondelhandler instead
    y_pred_upper = record_expected_values + 1.96 * record_variances.sqrt()
    y_pred_lower = record_expected_values - 1.96 * record_variances.sqrt()

    #rescale values
    #record_expected_values_rescaled = (record_expected_values * scale) + center
    #record_targets_rescaled = (record_targets * scale) + center
    #y_pred_upper_rescaled = record_expected_values_rescaled + 1.96 * record_variances.sqrt() * scale
    #y_pred_lower_rescaled = record_expected_values_rescaled - 1.96 * record_variances.sqrt() * scale

    #calculate the metrics

    mse_horizon = metrics.mse(record_targets, [record_expected_values],total=False)
    rmse_horizon = metrics.rmse(record_targets, [record_expected_values],total=False)
    sharpness_horizon = metrics.sharpness(None, [y_pred_upper, y_pred_lower],total=False)
    coverage_horizon = metrics.picp(record_targets, [y_pred_upper,
                                                    y_pred_lower],total=False)
    mis_horizon = metrics.mis(record_targets, [y_pred_upper, y_pred_lower], alpha=0.05,total=False)

    #collect metrics by disregarding the development over the horizon
    mse = metrics.mse(record_targets, [record_expected_values])
    rmse = metrics.rmse(record_targets, [record_expected_values])
    sharpness = metrics.sharpness(None, [y_pred_upper, y_pred_lower])
    coverage = metrics.picp(record_targets, [y_pred_upper,
                                                    y_pred_lower])
    mis = metrics.mis(record_targets, [y_pred_upper, y_pred_lower], alpha=0.05)

    #plot metrics
    plot_metrics(rmse_horizon, sharpness_horizon, coverage_horizon, mis_horizon, OUTDIR, 'metrics-evaluation')

    #plot forecast for sample days
    testhours = [0, 12, 24, 48, 100, 112]
    actual_time = pd.to_datetime(df.loc[split_index:, 'Time'])
    for i in testhours:
        hours = actual_time.iloc[i:i + FORECAST_HORIZON]
        evaluate_hours(record_targets[i].cpu().numpy(), record_expected_values[i].cpu().numpy(), y_pred_upper[i].cpu().numpy(), y_pred_lower[i].cpu().numpy(), i, OUTDIR, PAR['cap_limit'], hours)

    target_stations = [PAR['model_name']]
    print(results_table(target_stations, mse.cpu().numpy(), rmse.cpu().numpy(), sharpness.cpu().numpy(), coverage.cpu().numpy(), mis.cpu().numpy()))

    print('Done!!')
