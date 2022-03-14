"""
evaluates a trained Neural Network on its salient features regarding the time and feature dimension
creates a saliency heatmap
model should be trained beforehand
"""
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append("../")

MAIN_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(MAIN_PATH)

from proloaf import metrics
from proloaf import models
import proloaf.confighandler as ch
import proloaf.datahandler as dh
import proloaf.modelhandler as mh
import proloaf.tensorloader as tl

from random import seed

import torch.nn as nn
import optuna
from time import perf_counter

def create_mean_saliency_map(saliency_maps):
    """
    calculates the mean saliency map over several saliency maps from different time steps and creates a plot.
    The Plot is saved in oracles/interpretation/"model_name"/

    Parameters
    ----------
    saliency_maps: array of saliency map Tensors
    """
    ## create mean over all saliency maps
    print('create mean saliency map over all timesteps')
    saliency_maps_tensor1 = torch.zeros(length, history_horizon, num_features1)
    saliency_maps_tensor2 = torch.zeros(length, forecast_horizon, num_features2)

    for i, timestep in enumerate(saliency_maps):
        saliency_maps_tensor1[i] = timestep[0]

    for i, timestep in enumerate(saliency_maps):
        saliency_maps_tensor2[i] = timestep[1]

    mean_saliency_map = (torch.mean(saliency_maps_tensor1, 0), torch.mean(saliency_maps_tensor2, 0))
    fig, ax = create_saliency_heatmap_plot(mean_saliency_map)
    
    print('Done')

    fig.savefig(interpretation_plot_path + '/mean_heatmap')
    print('mean saliency map saved in '+ interpretation_plot_path)
    fig.show()


def create_reference(dataset: tl.TimeSeriesData, timestep, batch_size):
    """
    Creates the references for the saliency map optimization process.
    Random noise is drawn from a gaussian standard distribution with a mean of zero and the standard deviation
    of the original feature.
    The references are created by adding random noise to each time step of the original feature.
    For each feature a number of references are created, set by the batch_size parameter

    Parameters
    ----------
    dataloader: Tensor
                original data, created by the make_dataloader function
    timestep:   integer
                timestep, at which the references are to be created
    batch_size: integer
                number of references to be created per feature

    Returns
    -------
    features1_references: Tensor
        References for the encoder features
    features2_references: Tensor
        References for the decoder features
    """
    # creates reference for a certain timestep
    history_horizon = dataset.history_horizon
    forecast_horizon = dataset.forecast_horizon
    num_encoder_features = len(dataset.encoder_features)
    num_decoder_features = len(dataset.decoder_features)
    seed(1)  # seed random number generator

    features1_references_np = np.zeros(shape=(batch_size, history_horizon, num_encoder_features))
    features2_references_np = np.zeros(shape=(batch_size, forecast_horizon, num_decoder_features))

    inputs1_np = dataset[timestep][0].cpu().numpy()
    inputs2_np = dataset[timestep][1].cpu().numpy()

    for x in range(num_encoder_features):  # iterate through encoder features
        feature_x = inputs1_np[:, x]
        mu = 0
        sigma = abs(np.std(feature_x))  # 0.3 is chosen arbitrarily # hier np.std nehmen
        for j in range(batch_size):
            noise_feature1 = np.random.default_rng().normal(mu, sigma, history_horizon)  # create white noise series
            features1_references_np[j, :, x] = noise_feature1 + feature_x

    for x in range(num_decoder_features):  # iterate through decoder features
        feature_x = inputs2_np[:, x]
        mu = 0
        sigma = abs(np.std(feature_x))  # 0.3 is chosen arbitrarily
        for j in range(batch_size):
            noise_feature2 = np.random.default_rng().normal(mu, sigma, forecast_horizon)
            features2_references_np[j, :, x] = noise_feature2 + feature_x

    return torch.Tensor(features1_references_np).to(DEVICE), torch.Tensor(features2_references_np).to(DEVICE)


def create_saliency_plot(
             datetime,
             saliency_map,
             plot_path
):
    """
    Creates the saliency map plot, a plot with the prediction targets and predictions, and plots for the original inputs.
    The saliency map plot is split into an encoder(history horizon) part and a decoder(forecast horizon part) on the time axis.
    Features are grouped into 3 groups being:
        1 only Encoder
        2 Encoder and Decoder
        3 only Decoder
    """
    # function assumes 1 target
    # todo: throw error message if more than 1 target variable
    # todo: fix plot feature axes

    history_horizon = dataset.history_horizon
    forecast_horizon = dataset.forecast_horizon
    encoder_features = dataset.encoder_features
    decoder_features = dataset.decoder_features
    # font sizes
    plt.rc('font', size=30)  # default font size
    plt.rc('axes', labelsize=30)  # fontsize of the x and y labels
    plt.rc('axes', titlesize=30)  # fontsize of the title

    fig2, ax2 = plt.subplots(1, figsize=(20, 14))

    # saliency heatmap

    time_axis_length = len(datetime)
    common = list(set(encoder_features) & set(decoder_features))  # features which are both encoder and decoder features
    feature_axis_length = len(encoder_features) + len(decoder_features) - len(common)
    features = pd.array([''] * feature_axis_length)
    saliency_heatmap = np.full(
        (time_axis_length, feature_axis_length),
        fill_value=np.nan
    )  # for features not present in certain areas(nan), use different colour (white)
    counter = -1

    # only encoder features # todo: rewrite this common features part
    i = 0
    while i < len(encoder_features):
        if encoder_features[i] not in common:
            counter += 1
            features[counter] = encoder_features[i]
            saliency_heatmap[0:history_horizon, counter] = saliency_map[0][:, i].cpu().detach().numpy()
        i += 1
    # common features
    i = 0
    j = 0
    while i < len(encoder_features):
        if encoder_features[i] in common:
            counter += 1
            features[counter] = encoder_features[i]
            j = 0
            while j < len(decoder_features):
                if encoder_features[i] == decoder_features[j]:
                    saliency_heatmap[0:history_horizon + forecast_horizon, counter] = torch.cat(
                        (saliency_map[0][:, i], saliency_map[1][:, j]), dim=0
                    ).cpu().detach().numpy()
                    break
                j += 1
        i += 1
    # only decoder features
    i = 0
    while i < len(decoder_features):
        if decoder_features[i] not in common:
            counter += 1
            features[counter] = decoder_features[i]
            saliency_heatmap[history_horizon:, counter] = saliency_map[1][:, i].cpu().detach().numpy()
        i += 1
    saliency_heatmap = np.transpose(saliency_heatmap)  # swap axes

    im = ax2.imshow(saliency_heatmap, cmap='jet',
                    norm=None, aspect='auto', interpolation='nearest', vmin=0, vmax=1, origin='lower')

    # create datetime x-axis
    plot_datetime = pd.array([''] * time_axis_length)  # looks better for plot

    for h in range(datetime.array.size):
        if datetime.array.hour[h] == 0:  # only show full date once per day
            plot_datetime[h] = datetime.array.strftime('%b %d %Y %H:%M')[h]
        else:
            if datetime.array.hour[h] % 12 == 0:  # every 12th hour
                plot_datetime[h] = datetime.array.strftime('%H:%M')[h]

    # show ticks
    ax2.set_xticks(np.arange(len(datetime)))
    ax2.set_xticklabels(plot_datetime)
    feature_ticks = np.arange(len(features))
    ax2.set_yticks(feature_ticks)
    ax2.set_yticklabels(features)

    # rotate tick labels and set alignment
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # set titles and legends
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Features')
    cbar = fig2.colorbar(im)  # add colorbar

    # layout
    fig2.tight_layout()

    # save heatmap
    fig2.savefig(plot_path + '/heatmap')


def save_tensors(
         saliency_map,
         rmse,
         save_path
):
    """
    Saves relevant Tensors locally for future use
    """
    torch.save(saliency_map, save_path + '/saliency_map')
    torch.save(rmse, save_path + '/rmse')


def load_tensors(load_path):
    """
    loads locally saved saliency map and perturbated predictions from the 
    given path 
    """
    temp_saliency_map = torch.load(load_path + '/saliency_map')
    return temp_saliency_map


def get_perturbated_output(
        saliency_map,
        forecasting_model:models.EncoderDecoder,
        timestep,
        encoder_inputs,
        decoder_inputs
):
    # todo write this function
    return 0


def mask_weights_loss(mask_encoder, mask_decoder):  # penalizes high mask parameter values
    """
    penalizes high mask parameter values by calculating the frobenius 
    norm and dividing by the maximal possible norm
    """
    max_norm_encoder = torch.norm(torch.ones(mask_encoder.shape))
    max_norm_decoder = torch.norm(torch.ones(mask_decoder.shape))
    mask_encoder_matrix_norm = torch.norm(mask_encoder) / max_norm_encoder  # frobenius norm
    mask_decoder_matrix_norm = torch.norm(mask_decoder) / max_norm_decoder  # frobenius norm
    loss = mask_encoder_matrix_norm + mask_decoder_matrix_norm
    return loss


def mask_interval_loss(mask_encoder, mask_decoder):
    """
    this function encourages the mask values to stay in interval 0 to 1.
    The loss function is zero when the mask value is between zero and 1, otherwise it takes a value linearly rising with the mask norm
    """

    tresh_plus = nn.Threshold(1, 0)  # thresh for >1
    tresh_zero = nn.Threshold(0, 0)  # thresh for <0

    loss = (
            torch.norm(tresh_plus(mask_encoder))
            + torch.norm(tresh_plus(mask_decoder))
            + torch.norm(tresh_zero(torch.mul(mask_encoder, -1)))
            + torch.norm(tresh_zero(torch.mul(mask_decoder, -1)))
    )

    return loss


def loss_function(
        criterion,
        target_predictions,
        perturbated_predictions,
        mask,
        lambda1=0.1,
        lambda2=1e10,
):
    """
    Calculates the loss function for the mask optimization process.
    A batch here is the number of reference values created for each feature.
    The smallest destroying region loss is calculated by adding up the criterion loss 
    and the weighted mask weight and mask interval losses.
    """

    mask_encoder = mask[0]
    mask_decoder = mask[1]
    batch_size = perturbated_predictions.shape[0]
    target_prediction = target_predictions[0]
    target_copies = torch.zeros(perturbated_predictions.shape).to(DEVICE)

    for n in range(batch_size):  # target prediction is copied for all references in batch
        target_copies[n] = target_prediction

    loss1 = criterion(target_copies, perturbated_predictions)  # prediction loss
    loss2 = lambda1 * mask_weights_loss(mask_encoder, mask_decoder)  # abs value of mask weights
    loss3 = lambda2 * mask_interval_loss(mask_encoder, mask_decoder)

    ssr_loss = loss1 + loss2 + loss3
    # sdr_loss = -loss1 + loss2 + loss3
    return ssr_loss, loss1


# creates saliency map for one timestep:
def objective(trial):
    """
    Ojective function for the optuna optimizer, used for hyperparameter optimization.
    The learning rate and mask initialization value are subject to hyperparameter optimization.
    For each trial the objection function finds the saliency map with gradient descent,
    by updating the saliency map parameters according to the calculated loss.
    Stop counters help to speed up the process, by ending the trial, if the loss doesn't decrease fast enough.
    For each trial the saliency map and other relevant tensors are saved, 
    so the tensors of the best trial can be loaded at the end of the hyperparameter search.
    """
    torch.autograd.set_detect_anomaly(True)

    learning_rate = trial.suggest_loguniform("learning rate", low=1e-5, high=0.01)
    mask_init_value = trial.suggest_uniform('mask initialisation value', 0., 1.)

    inputs1_temp = torch.squeeze(encoder_input, dim=0).to(DEVICE)
    inputs2_temp = torch.squeeze(decoder_input, dim=0).to(DEVICE)

    # todo: rework saliency map as class
    saliency_map = (torch.full((CONFIG["history_horizon"], len(dataset.encoder_features)), fill_value=mask_init_value, device=DEVICE,
                               requires_grad=True),
                    torch.full((CONFIG["forecast_horizon"], len(dataset.decoder_features)), fill_value=mask_init_value, device=DEVICE,
                               requires_grad=True))

    optimizer = torch.optim.Adam(saliency_map, lr=learning_rate)

    stop_counter = 0

    # calculate mask
    for epoch in range(MAX_EPOCHS):  # mask 'training' epochs

        # create inverse masks
        inverse_saliency_map1 = torch.sub(torch.ones(inputs1_temp.shape, device=DEVICE),
                                          saliency_map[0]).to(DEVICE)  # elementwise 1-m
        inverse_saliency_map2 = torch.sub(torch.ones(inputs2_temp.shape, device=DEVICE),
                                          saliency_map[1]).to(DEVICE)  # elementwise 1-m
        input_summand1 = torch.mul(inputs1_temp, saliency_map[0]).to(DEVICE)  # element wise multiplication
        input_summand2 = torch.mul(inputs2_temp, saliency_map[1]).to(DEVICE)  # element wise multiplication

        # create perturbated series through mask
        # todo: write function for getting perturbated inputs from a saliency map
        reference_summand1 = torch.mul(features1_references, inverse_saliency_map1).to(DEVICE)
        perturbated_input1 = torch.add(input_summand1, reference_summand1).to(DEVICE)
        reference_summand2 = torch.mul(features2_references, inverse_saliency_map2).to(DEVICE)
        perturbated_input2 = torch.add(input_summand2, reference_summand2).to(DEVICE)

        # get prediction
        forecasting_model.model.train()
        perturbated_prediction = forecasting_model.predict(
            perturbated_input1,
            perturbated_input2
        ).to(DEVICE)
        loss, rmse = loss_function(
            criterion,
            prediction,
            perturbated_prediction,
            saliency_map
        )

        optimizer.zero_grad()  # set all gradients zero

        # todo make stop counter function
        if (epoch >= 1000) and (epoch < 3000):
            if (loss > 0.2) and (loss < 1):  # loss <1 to prevent stopping because mask out of [0,1] boundary
                stop_counter += 1  # stop counter to prevent stopping due to temporary loss jumps
                if stop_counter == 10:
                    print('stopping...')
                    break
            else:
                stop_counter = 0

        elif (epoch >= 3000) and (epoch < 5000):
            if (loss > 0.1) and (loss < 1):  # loss <1 to prevent stopping because mask out of [0,1] boundary
                stop_counter += 1  # stop counter to prevent stopping due to temporary loss jumps
                if stop_counter == 10:
                    print('stopping...')
                    break
            else:
                stop_counter = 0

        elif (epoch >= 5000) and (epoch < 10000):
            if (loss > 0.05) and (loss < 1):  # loss <1 to prevent stopping because mask out of [0,1] boundary
                stop_counter += 1  # stop counter to prevent stopping due to temporary loss jumps
                if stop_counter == 10:
                    print('stopping...')
                    break
            else:
                stop_counter = 0

        loss.backward()  # backpropagate mean loss
        optimizer.step()  # update mask parameters

        if epoch % 1000 == 0:  # print every 100 epochs
            print('epoch ', epoch, '/', MAX_EPOCHS, '...    loss:', loss.item())

    # trial_id = trial.number
    trial.set_user_attr("saliency map", saliency_map)
    trial.set_user_attr("rmse", rmse.detach())
    trial.set_user_attr("perturbated_prediction", perturbated_prediction.detach())

    return loss


if __name__ == "__main__":
    # todo: rework interpreter as class structure
    # todo: put all classes and functions in /src
    # todo write parser with cli.py wich reads station name
    # todo write an interpreter config file for the settings.
    # todo Throw message if interpreter config doesn't exist and create dummy config automatically

    MAIN_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    INTERPRETATION_PATH = './oracles/interpretation/'
    sys.path.append(MAIN_PATH)
    MAX_BATCH_SIZE = 10
    MAX_EPOCHS = 1 # 10000
    N_TRIALS = 1 # 50  # hyperparameter tuning trials
    MODEL_NAME = 'test_opsd'  # path relative to targets folder
    SEP = ';'  # seperation for csv data
    criterion = metrics.Rmse()  # loss function criterion
    # time steps to interpret (beginning of history horizon) to calculate: index in csv table -2 -history horizon
    timesteps = [0, 100]

    print('model: ', MODEL_NAME)

    # Data preperation

    # import config and extract relevant config variables
    CONFIG_PATH = './targets/' + MODEL_NAME + '/config.json'
    config_file = os.path.join(MAIN_PATH, CONFIG_PATH)
    CONFIG = ch.read_config(
        config_path=config_file,
        main_path=MAIN_PATH
    )
    TUNING_CONFIG = ch.read_config(
                config_path=CONFIG["exploration_path"],
                main_path=MAIN_PATH,
            )

    path = os.path.join(MAIN_PATH, INTERPRETATION_PATH)
    if not os.path.exists(path):
        os.mkdir(path)
    model_name = CONFIG["model_name"]
    model_interpretation_path = os.path.join(path, model_name + '/')
    if not os.path.exists(model_interpretation_path):
        os.mkdir(model_interpretation_path)

    # todo has this got to be in the main function? maybe put at beginning of document
    cuda_id = CONFIG["cuda_id"]
    if torch.cuda.is_available():
        DEVICE = 'cuda'
        if cuda_id is not None:
            torch.cuda.set_device(cuda_id)
        print('Device: ', DEVICE)
        print('Current CUDA ID: ', torch.cuda.current_device())

    else:
        DEVICE = 'cpu'
        print(DEVICE)

    # import data
    print('reading csv...')
    df = pd.read_csv(os.path.join(MAIN_PATH, CONFIG["data_path"]), sep=SEP)
    time_column = df.loc[:, "Time"]
    print('done')

    # get scaler
    scaler = dh.MultiScaler(CONFIG["feature_groups"])

    dataset = tl.TimeSeriesData(  # todo: preperation steps needed?
        df,
        preparation_steps=[
            dh.set_to_hours,
            dh.fill_if_missing,
            dh.add_cyclical_features,
            dh.add_onehot_features,
            scaler.fit_transform,
            dh.check_continuity,
        ],
        device=DEVICE,
        **CONFIG
    )

    dataloader = dataset.make_data_loader(shuffle=False).to(DEVICE)

    print('timesteps:', len(dataset), '\n',
          'number of encoder features:', len(dataset.encoder_features), '\n',
          'number of decoder features:', len(dataset.decoder_features), '\n',
          'number of targets:', len(dataset.target_id), '\n',
          'history_horizon:', CONFIG["history_horizon"], '\n',
          'forecast_horizon:', CONFIG["forecast_horizon"])

    # load the trained forecasting NN model
    print('loading forecasting model...')
    INMODEL_PATH = os.path.join(
        MAIN_PATH,
        CONFIG.get("output_path", ""),
        f"{CONFIG['model_name']}.pkl",
    )

    modelhandler = mh.ModelHandler(
        work_dir=MAIN_PATH,
        config=CONFIG,
        tuning_config=TUNING_CONFIG,
        device=DEVICE,
    )
    forecasting_model = modelhandler.load_model(INMODEL_PATH).to(DEVICE)
    print('Done.')

    t0_start = perf_counter()

    results_df = pd.DataFrame(columns=['RMSE PERTURBATED',
                                       'RMSE ORIGINAL',
                                       'RMSE DIFFERENCE PERTURBATED ORIGINAL'],
                              index=timesteps)
    for timestep in timesteps:

        # create path for timestep
        timestep_plot_path = model_interpretation_path + str(timestep)
        if not os.path.exists(timestep_plot_path):
            os.mkdir(timestep_plot_path)

        t1_start = perf_counter()
        print('\n\ntimestep: ', timestep)
        datetime = pd.to_datetime(
            time_column.iloc[timestep:timestep + CONFIG["history_horizon"] + CONFIG["forecast_horizon"]])
        # get original inputs and predictions
        encoder_input = torch.unsqueeze(dataset[timestep][0], 0).to(DEVICE)
        decoder_input = torch.unsqueeze(dataset[timestep][1], 0).to(DEVICE)
        target = torch.unsqueeze(dataset[timestep][2], 0).to(DEVICE)
        with torch.no_grad():
            prediction = forecasting_model.predict(encoder_input, decoder_input).to(DEVICE)

        # obtain reference input data
        features1_references, features2_references = create_reference(dataset, timestep, MAX_BATCH_SIZE)

        # create saliency map
        print('create saliency maps...')
        study = optuna.create_study()
        study.optimize(
            objective,
            n_trials=N_TRIALS)
        print('Done')

        # load best saliency map
        best_trial_id = study.best_trial.number
        # saliency_map, perturbated_prediction = load_interpretation_tensors(tensor_save_path, best_trial_id)
        saliency_map = study.best_trial.user_attrs['saliency map']
        rmse_p = study.best_trial.user_attrs['rmse'] # rmse of perturbated prediction and target
        perturbated_prediction = study.best_trial.user_attrs['perturbated_prediction']

        # todo: get perturbed predictions
        # save plot for best saliency map
        create_saliency_plot(
                         datetime,
                         saliency_map,
                         timestep_plot_path
        )
        t1_stop = perf_counter()
        print("Elapsed time: ", t1_stop - t1_start)

        # rmse_original = criterion(target, prediction).cpu().detach().numpy()
        # rmse_diff = rmse_perturbated - rmse_original  # difference in rmse scores between perturbated and original prediction
        # data = {
        #     'RMSE PERTURBATED': rmse_perturbated,
        #     'RMSE ORIGINAL': rmse_original,
        #     'RMSE DIFFERENCE PERTURBATED ORIGINAL': rmse_diff}
        # results_df.loc[timestep] = data
        # save_path = model_interpretation_path
        # results_df.to_csv(save_path + 'rmse.csv', sep=';', index=True)

        save_tensors(saliency_map, rmse_p, timestep_plot_path)

    t0_stop = perf_counter()
    print("Total elapsed time: ", t0_stop - t0_start)
