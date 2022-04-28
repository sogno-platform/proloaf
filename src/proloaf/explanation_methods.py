import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from random import seed
import torch.nn as nn
import optuna
from time import perf_counter

sys.path.append("../")
MAIN_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(MAIN_PATH)

from proloaf import metrics
from proloaf import models
from proloaf.event_logging import create_event_logger
import proloaf.confighandler as ch
import proloaf.datahandler as dh
import proloaf.modelhandler as mh
import proloaf.tensorloader as tl

logger = create_event_logger(__name__)
optuna.logging.enable_propagation()


class SaliencyMapUtil:

    def __init__(
            self,
            target: str,
            sep=';'
    ):

        # read explanation config
        logger.info('reading explanation.json config')
        ex_config_path = os.path.join('targets', target, 'explanation.json')
        self._explanation_config = ch.read_config(
            config_path=ex_config_path,
            main_path=MAIN_PATH
        )

        # read forecasting model config
        logger.info('reading model config...')
        config_path = './targets/' + target + '/config.json'
        model_config = ch.read_config(
            config_path=os.path.join(MAIN_PATH, config_path),
            main_path=MAIN_PATH
        )
        tuning_config_path = './targets/' + target + '/tuning.json'
        model_tuning_config = ch.read_config(
            config_path=os.path.join(MAIN_PATH, tuning_config_path),
            main_path=MAIN_PATH,
        )

        # import data
        logger.info('importing data...')
        df = pd.read_csv(os.path.join(MAIN_PATH, model_config["data_path"]), sep=sep)

        # setting device
        self._device = self.set_device(self._explanation_config["cuda_id"])
        logger.debug('Device: {}'.format(self._device))

        # get scaler
        scaler = dh.MultiScaler(model_config["feature_groups"])

        # create timeseries dataset
        logger.info('preparing the dataset...')
        self._dataset = tl.TimeSeriesData(
            df,
            preparation_steps=[
                dh.set_to_hours,
                dh.fill_if_missing,
                dh.add_cyclical_features,
                dh.add_onehot_features,
                scaler.fit_transform,
                dh.check_continuity,
            ],
            device=self._device,
            **model_config
        )
        self._index_copy = self._dataset.data.index  # gets replaced by Time column after to_tensor()
        self._dataset.to_tensor()  # prepare dataset

        # loading the forecasting model
        try:
            logger.info('loading the forecasting model')
            model_wrap_path = os.path.join(MAIN_PATH, model_config["output_path"], model_config["model_name"] + '.pkl')
            self._model_wrap = mh.ModelHandler.load_model(path=model_wrap_path, locate=self._device)
        except:
            logger.error("An error has occurred while trying to load the forecasting model."
                         "The model has to be trained and saved as a loadable file.")

        # initialize saliency map
        logger.debug('initializing saliency map...')

        # todo self._saliency_map._encoder_map = ...
        self._saliency_map = (
            torch.zeros(self.history_horizon, self.num_encoder_features, requires_grad=True, device=self._device),
            torch.zeros(self.forecast_horizon, self.num_decoder_features, requires_grad=True, device=self._device)
        )

        self.datetime = pd.to_datetime(self._explanation_config["date"], format="%d.%m.%Y %H:%M:%S")

        # set interpretation path
        self._path = os.path.join(
            MAIN_PATH,
            self._explanation_config["rel_interpretation_path"],
            target + '/')
        if not os.path.exists(self._path):
            os.mkdir(self._path)

        self._optimization_done = False

    @staticmethod
    def set_device(cuda_id):
        logger.info('setting computation device...')
        if torch.cuda.is_available():
            if cuda_id is not None:
                torch.cuda.set_device(cuda_id)
            logger.debug('Current CUDA ID: {}'.format(torch.cuda.current_device()))
            device = 'cuda'
            return device

        else:
            device = 'cpu'
            return device

    @property
    def datetime(self):
        return self._datetime

    @datetime.setter
    def datetime(self, datetime):
        if not isinstance(datetime, pd.Timestamp):
            raise TypeError("Only pandas Timestamps are accepted as datetime variables")
        self._datetime = datetime
        logger.info("Date and Time set to: {}".format(datetime))

        self._update_time_step()

    def _update_time_step(self):
        try:
            time_step = self._index_copy[
                pd.to_datetime(self._dataset.data.index) == self.datetime
                ]
            time_step = time_step.values
            assert len(time_step) == 1
            time_step = int(time_step[0])
            assert isinstance(time_step, int)
            logger.debug('Timestep for saliency map with the date {!s} is {!s}'.format(self.datetime, time_step))
            self._time_step = time_step
        except:
            logger.error("An error has occurred while trying to read the datetime."
                         " Please use pandas datetime and correct format.")

    @property
    def time_step(self):
        if not hasattr(self, '_time_step'):
           raise RuntimeError("Time step has not been set yet. Please set a date first")
        return self._time_step

    @time_step.setter
    def time_step(self, time_step):
        raise RuntimeError("The time step should not be set directly."
                           "It is automatically set and updated, when setting the datetime property")

    @property
    def history_horizon(self):
        return self._dataset.history_horizon

    @property
    def forecast_horizon(self):
        return self._dataset.forecast_horizon

    @property
    def num_encoder_features(self):
        return len(self._dataset.encoder_features)

    @property
    def num_decoder_features(self):
        return len(self._dataset.decoder_features)

    @property
    def encoder_input(self):
        return torch.unsqueeze(self._dataset[self.time_step][0], 0).to(self._device)

    @property
    def decoder_input(self):
        return torch.unsqueeze(self._dataset[self.time_step][1], 0).to(self._device)

    @property
    def encoder_references(self):
        return self._encoder_references

    @property
    def decoder_references(self):
        return self._decoder_references

    def _create_references(self):
        """
            Creates the references for the saliency map optimization process.
            Random noise is drawn from a gaussian standard distribution with a mean of zero and the standard deviation
            of the original feature.
            The references are created by adding random noise to each time step of the original feature.
            For each feature a number of references are created, set by the batch_size parameter

            """

        # creates reference for a certain timestep
        seed(1)  # seed random number generator

        features1_references_np = np.zeros(
            shape=(self._explanation_config["ref_batch_size"],
                   self.history_horizon,
                   self.num_encoder_features)
        )
        features2_references_np = np.zeros(
            shape=(self._explanation_config["ref_batch_size"],
                   self.forecast_horizon,
                   self.num_decoder_features)
        )

        inputs1_np = self._dataset[self.time_step][0].cpu().numpy()
        inputs2_np = self._dataset[self.time_step][1].cpu().numpy()

        for x in range(self.num_encoder_features):  # iterate through encoder features
            feature_x = inputs1_np[:, x]
            mu = 0
            sigma = abs(np.std(feature_x))  # 0.3 is chosen arbitrarily # hier np.std nehmen
            for j in range(self._explanation_config["ref_batch_size"]):
                # create white noise series
                noise_feature1 = np.random.default_rng().normal(mu, sigma, self.history_horizon)
                features1_references_np[j, :, x] = noise_feature1 + feature_x

        for x in range(self.num_decoder_features):  # iterate through decoder features
            feature_x = inputs2_np[:, x]
            mu = 0
            sigma = abs(np.std(feature_x))  # 0.3 is chosen arbitrarily
            for j in range(self._explanation_config["ref_batch_size"]):
                noise_feature2 = np.random.default_rng().normal(mu, sigma, self.forecast_horizon)
                features2_references_np[j, :, x] = noise_feature2 + feature_x

        self._encoder_references = torch.Tensor(features1_references_np).to(self._device)
        self._decoder_references = torch.Tensor(features2_references_np).to(self._device)

    def _fill_with(self, fill_value):  # todo write documention string

        temp_saliency_map = (
            torch.full(
                (self.history_horizon, self.num_encoder_features),
                fill_value=fill_value,
                device=self._device,
                requires_grad=True
            ),
            torch.full(
                (self.forecast_horizon, self.num_decoder_features),
                fill_value=fill_value,
                device=self._device,
                requires_grad=True)
        )
        return temp_saliency_map

    def _get_perturbated_input(self, saliency_map):  # todo write reference

        inverse_saliency_map1 = torch.sub(
            torch.ones(saliency_map[0].shape, device=self._device),
            saliency_map[0]
        ).to(self._device)  # elementwise 1-m

        inverse_saliency_map2 = torch.sub(
            torch.ones(saliency_map[1].shape, device=self._device),
            saliency_map[1]
        ).to(self._device)  # elementwise 1-m

        input_summand1 = torch.mul(
            torch.squeeze(self.encoder_input, dim=0).to(self._device),
            saliency_map[0]
        ).to(self._device)  # element wise multiplication

        input_summand2 = torch.mul(
            torch.squeeze(self.decoder_input, dim=0).to(self._device),
            saliency_map[1]
        ).to(self._device)  # element wise multiplication

        reference_summand1 = torch.mul(self.encoder_references, inverse_saliency_map1).to(self._device)
        reference_summand2 = torch.mul(self.decoder_references, inverse_saliency_map2).to(self._device)

        perturbated_input1 = torch.add(input_summand1, reference_summand1).to(self._device)
        perturbated_input2 = torch.add(input_summand2, reference_summand2).to(self._device)

        return perturbated_input1, perturbated_input2

    def _get_perturbated_prediction(self, saliency_map):  # todo write reference

        # perturbate input
        perturbated_input1, perturbated_input2 = self._get_perturbated_input(saliency_map)

        # get prediction of perturbated input
        self._model_wrap.model.train()
        perturbated_prediction = self._model_wrap.predict(
            perturbated_input1,
            perturbated_input2
        ).to(self._device)

        return perturbated_prediction

    def _loss_function(
            self,
            target_predictions,
            perturbated_predictions,
            mask,
            criterion=metrics.Rmse(),
            lambda1=0.1,
            lambda2=1e10,
    ):
        """
        Calculates the loss function for the mask optimization process
            which is calculated by the smallest supporting region principle.
        A batch is the number of reference values created for each feature.
        The smallest supporting region loss is calculated by adding up the criterion loss
            the weighted mask weight and mask interval losses.
        """

        mask_encoder = mask[0]
        mask_decoder = mask[1]

        def mask_interval_loss():
            """
            this function encourages the mask values to stay in interval 0 to 1.
            The loss function is zero when the mask value is between zero and 1,
            otherwise it takes a value linearly rising with the mask norm
            """
            tresh_plus = nn.Threshold(1, 0)  # thresh for >1
            tresh_zero = nn.Threshold(0, 0)  # thresh for <0

            mi_loss = (
                    torch.norm(tresh_plus(mask_encoder))
                    + torch.norm(tresh_plus(mask_decoder))
                    + torch.norm(tresh_zero(torch.mul(mask_encoder, -1)))
                    + torch.norm(tresh_zero(torch.mul(mask_decoder, -1)))
            )

            return mi_loss

        def mask_weights_loss():  # penalizes high mask parameter values
            """
            penalizes high mask parameter values by calculating the frobenius
            norm and dividing by the maximal possible norm
            """
            max_norm_encoder = torch.norm(torch.ones(mask_encoder.shape))
            max_norm_decoder = torch.norm(torch.ones(mask_decoder.shape))
            mask_encoder_matrix_norm = torch.norm(mask_encoder) / max_norm_encoder  # frobenius norm
            mask_decoder_matrix_norm = torch.norm(mask_decoder) / max_norm_decoder  # frobenius norm

            mw_loss = mask_encoder_matrix_norm + mask_decoder_matrix_norm
            return mw_loss

        batch_size = perturbated_predictions.shape[0]
        target_prediction = target_predictions[0]
        target_copies = torch.zeros(perturbated_predictions.shape).to(self._device)

        for n in range(batch_size):  # target prediction is copied for all references in batch
            target_copies[n] = target_prediction

        loss1 = criterion(target_copies, perturbated_predictions)  # prediction loss
        loss2 = lambda1 * mask_weights_loss()  # abs value of mask weights
        loss3 = lambda2 * mask_interval_loss()

        ssr_loss = loss1 + loss2 + loss3
        # sdr_loss = -loss1 + loss2 + loss3
        return ssr_loss, loss1

    def optimize(self):

        # todo rewrite function to automatically compute list of timestamps
        #  without having to reload the model every time
        # start counter
        logger.info('starting timer...')
        t1_start = perf_counter()

        # create references
        logger.info('creating references...')
        self._create_references()

        with torch.no_grad():
            prediction = self._model_wrap.predict(self.encoder_input, self.decoder_input).to(self._device)

        def objective(trial):
            """
            Ojective function for the optuna optimizer, used for hyperparameter optimization.
            The learning rate and mask initialization value are subject to hyperparameter optimization.
            For each trial the objection function finds the saliency map with gradient descent,
            by updating the saliency map parameters according to the calculated loss.
            The objective is to minimize the loss function.
            Stop counters help to speed up the process, by ending the trial, if the loss doesn't decrease fast enough.
            For each trial the saliency map and other relevant tensors are saved,
            so the tensors of the best trial can be loaded at the end of the hyperparameter search.
            """

            torch.autograd.set_detect_anomaly(True)

            learning_rate = trial.suggest_loguniform(
                "learning rate",
                low=self._explanation_config["lr_low"],
                high=self._explanation_config["lr_low"])
            mask_init_value = trial.suggest_uniform('mask initialisation value', 0., 1.)

            # todo: rework saliency map as class
            temp_saliency_map = self._fill_with(mask_init_value)

            optimizer = torch.optim.Adam(temp_saliency_map, lr=learning_rate)

            # calculate mask
            assert self._explanation_config["max_epochs"] > 0

            for epoch in range(self._explanation_config["max_epochs"]):  # mask 'training' epochs

                perturbated_prediction = self._get_perturbated_prediction(temp_saliency_map)
                loss, rmse = self._loss_function(
                    prediction,
                    perturbated_prediction,
                    temp_saliency_map
                )

                optimizer.zero_grad()  # set all gradients zero

                loss.backward()  # backpropagate mean loss
                optimizer.step()  # update mask parameters/minimize loss function

                self.print_epoch(epoch, loss)

            # trial_id = trial.number
            trial.set_user_attr("saliency map", temp_saliency_map)
            trial.set_user_attr("rmse", rmse.detach())
            trial.set_user_attr("perturbated_prediction", perturbated_prediction.detach())

            return loss

        # create saliency map

        logger.info('create saliency map...')
        study = optuna.create_study()
        study.optimize(
            objective,
            n_trials=self._explanation_config["n_trials"])

        t1_stop = perf_counter()
        logger.info("Elapsed time: {}".format(t1_stop - t1_start))

        # load best saliency map
        best_saliency_map = study.best_trial.user_attrs['saliency map']

        # save best saliency map
        self._saliency_map = best_saliency_map
        self._optimization_done = True

    def print_epoch(self, epoch, loss):

        if epoch % 100 == 0:  # print every 100 epochs
            logger.debug(
                'epoch {} / {} \t loss: {}'.format(
                    epoch,
                    self._explanation_config["max_epochs"],
                    loss.item()
                )
            )

    def rmse(self):  # todo write this function
        pass

    def save(self):  # todo save with name of specific timestep
        """
        Saves the whole class instance after optimization for potential future use and analyzing
        """
        if self._optimization_done:
            save_path = os.path.join(self._path, str(self.datetime.date()) + '_save')
            torch.save(self, save_path)
        else:
            logger.error("Please use optimize(), before saving the instance.")

    @staticmethod
    def load(target: str, date: pd.Timestamp = ''):  # todo ask for timestep
        try:
            default_path = os.path.join(
                MAIN_PATH,
                'oracles/interpretation/',
                target,
                str(date.date()) + '_save'
            )
            self = torch.load(default_path)
            if not isinstance(self, SaliencyMapUtil):
                raise TypeError
        except TypeError:
            logger.error('The file you tried to load is not a SaliencyMapUtil instance')
            self = None
        except FileNotFoundError:
            logger.error('no save file found in {}'.format(default_path))
            self = None
        return self

    def plot(
            self,
            plot_path=''
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
        # todo: fix plot feature axes (says only features)

        logger.info('creating saliency map plot...')

        # font sizes
        plt.rc('font', size=30)  # default font size
        plt.rc('axes', labelsize=30)  # fontsize of the x and y labels
        plt.rc('axes', titlesize=30)  # fontsize of the title

        fig2, ax2 = plt.subplots(1, figsize=(20, 14))

        # todo check for hourly resolution
        # create time axis
        start_index = self.datetime - pd.Timedelta(self.history_horizon, unit='h') # assumes hourly resolution
        stop_index = self.datetime + pd.Timedelta(self.forecast_horizon-1, unit='h') #datetime is first timestep of forecasting horizon
        time_axis = pd.date_range(start_index, stop_index, freq='h')
        time_axis_length = len(time_axis)

        # saliency heatmap
        # common = list(
        #     set(encoder_features) & set(decoder_features))  # features which are both encoder and decoder features
        # feature_axis_length = len(encoder_features) + len(decoder_features) - len(common)

        features = self._dataset.encoder_features + self._dataset.decoder_features
        saliency_heatmap = np.full(
            (time_axis_length, len(features)),
            fill_value=np.nan
        )  # for features not present in certain areas(nan), use different colour (white)

        # copy saliency map into one connected map
        saliency_heatmap[0:self.history_horizon, 0:self.num_encoder_features] =\
            self._saliency_map[0].cpu().detach().numpy()
        saliency_heatmap[self.history_horizon:, self.num_encoder_features:] =\
            self._saliency_map[1].cpu().detach().numpy()

        saliency_heatmap = np.transpose(saliency_heatmap)  # swap axes

        im = ax2.imshow(saliency_heatmap, cmap='jet',
                        norm=None, aspect='auto', interpolation='nearest', vmin=0, vmax=1, origin='lower')

        # create datetime x-axis
        plot_datetime = pd.array([''] * time_axis_length)  # looks better for plot
        datetime = time_axis
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
        if plot_path == '':  # if plot plath was not specified use default
                plot_path = os.path.join(self._path, str(self.datetime.date()))

        temp_save_path = plot_path + '_heatmap'
        fig2.savefig(temp_save_path)
        logger.info('plot saved in {}.'.format(temp_save_path))


def stop_function(epoch, loss):
    # todo probably going to be replaced by optuna.pruner or deleted completely
    if (epoch >= 1000) and (epoch < 3000):
        if (loss > 0.2) and (loss < 1):  # loss <1 to prevent stopping because mask out of [0,1] boundary
            return False
        else:
            return True

    elif (epoch >= 3000) and (epoch < 5000):
        if (loss > 0.1) and (loss < 1):  # loss <1 to prevent stopping because mask out of [0,1] boundary
            return False
        else:
            return True

    elif (epoch >= 5000) and (epoch < 10000):
        if (loss > 0.05) and (loss < 1):  # loss <1 to prevent stopping because mask out of [0,1] boundary
            return False
        else:
            return True