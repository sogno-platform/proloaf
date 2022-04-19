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

# todo create config file for global config
INTERPRETATION_PATH = os.path.join(MAIN_PATH, './oracles/interpretation/')
if not os.path.exists(INTERPRETATION_PATH):
    os.mkdir(INTERPRETATION_PATH)
REF_BATCH_SIZE = 10
MAX_EPOCHS = 1 # 10000
N_TRIALS = 1 # 50  # hyperparameter tuning trials
LR_LOW = 1e-5 #learning rate low boundary
LR_HIGH = 0.01 #learning rate low boundary


class SaliencyMapUtil:

    def __init__(
            self,
            target: str,
            datetime: pd.DatetimeIndex,
            ref_batch_size=REF_BATCH_SIZE,
            sep=';'
    ):
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
        def set_device():
            logger.info('setting computation device...')
            cuda_id = model_config["cuda_id"]  # todo read cuda_id from interpreter config
            if torch.cuda.is_available():
                if cuda_id is not None:
                    torch.cuda.set_device(cuda_id)
                logger.debug('Current CUDA ID: {}'.format(torch.cuda.current_device()))
                device = 'cuda'
                return device

            else:
                device = 'cpu'
                return device

        self._device = set_device()
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

        # create modelhandler
        #logger.debug('preparing the modelhandler...')

        modelhandler = mh.ModelHandler(
            work_dir=MAIN_PATH,
            config=model_config,
            tuning_config=model_tuning_config,
            device=self._device,
        )

        # load forecasting model
        try:
            logger.info('loading the forecasting model')
            model_wrap_path = os.path.join(MAIN_PATH, model_config["output_path"], model_config["model_name"] + '.pkl')
            self._model_wrap = mh.ModelHandler.load_model(path=model_wrap_path, locate=self._device)
        except:
            logger.error("An error has occured while trying to load the forecasting model."
                         "The model has to be trained and saved as a loadable file.")

        # initialize saliency map
        logger.debug('initializing saliency map...')

        # todo determine if there are features which are both encoder and decoder

        # todo self._saliency_map._encoder_map = ...
        self._saliency_map = (
            torch.zeros(self.history_horizon(), self.num_encoder_features(), requires_grad=True, device=self._device),
            torch.zeros(self.forecast_horizon(), self.num_decoder_features(), requires_grad=True, device=self._device)
        )

        assert isinstance(datetime, pd.Timestamp)
        self._datetime = datetime

        def datetime_to_timestep():
            try:
                time_step = self._dataset.data.index[
                    pd.to_datetime(self._dataset.data.Time) == self._datetime
                ]
                time_step = time_step.values
                assert len(time_step) == 1
                time_step = int(time_step[0])
                assert isinstance(time_step, int)
                logger.debug('Timestep for saliency map with the date {!s} is {!s}'.format(self._datetime, time_step))
                return time_step
            except:
                logger.error("An error has occurred while trying to read the datetime."
                             " Please use pandas datetime and correct format.")

        self._time_step = datetime_to_timestep()

        logger.debug(
                    'saliency map initialized for the forecast of {} hours after the '
                    'date: {}, with a history horizon of {}.\n'
                    'The current forecasting model is set to {} '.format(
                        self.forecast_horizon(),
                        self._datetime,
                        self.history_horizon(),
                        target
                    )
                )

        self._ref_batch_size = ref_batch_size

        self._path = os.path.join(INTERPRETATION_PATH, target + '/')
        if not os.path.exists(self._path):
            os.mkdir(self._path)



        self._optimization_done = False

    def history_horizon(self):
        return self._dataset.history_horizon

    def forecast_horizon(self):
        return self._dataset.forecast_horizon

    def num_encoder_features(self):
        return len(self._dataset.encoder_features)

    def num_decoder_features(self):
        return len(self._dataset.decoder_features)

    def _create_references(self):
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
        seed(1)  # seed random number generator

        features1_references_np = np.zeros(
            shape=(self._ref_batch_size,
                   self.history_horizon(),
                   self.num_encoder_features())
        )
        features2_references_np = np.zeros(
            shape=(self._ref_batch_size,
                   self.forecast_horizon(),
                   self.num_decoder_features())
        )

        self._dataset.to_tensor()
        inputs1_np = self._dataset[self._time_step][0].cpu().numpy()
        inputs2_np = self._dataset[self._time_step][1].cpu().numpy()

        for x in range(self.num_encoder_features()):  # iterate through encoder features
            feature_x = inputs1_np[:, x]
            mu = 0
            sigma = abs(np.std(feature_x))  # 0.3 is chosen arbitrarily # hier np.std nehmen
            for j in range(self._ref_batch_size):
                noise_feature1 = np.random.default_rng().normal(mu, sigma, self.history_horizon())  # create white noise series
                features1_references_np[j, :, x] = noise_feature1 + feature_x

        for x in range(self.num_decoder_features()):  # iterate through decoder features
            feature_x = inputs2_np[:, x]
            mu = 0
            sigma = abs(np.std(feature_x))  # 0.3 is chosen arbitrarily
            for j in range(self._ref_batch_size):
                noise_feature2 = np.random.default_rng().normal(mu, sigma, self.forecast_horizon())
                features2_references_np[j, :, x] = noise_feature2 + feature_x

        return torch.Tensor(features1_references_np).to(self._device), torch.Tensor(features2_references_np).to(self._device)

    def _fill_with(self, fill_value): # todo write documention string

        temp_saliency_map = (
            torch.full(
                (self.history_horizon(), self.num_encoder_features()),
                fill_value=fill_value,
                device=self._device,
                requires_grad=True
            ),
            torch.full(
                (self.forecast_horizon(), self.num_decoder_features()),
                fill_value=fill_value,
                device=self._device,
                requires_grad=True)
        )
        return temp_saliency_map

    def _write(self, new_saliency_map): # todo write this function
        if not isinstance(new_saliency_map, list): #todo list of what?
            TypeError("Saliency Map must be list of two 2D torch tensors")

    def _get_perturbated_prediction(self): # todo write this function
        pass

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
            The loss function is zero when the mask value is between zero and 1, otherwise it takes a value linearly rising with the mask norm
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

    def optimize(
            self,
            lr_low=LR_LOW,
            lr_high=LR_HIGH
    ):

        # todo rewrite function to automatically compute list of timestamps
        #  without having to reload the model every time
        # start counter
        logger.info('starting timer...')
        t1_start = perf_counter()

        # create references
        logger.info('creating references...')
        (encoder_references, decoder_references) = self._create_references()

        # get original inputs and predictions
        encoder_input = torch.unsqueeze(self._dataset[self._time_step][0], 0).to(self._device)
        decoder_input = torch.unsqueeze(self._dataset[self._time_step][1], 0).to(self._device)
        target = torch.unsqueeze(self._dataset[self._time_step][2], 0).to(self._device)

        with torch.no_grad():
            prediction = self._model_wrap.predict(encoder_input, decoder_input).to(self._device)

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

            learning_rate = trial.suggest_loguniform("learning rate", low=lr_low, high=lr_high)
            mask_init_value = trial.suggest_uniform('mask initialisation value', 0., 1.)

            inputs1_temp = torch.squeeze(encoder_input, dim=0).to(self._device)
            inputs2_temp = torch.squeeze(decoder_input, dim=0).to(self._device)

            # todo: rework saliency map as class
            temp_saliency_map = self._fill_with(mask_init_value)

            optimizer = torch.optim.Adam(temp_saliency_map, lr=learning_rate)

            stop_counter = 0

            # calculate mask
            for epoch in range(MAX_EPOCHS):  # mask 'training' epochs

                # create inverse masks # todo use _get perturbated prediction function
                inverse_saliency_map1 = torch.sub(torch.ones(inputs1_temp.shape, device=self._device),
                                                  temp_saliency_map[0]).to(self._device)  # elementwise 1-m
                inverse_saliency_map2 = torch.sub(torch.ones(inputs2_temp.shape, device=self._device),
                                                  temp_saliency_map[1]).to(self._device)  # elementwise 1-m
                input_summand1 = torch.mul(inputs1_temp, temp_saliency_map[0]).to(self._device)  # element wise multiplication
                input_summand2 = torch.mul(inputs2_temp, temp_saliency_map[1]).to(self._device)  # element wise multiplication

                # create perturbated series through mask
                # todo: write function for getting perturbated inputs from a saliency map
                reference_summand1 = torch.mul(encoder_references, inverse_saliency_map1).to(self._device)
                perturbated_input1 = torch.add(input_summand1, reference_summand1).to(self._device)
                reference_summand2 = torch.mul(decoder_references, inverse_saliency_map2).to(self._device)
                perturbated_input2 = torch.add(input_summand2, reference_summand2).to(self._device)

                # get prediction
                self._model_wrap.model.train()
                perturbated_prediction = self._model_wrap.predict(
                    perturbated_input1,
                    perturbated_input2
                ).to(self._device)

                loss, rmse = self._loss_function(
                    prediction,
                    perturbated_prediction,
                    temp_saliency_map
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
                optimizer.step()  # update mask parameters/minimize loss function

                if epoch % 1000 == 0:  # print every 100 epochs
                    print('epoch ', epoch, '/', MAX_EPOCHS, '...    loss:', loss.item())

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
            n_trials=N_TRIALS)

        t1_stop = perf_counter()
        logger.info("Elapsed time: {}".format(t1_stop - t1_start))

        # load best saliency map
        best_saliency_map = study.best_trial.user_attrs['saliency map']


        # save best saliency map
        self._saliency_map = best_saliency_map
        self._optimization_done = True

    def rmse(self):  # todo write this function
        pass

    def save(self):
        """
        Saves the whole class instance after optimization for potential future use and analyzing
        """
        if self._optimization_done:
            save_path = os.path.join(self._path, 'save')
            torch.save(self, save_path)
        else:
            logger.error("Please use optimize(), before saving the instance.")

    @staticmethod
    def load(path):
        self = torch.load(path)
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
        encoder_features = self._dataset.encoder_features
        decoder_features = self._dataset.decoder_features

        # font sizes
        plt.rc('font', size=30)  # default font size
        plt.rc('axes', labelsize=30)  # fontsize of the x and y labels
        plt.rc('axes', titlesize=30)  # fontsize of the title

        fig2, ax2 = plt.subplots(1, figsize=(20, 14))

        # todo check for hourly resolution
        # create time axis
        start_index = self._datetime - pd.Timedelta(self.history_horizon(), unit='h') # assumes hourly resolution
        stop_index = self._datetime + pd.Timedelta(self.forecast_horizon()-1, unit='h') #datetime is first timestep of forecasting horizon
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
        saliency_heatmap[0:self.history_horizon(), 0:self.num_encoder_features()] =\
            self._saliency_map[0].cpu().detach().numpy()
        saliency_heatmap[self.history_horizon():, self.num_encoder_features():] =\
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
                plot_path = os.path.join(self._path, str(self._datetime.date()))

        temp_save_path = plot_path + '_heatmap'
        fig2.savefig(temp_save_path)
        logger.info('plot saved in {}.'.format(temp_save_path))

