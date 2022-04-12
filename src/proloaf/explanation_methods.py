import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

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

from random import seed

import torch.nn as nn
import optuna
from time import perf_counter

logger = create_event_logger(__name__)


class SaliencyMap:

    def __init__(
            self,
            target: str,
            datetime: pd.DatetimeIndex,
            ref_batch_size=10,
            device='cpu',
            sep=';'
    ):
        logger.debug('reading config...')
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
        logger.debug('importing data...')
        df = pd.read_csv(os.path.join(MAIN_PATH, model_config["data_path"]), sep=sep)
        #time_column = df.loc[:, "Time"]

        # get scaler
        scaler = dh.MultiScaler(model_config["feature_groups"])

        # todo: preperation steps needed?
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
            device=device,
            **model_config
        )

        # load the trained forecasting NN model
        logger.debug('loading forecasting model...')

        self._modelhandler = mh.ModelHandler(
            work_dir=MAIN_PATH,
            config=model_config,
            tuning_config=model_tuning_config,
            device=device,
        )

        logger.debug('initializing saliency map...')
        history_horizon = self._modelhandler.model_wrap.history_horizon
        forecasting_horizon = self._modelhandler.model_wrap.forecast_horizon
        # todo determine if there are features which are both encoder and decoder
        num_encoder_features = len(self._modelhandler.model_wrap.encoder_features)
        num_decoder_features = len(self._modelhandler.model_wrap.decoder_features)
        logger.debug('number of encoder features: {}'.format(num_encoder_features))
        logger.debug('number of decoder features: {}'.format(num_decoder_features))

        self._saliency_map = (
            torch.zeros(history_horizon, num_encoder_features, requires_grad=True, device=device),
            torch.zeros(forecasting_horizon, num_decoder_features, requires_grad=True, device=device)
        )

        assert isinstance(datetime, pd.Timestamp)
        self._datetime = datetime
        logger.debug(
                    'saliency map initialized for the forecast of {} hours after the '
                    'date: {}, with a history horizon of {}.\n'
                    'The current forecasting model is set to {} '.format(
                        forecasting_horizon,
                        datetime,
                        history_horizon,
                        target
                    )
                )

        self._ref_batch_size = ref_batch_size
        self._device = device

    def _get_timestep(self):
        try:
            timestep = self._dataset.data.index[
                pd.to_datetime(self._dataset.data.Time) == self._datetime
            ]
            timestep = timestep.values
            assert len(timestep) == 1
            timestep = int(timestep[0])
            assert isinstance(timestep, int)
            self._timestep = timestep
        except:
            logger.error("An error has occurred while trying to read the datetime.")
        logger.debug('Timestep for saliency map with the date {!s} is {!s}'.format(self._datetime, timestep))
        return timestep


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

        dataset = self._dataset
        batch_size = self._ref_batch_size
        timestep = self._get_timestep()

        # creates reference for a certain timestep
        history_horizon = dataset.history_horizon
        forecast_horizon = dataset.forecast_horizon
        num_encoder_features = len(dataset.encoder_features)
        num_decoder_features = len(dataset.decoder_features)
        seed(1)  # seed random number generator

        features1_references_np = np.zeros(shape=(batch_size, history_horizon, num_encoder_features))
        features2_references_np = np.zeros(shape=(batch_size, forecast_horizon, num_decoder_features))

        dataset.to_tensor()
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

        return torch.Tensor(features1_references_np).to(self._device), torch.Tensor(features2_references_np).to(self._device)

    #__create_references = _create_references()

    def optimize(self):
        pass

    def plot(self):
        pass
