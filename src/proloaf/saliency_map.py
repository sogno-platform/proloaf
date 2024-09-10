import pandas as pd
import torch
import torch.nn as nn
import torch.optim.lr_scheduler
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from random import seed
import optuna
from time import perf_counter
from proloaf import metrics
import proloaf.event_logging as el
import proloaf.confighandler as ch
import proloaf.datahandler as dh
import proloaf.modelhandler as mh
import proloaf.tensorloader as tl

from proloaf.event_logging import create_event_logger
from proloaf.event_logging import timer

logger = create_event_logger(__name__)

sys.path.append("../")
MAIN_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(MAIN_PATH)

logger = el.create_event_logger(__name__)
optuna.logging.enable_propagation()
torch.set_default_dtype(torch.float32)

plt.rc("font", size=30)  # default font size
plt.rc("axes", labelsize=30)  # fontsize of the x and y labels
plt.rc("axes", titlesize=30)  # fontsize of the title

# todo tensorboard log
# todo run notebook
# todo embed saliency_map.md into website


class _SaliencyMap:
    def __init__(
        self,
        history_horizon,
        forecast_horizon,
        num_encoder_features,
        num_decoder_features,
        num_aux_features,
        num_targets,
        device,
        fill_value=float(0),
    ):
        encoder_map = torch.full((history_horizon, num_encoder_features), fill_value, device=device, requires_grad=True)
        encoder_aux_map = torch.full(
            (history_horizon, num_aux_features), fill_value=fill_value, device=device, requires_grad=True
        )
        decoder_map = torch.full(
            (forecast_horizon, num_decoder_features), fill_value=fill_value, device=device, requires_grad=True
        )
        decoder_aux_map = torch.full(
            (forecast_horizon, num_aux_features), fill_value=fill_value, device=device, requires_grad=True
        )
        last_target_map = torch.full((1, num_targets), fill_value=fill_value, device=device, requires_grad=True)
        self.input_maps = (encoder_map, encoder_aux_map, decoder_map, decoder_aux_map, last_target_map)

    def tensor_repr(self):
        return self.input_maps

    def get_sigmoid_maps(self):  # bound to [0,1]
        """ "
        applies the sigmoid function element wise to the mask, to create a saliency map.
        The internal parameters can take any arbitrary values
        while the actual saliency map can only have values between zero and one
        """

        return tuple(torch.sigmoid(map) for map in self.input_maps)


class SaliencyMapHandler:

    def __init__(self, target: str, sep=";"):

        # read saliency config
        logger.info("reading saliency.json config")
        ex_config_path = os.path.join("targets", target, "saliency.json")
        self._saliency_config = ch.read_config(config_path=ex_config_path, main_path=MAIN_PATH)

        # read forecasting model config
        logger.info("reading model config...")
        config_path = "./targets/" + target + "/config.json"
        model_config = ch.read_config(config_path=os.path.join(MAIN_PATH, config_path), main_path=MAIN_PATH)

        # import data
        logger.info("importing data...")
        df = pd.read_csv(os.path.join(MAIN_PATH, model_config["data_path"]), sep=sep)

        # setting device
        self._device = self.set_device(self._saliency_config["cuda_id"])
        logger.debug("Device: {}".format(self._device))

        # get scaler
        scaler = dh.MultiScaler(model_config["feature_groups"])

        # create timeseries dataset
        logger.info("preparing the dataset...")

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
            **model_config,
        )

        self._index_copy = self._dataset.data.index  # gets replaced by Time column after to_tensor()
        self._dataset.to_tensor()  # prepare dataset

        # loading the forecasting model
        try:
            logger.info("loading the forecasting model")
            model_wrap_path = os.path.join(MAIN_PATH, model_config["output_path"], model_config["model_name"] + ".pkl")
            self._model_wrap = mh.ModelHandler.load_model(path=model_wrap_path, locate=self._device)
        except:
            logger.error(
                "An error has occurred while trying to load the forecasting model."
                "The model has to be trained and saved as a loadable file."
            )

        # initialize saliency map
        logger.debug("initializing saliency map...")

        self._best_mask = self.init_saliency_map()

        # normation for matrix norm calculation
        self._max_norm = np.sqrt(sum(sig.numel() for sig in self._best_mask.get_sigmoid_maps()))
        # self._max_norm = torch.norm(
        #     torch.ones(
        #         self.history_horizon + self.forecast_horizon,
        #         self.num_encoder_features + self.num_decoder_features + self.num_aux_features,
        #     )
        # )

        self.datetime = pd.to_datetime(self._saliency_config["date"], format="%d.%m.%Y %H:%M:%S")

        # set interpretation path
        self._path = os.path.join(MAIN_PATH, self._saliency_config["rel_interpretation_path"], target + "/")

        if not os.path.exists(os.path.join(MAIN_PATH, self._saliency_config["rel_interpretation_path"])):
            os.mkdir(os.path.join(MAIN_PATH, self._saliency_config["rel_interpretation_path"]))

        if not os.path.exists(self._path):
            os.mkdir(self._path)

        self._optimization_done = False
        self.model_prediction = self._get_model_prediction()

    def init_saliency_map(self, init_value: float = float(0)):
        return _SaliencyMap(
            self.history_horizon,
            self.forecast_horizon,
            self.num_encoder_features,
            self.num_decoder_features,
            self.num_aux_features,
            self.num_targets,
            self._device,
            init_value,
        )

    @property
    def saliency_map(self):
        if not self._optimization_done:
            logger.error("The saliency map has not been optimized yet. The result shows only initialization values")
        return self._best_mask.get_sigmoid_maps()

    @staticmethod
    def set_device(cuda_id):
        logger.info("setting computation device...")
        if torch.cuda.is_available():
            if cuda_id is not None:
                torch.cuda.set_device(cuda_id)
            logger.debug("Current CUDA ID: {}".format(torch.cuda.current_device()))
            device = "cuda"
            return device

        else:
            device = "cpu"
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
            time_step = self._index_copy[pd.to_datetime(self._dataset.data.index) == self.datetime]
            time_step = time_step.values
            if not len(time_step) == 1:
                raise ValueError
            time_step = int(time_step[0])
            if not isinstance(time_step, int):
                raise TypeError
            logger.debug("Timestep for saliency map with the date {!s} is {!s}".format(self.datetime, time_step))
            self._time_step = time_step

        except ValueError:
            logger.error("An error has occurred while trying to read the datetime." "More then one time step.")
        except TypeError:
            logger.error("An error has occurred while trying to read the datetime." "Time step index is no integer")

    @property
    def time_step(self):
        if not hasattr(self, "_time_step"):
            logger.error("Time step has not been set yet. Please set a date first")
        else:
            return self._time_step

    @time_step.setter
    def time_step(self, time_step):
        raise RuntimeError(
            "The time step should not be set directly."
            "It is automatically set and updated, when setting the datetime property"
        )

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
    def num_aux_features(self):
        return len(self._dataset.aux_features)

    @property
    def num_decoder_features(self):
        return len(self._dataset.decoder_features)

    @property
    def num_targets(self):
        return len(self._dataset.target_id)

    def get_input(self):
        return self._dataset[self.time_step][:-1]

    @property
    def target(self):  # changes dynamically with time step
        return self._dataset[self.time_step][-1]

    @property
    def encoder_references(self):
        return self._encoder_references

    @property
    def decoder_references(self):
        return self._decoder_references

    @property
    def model_prediction(self):
        return self._model_prediction

    @model_prediction.setter
    def model_prediction(self, model_prediction):

        if hasattr(self, "_model_prediction"):
            logger.warning("Model prediction has already been made.")
            return  # model prediction only has to be made once
        else:
            self._model_prediction = model_prediction

    def _get_model_prediction(self):
        with torch.no_grad():
            return self._model_wrap.predict(*(input.unsqueeze(0) for input in self.get_input())).to(self._device)

    def get_references(self):
        if not hasattr(self, "_references"):
            logger.debug("References have not been created. Creating them now ...")
            self._references = self._create_references()
            logger.debug("... done")
        return self._references

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
        inputs_np = tuple(input.cpu().numpy() for input in self.get_input())

        # Test refactor
        test_sigmas = tuple(
            np.std(input, axis=0)[np.newaxis, np.newaxis, :]
            .repeat(self._saliency_config["ref_batch_size"], axis=0)
            .repeat(len(input), axis=1)
            for input in inputs_np
        )
        test_mus = tuple(np.zeros_like(sig) for sig in test_sigmas)
        test_noise = tuple(np.random.default_rng().normal(mu, sig, mu.shape) for mu, sig in zip(test_mus, test_sigmas))
        references_np = tuple(input[np.newaxis] + noise for input, noise in zip(inputs_np, test_noise))
        return tuple(torch.tensor(ref, dtype=torch.float32).to(self._device) for ref in references_np)

    def _get_perturbated_inputs(self, saliency_map: _SaliencyMap):
        """
        Perturbs the input data with the references. Each saliency map should take a value between zero and one
        for each time step and feature. A saliency map value of one leads to no perturbation and keeps the original
        input value.
        A saliency map value of zero leads to maximal perturbation, which means the original value is swapped by
        the reference value.
        A saliency map between zero and one leads to a weighted sum of the original input and the reference value.

        Parameters
        ----------
        saliency_map: The saliency map with which to perturbate the inputs.
                    All values should be between zero and one
        batch_number: The number of the reference of the reference batch.
                    The maximum batch number is set in the saliency.json config file

        Returns
        -------
        The perturbed input

        """
        # shape, sigmoid_map
        sigmoid_maps = saliency_map.get_sigmoid_maps()
        inputs = self.get_input()
        references = self.get_references()
        # references = tuple(
        #     ref[batch_number] for ref in self.get_references()
        # )
        inverse_saliency_maps = tuple(
            torch.sub(torch.ones(map.shape, device=self._device), map).to(self._device) for map in sigmoid_maps
        )  # elementwise 1-m

        input_summands = tuple(
            torch.mul(input, sig_map).to(self._device) for input, sig_map in zip(inputs, sigmoid_maps)
        )  # element wise multiplication

        reference_summands = tuple(
            torch.mul(ref, inv_sal_map).to(self._device) for ref, inv_sal_map in zip(references, inverse_saliency_maps)
        )

        perturbed_inputs = tuple(
            torch.add(input_summand, reference_summand).to(self._device)
            for input_summand, reference_summand in zip(input_summands, reference_summands)
        )
        return perturbed_inputs  # perturbated_input_enc, perturbated_input_dec

    def _get_perturbated_predictions(self, saliency_map: _SaliencyMap):
        """
        calculates the perturbed prediction by feeding the prediction model with the perturbed input.

        Parameters
        ----------
        saliency_map: The saliency map with which to perturbate the inputs.
                    All values should be between zero and one
        batch_number: The number of the reference of the reference batch.
                    The maximum batch number is set in the saliency.json config file

        Returns
        -------

        """
        # perturbate input
        perturbed_inputs = self._get_perturbated_inputs(saliency_map)

        # get prediction of perturbed input
        self._model_wrap.model.train()  # set model to train mode
        perturbated_predictions = self._model_wrap.predict(*tuple(p_input for p_input in perturbed_inputs)).to(
            self._device
        )

        return perturbated_predictions  # [batch_size,forecast_horizon, feature, feature_parameter]

    def criterion_loss(self, perturbated_prediction, criterion=metrics.Rmse()):
        """

        Parameters
        ----------
        perturbated_prediction
        criterion: the criterion with which to calculate the criterion loss

        Returns
        -------
        the criterion loss between the original model prediction and the prediction after perturbation
        """
        # model_prediction returns [batch,forecast_horizon, predictions]
        # batch size = 1
        target_prediction = self.model_prediction[0:]  # -> [forecast_horizon, predictions]

        return criterion(
            target_prediction[..., 0], perturbated_prediction
        )  # XXX it is not good to do [...,0] here it is not guaranteed that 0 is the index of the prediciton
        # (or that a prediction is even contained)

    def mask_weights_loss(self, saliency_map: _SaliencyMap):  # penalizes high mask parameter values
        """
        penalizes high mask parameter values by calculating the frobenius
        norm and normalize by dividing through "max_norm"
        saliency map should be a mask, which is restricted by [0,1] bounds (after sigmoid function)
        """
        saliency_map.get_sigmoid_maps()

        mw_loss = sum(tuple(torch.norm(sal_map) for sal_map in saliency_map.get_sigmoid_maps())) / self._max_norm

        return mw_loss

    def _loss_function(
        self,
        saliency_map: _SaliencyMap,
        perturbated_prediction,
        lambda_,
    ):
        """
        Calculates the loss function for the mask optimization process
            which is calculated by the smallest supporting region principle.
        A batch is the number of reference values created for each feature.
        The smallest supporting region loss is calculated by adding up the criterion loss and
            the mask weight loss.
        Lambda determines how big the mask weight loss should be weighted in the summation of the loss.

        """

        # force a probabilistic distribution ([0,1] bounds)
        # XXX is there just one prediction in here?
        loss_predictions = (
            self.criterion_loss(perturbated_prediction) / self._saliency_config["ref_batch_size"]
        )  # prediction loss
        loss_weights = lambda_ * self.mask_weights_loss(saliency_map)  # abs value of mask weights
        # loss2 =  self.mask_weights_loss(saliency_map)  # abs value of mask weights
        if self._init_loss_pred is None:
            self._init_loss_pred = loss_predictions.detach()

        if self._init_loss_weights is None:
            self._init_loss_weights = loss_weights.detach()
        if self._saliency_config.get("relative_errors", False):
            ssr_loss = loss_predictions / self._init_loss_pred + lambda_ * loss_weights / self._init_loss_weights
        else:
            ssr_loss = loss_predictions + lambda_ * loss_weights

        return ssr_loss.squeeze(), loss_predictions.squeeze()

    def _batch_loss(self, mask: _SaliencyMap, lambda_):
        """
        calculates the loss for the whole batch of references by summing the individual losses up.
        Before the loss is calculated for each reference, the sigmoid function is applied to the saliency map,
        so that the values only lie between zero and one

        Parameters
        ----------
        mask
        lambda_

        Returns
        -------
        the loss for the whole batch of references

        """
        batch_loss = torch.tensor(0.0, device=self._device)
        perturbated_prediction = self._get_perturbated_predictions(mask)
        for batch in perturbated_prediction:
            loss, rmse = self._loss_function(mask, batch.unsqueeze(0), lambda_)
            # logger.debug(f"batch:{batch!s}\t loss: {loss.item()}")
            batch_loss += loss
        return batch_loss

    def _objective(self, trial):
        """
        Ojective function for the optuna optimizer, used for hyperparameter optimization.
        The learning rate is subject to hyperparameter optimization.
        For each trial the objective function finds the saliency map with gradient descent,
        by updating the saliency map parameters according to the calculated loss.
        The objective is to minimize the loss function.
        For each trial the saliency map and other relevant tensors are saved,
        so the tensors of the best trial can be loaded at the end of the hyperparameter search.
        """

        torch.autograd.set_detect_anomaly(True)
        lambda_ = 1
        learning_rate = trial.suggest_float(
            "learning rate", low=self._saliency_config["lr_low"], high=self._saliency_config["lr_high"], log=True
        )

        mask = self.init_saliency_map(init_value=float(0))
        start_lr = 1
        optimizer = torch.optim.Adam(mask.tensor_repr(), lr=learning_rate)

        assert self._saliency_config["max_epochs"] > 0
        loss = np.inf
        for epoch in range(self._saliency_config["max_epochs"]):  # mask 'training' epochs
            loss = self._batch_loss(mask, self._saliency_config["lambda"])
            optimizer.zero_grad()  # set all gradients zero
            loss.backward()  # backpropagate mean loss
            optimizer.step()  # update mask parameters/minimize loss function

            if trial.should_prune():
                raise optuna.TrialPruned()
            self.print_epoch(epoch, loss, print_every=200)

        # trial_id = trial.number
        trial.set_user_attr("mask", mask)

        return loss

    @timer(logger)
    def create_saliency_map(self):
        self._optimize_time_step()

    def _optimize_time_step(self):
        """
        Creates an optuna study for the saliency map creation, which optimized the objective function.
        After, the best mask of all trials is saved.
        """
        # create references
        logger.info("creating references...")
        self._create_references()
        self._init_loss_pred = None
        self._init_loss_weights = None
        # create saliency map

        logger.info("create saliency map...")
        study = optuna.create_study(pruner=optuna.pruners.MedianPruner())
        study.optimize(self._objective, n_trials=self._saliency_config["n_trials"])

        # load best saliency map
        best_mask = study.best_trial.user_attrs["mask"]

        # save best saliency map
        self._best_mask = best_mask
        self._optimization_done = True

    def print_epoch(self, epoch, loss, print_every: int = 1):

        if epoch % print_every == 0:  # print every .. epochs
            logger.debug(
                "epoch {} / {} \t epoch loss: {}".format(epoch, self._saliency_config["max_epochs"], loss.item())
            )

    def save(self):
        """
        Saves the whole class instance after optimization for potential future use and analyzing
        """
        if self._optimization_done:
            save_path = os.path.join(self._path, str(self.datetime.date()) + "_save")
            torch.save(self, save_path)
        else:
            logger.error("Please use optimize(), before saving the instance.")

    @staticmethod
    def load(target: str, date: pd.Timestamp = "", rel_path: str = "oracles/interpretation/"):
        try:
            default_path = os.path.join(MAIN_PATH, "oracles/interpretation/", target, str(date.date()) + "_save")
            self = torch.load(default_path)
            if not isinstance(self, SaliencyMapHandler):
                raise TypeError
        except TypeError:
            logger.error("The file you tried to load is not a SaliencyMapUtil instance")
            self = None
        except FileNotFoundError:
            logger.error("no save file found in {}".format(default_path))
            self = None
        return self

    def plot(self, plot_path=""):
        """
        Creates the saliency map plot.
        The saliency map plot is split into an encoder(history horizon) part and a decoder(forecast horizon part)
            on the time axis.
        """
        # TODO Obviously this doesn't work with multiple target features
        # assert len(self._model_wrap.target_id) == 1  # function assumes 1 target
        logger.info("creating saliency map plot...")

        fig, ax = plt.subplots(1, figsize=(20, 14))

        # create time axis
        start_index = self.datetime - pd.Timedelta(self.history_horizon, unit="h")  # assumes hourly resolution
        stop_index = self.datetime + pd.Timedelta(
            self.forecast_horizon - 1, unit="h"
        )  # datetime is first timestep of forecasting horizon
        time_axis = pd.date_range(start_index, stop_index, freq="h")
        time_axis_length = len(time_axis)

        # saliency heatmap
        # common = list(
        #     set(encoder_features) & set(decoder_features))  # features which are both encoder and decoder features
        # feature_axis_length = len(encoder_features) + len(decoder_features) - len(common)

        features = self._dataset.encoder_features + self._dataset.decoder_features + self._dataset.aux_features
        saliency_heatmap = np.full(
            (time_axis_length, len(features)), fill_value=np.nan
        )  # for features not present in certain areas(nan), use different colour (white)

        # apply sigmoid again (0,1 boundary)
        sigmoid_map = self._best_mask.get_sigmoid_maps()

        # copy saliency map into one connected map
        saliency_heatmap[0 : self.history_horizon, 0 : self.num_encoder_features] = (
            sigmoid_map[0].cpu().detach().numpy()
        )
        saliency_heatmap[0 : self.history_horizon, self.num_encoder_features + self.num_decoder_features :] = (
            sigmoid_map[1].cpu().detach().numpy()
        )

        saliency_heatmap[
            self.history_horizon :, self.num_encoder_features : self.num_encoder_features + self.num_decoder_features
        ] = (sigmoid_map[2].cpu().detach().numpy())

        saliency_heatmap[self.history_horizon :, self.num_encoder_features + self.num_decoder_features :] = (
            sigmoid_map[3].cpu().detach().numpy()
        )

        saliency_heatmap = np.transpose(saliency_heatmap)  # swap axes

        im = ax.imshow(
            saliency_heatmap,
            cmap="jet",
            norm=None,
            aspect="auto",
            interpolation="nearest",
            vmin=0,
            vmax=1,
            origin="lower",
        )

        # create datetime x-axis
        plot_datetime = pd.array([""] * time_axis_length)  # looks better for plot
        datetime = time_axis
        for h in range(datetime.array.size):
            if datetime.array.hour[h] == 0:  # only show full date once per day
                plot_datetime[h] = datetime.array.strftime("%b %d %Y %H:%M")[h]
            else:
                if datetime.array.hour[h] % 12 == 0:  # every 12th hour
                    plot_datetime[h] = datetime.array.strftime("%H:%M")[h]

        # show ticks
        ax.set_xticks(np.arange(len(datetime)))
        ax.set_xticklabels(plot_datetime)
        feature_ticks = np.arange(len(features))
        ax.set_yticks(feature_ticks)
        ax.set_yticklabels(features)

        # rotate tick labels and set alignment
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # set titles and legends
        ax.set_xlabel("Time")
        ax.set_ylabel("Features")
        cbar = fig.colorbar(im)  # add colorbar

        # layout
        fig.tight_layout()

        # save heatmap
        if plot_path == "":  # if plot plath was not specified use default
            plot_path = os.path.join(self._path, str(self.datetime.date()))

        temp_save_path = plot_path + "_heatmap"
        fig.savefig(temp_save_path)

        logger.info("plot saved in {}.".format(temp_save_path))
        # tmp = plt.figure()
        fig, ax = plt.subplots(1, figsize=(20, 14))
        flat_sal_map = saliency_heatmap.flatten()
        flat_sal_map = np.sort(flat_sal_map[~np.isnan(flat_sal_map)])
        ax.plot(flat_sal_map)
        fig.tight_layout()
        temp_save_path = plot_path + "_dist"
        fig.savefig(temp_save_path)
        logger.info("plot saved in {}.".format(temp_save_path))

    def plot_predictions(self, plot_path: str = ""):
        """
        creates a plot with the target, the forecasting model prediction without perturbation
        and the forecasting model prediction with mask perturbation.
        Can only be called after the mask/saliency map has been calculated.
        The perturbed prediction is calculated by taking the mean over the whole random noise batch batch.

        """
        if not self._optimization_done:
            logger.error("The saliency map hasn't been optimized yet.")
            return
        else:
            saliency_map = self._best_mask
            target = self._dataset[self.time_step][-1].detach().cpu().numpy()
            # perturbed_predictions = [
            #     torch.unsqueeze(self._get_perturbated_predictions(saliency_map, i), dim=0)
            #     for i in range(self._saliency_config["ref_batch_size"])
            # ]
            # mean_perturbed_prediction = torch.mean(torch.cat(perturbed_predictions), dim=0)[:, 0].detach().numpy()
            mean_perturbed_prediction = (
                self._get_perturbated_predictions(saliency_map).mean(dim=0)[..., 0].detach().cpu().numpy()
            )
            model_prediction = self.model_prediction[0, :, :, 0].detach().cpu().numpy()
            fig, ax = plt.subplots(1, figsize=(20, 14))
            ax.set_xlabel("Time")
            ax.set_ylabel("Predictions")

            plt.plot(target, label="target")
            plt.plot(mean_perturbed_prediction, label="mean perturbed prediction")
            plt.plot(model_prediction, label="model prediction")
            ax.legend()
            if plot_path == "":  # if plot plath was not specified use default
                plot_path = os.path.join(self._path, str(self.datetime.date()))

            temp_save_path = plot_path + "_predictions"
            fig.savefig(temp_save_path)
            # plt.show()
