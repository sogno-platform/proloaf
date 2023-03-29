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
Provides implementations of different loss functions, as well as functions for evaluating model performance
"""
from __future__ import annotations
import sys
import numpy as np
import torch
from abc import ABC, abstractmethod
from typing import List, Tuple, Union, Literal, Optional, Iterable
import inspect
from statistics import NormalDist


class QuantilePrediction:
    """
    Common prediction format.

    Parameters
    ----------
    values: torch.tensor
        Tensor (..., quantile) representing the predicted values for each quantile.
    quantiles: Iterable[float]:
        List of the quantiles in the same order as the predictions.
    """

    def __init__(self, values: torch.tensor, quantiles: Iterable[float]):
        self.values = values
        self.quantiles = list(quantiles)

    def get_gauss_params(self) -> torch.tensor:
        """
        Estimate expectation value and std. deviation from quantile prediction.
        This is approximation, for the mean the pdf is treated as a histogram.
        For the std. dev, (quantile_value - mean)/quantile_z is averaged over the quantiles.


        Returns
        -------
        torch.tensor
            Tensor (..., valuetype), [...,0] corresponds to the predicted mean, while [...,1] corresponds to the std. deviation.
            Meaning of dimensions depends on the dimensions of the values in `self.values`. Usually the tensor will be 3D (sample, )

        Raises
        ------
        ValueError
            If the median (quantile 0.5) is not in the predicted quantiles.
        """
        mean = self.get_mean()  # self.get_quantile(0.5)
        # compare_to_get_mean = torch.max(torch.abs(self.get_mean() - mean))
        intervals = self.select_quantiles(
            [quant for quant in self.quantiles if quant != 0.5]
        ).values

        z_values = torch.tensor(
            [NormalDist().inv_cdf(quant) for quant in self.quantiles if quant != 0.5],
            device=self.values.device,
        )

        # Assume median is expectation value (which is true assuming gaussian distribution)
        # calculate std-deviation for each quantile
        # should be the same for each quantile (but not timestep and sample) assuming perfectly normal distributed predictions
        sigma = (intervals - mean) / z_values
        # TODO replace torch.nansum/size with torch.nanmean once that makes it into official pytorch
        sigma = torch.nansum(sigma, dim=-1, keepdim=True) / (sigma.size()[-1])
        return torch.cat((mean, sigma), dim=-1)

    @staticmethod
    def from_gauss_params(
        values: torch.tensor, quantiles: Iterable[float]
    ) -> QuantilePrediction:
        """
        Estimate expectation value and std. deviation from quantile prediction.

        Note: Currently the estimation median = mean is used

        Parameters
        -------
        values : torch.tensor
            Tensor (..., valuetype), [...,0] corresponds to the predicted mean, while [...,1] corresponds to the std. deviation.
            Meaning of dimensions depends on the dimensions of the values in `self.values`. Usually the tensor will be 3D (sample, )

        quantiles : Iterable[float]
            Quantiles to be included in the QuantilePrediction.
        """
        quantiles = list(quantiles)
        z_values = torch.tensor(
            [NormalDist().inv_cdf(quant) for quant in quantiles], device=values.device
        )
        return QuantilePrediction(
            values[..., 0:1] + z_values * values[..., 1:2], quantiles
        )

    def get_mean(self) -> torch.tensor:
        """
        Approximates the mean from the quantile values.
        From the quantiles a histogram is created, the mean is approximated as the mean of that histogram.
        The probability density is implicitly approximated as constant between to neighboring quantiles.
        """
        quantiles = np.array(self.quantiles)
        sorted_idx = np.argsort(quantiles)
        quantiles = quantiles[sorted_idx]
        intervals = torch.tensor(
            quantiles[1:] - quantiles[:-1], device=self.values.device
        )
        values = self.values[..., sorted_idx]
        center_values = (values[..., 1:] + values[..., :-1]) / 2

        # weighted sum over the intervals

        return torch.sum(
            (intervals * center_values), dim=-1, keepdim=True
        ) / (  # intervals[None, None, :]
            quantiles[-1] - quantiles[0]
        )

    def get_quantile(self, quantile: float) -> torch.tensor:
        """
        Get a specific quantile prediction from this IntervalPrediction.

        Parameters
        ----------
        quantile: float,
            Quantile you want to select.

        Returns
        -------
        torch.tensor:
            The selected quantile as Tensor removes the quantile dimension (last dimension of `self.value`)

        Raises:
            ValueError if the requested quantile is not among the predicted ones.
        """
        return self.values[..., self.quantiles.index(quantile)]

    def select_quantiles(
        self, quantiles: Iterable[float], inplace=False
    ) -> QuantilePrediction:
        """
        Get a narrow the prediction down to the selected quantiles.

        Parameters
        ----------
        quantiles: List[float],
            Quantiles you want to select.
        inplace:
            If False a new QuantilePrediction will be created and return,
            if True the existing QuantilePrediction will be modified.

        Returns
        -------
        torch.tensor:
            The selected quantile as Tensor (..., quantile).
            The last dimension coresponds to quantiles.

        Raises:
            ValueError if one of the requested quantiles is not among the predicted ones.
        """
        quantiles = list(quantiles)
        indices = [self.quantiles.index(quant) for quant in quantiles]
        if inplace:
            self.values = self.values[..., indices]
            self.quantiles = quantiles
            return self
        else:
            return QuantilePrediction(self.values[..., indices], quantiles)

    def select_upper_bound(self, inplace=False) -> QuantilePrediction:
        """
        Generates QuantilePrediction only containing the greatest quantile.

        Parameters
        ----------
        inplace:
            If False a new QuantilePrediction will be created and return, if True the existing QuantilePrediction will be modified.

        Returns
        -------
        QuantilePrediction
            contains the greatest quantile.
        """
        upper_quantile = max(self.quantiles)
        return self.select_quantiles((upper_quantile,), inplace=inplace)

    def select_lower_bound(self, inplace=False) -> QuantilePrediction:
        """
        Generates QuantilePrediction only containing the lowest quantile.

        Parameters
        ----------
        inplace:
            If False a new QuantilePrediction will be created and return, if True the existing QuantilePrediction will be modified.

        Returns
        -------
        QuantilePrediction
            contains the lowest quantile.
        """
        lower_quantile = min(self.quantiles)
        return self.select_quantiles((lower_quantile,), inplace=inplace)


class Metric(ABC):
    """
    Base class for prediction evaluation metrics, defining the function itself,
    saving all metric specific parameters in per instance and providing some methods to convieniently use metrics.

    Parameters
    ----------
    func : callable
        A callable loss function
    **options
        Any parameters that belong to the metric, like quantiles etc.
    """

    alpha = 0.05

    def __init__(self, **options):
        self.options: dict = options
        self.input_labels: List[str]
        self.id: str = self.__class__.__name__

    def __call__(
        self,
        target: torch.tensor,
        predictions: torch.tensor,
        avg_over: Union[Literal["time"], Literal["sample"], Literal["all"]] = "all",
        **kwargs,
    ) -> torch.tensor:
        """
        Calculates the value of this metric using the options set in '__init__()'

        Parameters
        ----------
        target: torch.tensor
            Target values from the training or validation dataset. Dimensions have to be (sample number, timestep, 1).
        predictions: torch.tensor
            Predicted values on the same dataset as target. Dimension have to be (sample number, timestep, label number).
        avg_over: str
            One of "time", "sample", "all", averages the the results over the coresponding axis.

        Returns
        -------
        torch.tensor
            The gaussian negative log likelihood loss, which depending on the value of 'avg_over'
            is either a scalar (overall loss) or 1d-array over the horizon or the sample.
        """
        return self.func(
            target=target,
            predictions=predictions,
            avg_over=avg_over,
            **{
                **self.options,  # Combine preset and acute parameters perfering the acute ones
                **kwargs,
            },
        )

    @staticmethod
    def set_global_default_alpha(alpha=0.05):
        """
        Set default alpha (probability for type 1 Error) for all further metrics.
        This might be more consitent than making sure all metrics use the same value.

        Parameters
        ----------
        alpha : float, default = 0.05
            Probability for type 1 Error
        """
        Metric.alpha = alpha

    # @abstractmethod
    def get_quantile_prediction(
        self, predictions: torch.tensor, quantiles: Optional[List[float]], **kwargs
    ) -> QuantilePrediction:
        """
        Calculates the values for each quantile for the metric.
        For metrics using the normal distribution this will correspond to the confidence interval.
        Parameters
        ----------
        predictions: torch.tensor
            Predicted values on the same dataset as target. Dimension have to be (sample number, timestep, label number).

        Returns
        -------
        QuantilePrediciton
            Prediciton for the specified quantiles.
        """
        raise NotImplementedError(
            f"get_prediction is not available for {self.__class__}"
        )

    def from_quantiles(
        self,
        target: torch.tensor,
        quantile_prediction: QuantilePrediction,
        avg_over: Union[Literal["time"], Literal["sample"], Literal["all"]],
        **kwargs,
    ) -> torch.tensor:
        """
        Calculates the value of the metric based on the values for each quantile over the timeframe.

        Parameters
        ----------
        target: torch.tensor
            Target values from the training or validation dataset. Dimensions have to be (sample number, timestep, 1).
        quantile_prediction: QuantilePrediction
            A prediction for several quantiles.
            Some of the metrics have additional requirements, like a predicted median or atleast one symetric interval in the quantiles.
        avg_over: str
            One of "time", "sample", "all", averages the the results over the coresponding axis.

        Returns
        -------
        torch.tensor
            Value of the metric, which depending on the value of 'avg_over'
            is either a 0d-tensor (overall loss) or 1d-tensor over the horizon or the sample.
        """
        raise NotImplementedError(
            f"from_quantiles is not available for {self.__class__}"
        )

    @staticmethod
    @abstractmethod
    def func(
        target: torch.tensor,
        predictions: torch.tensor,
        avg_over: Union[Literal["time"], Literal["sample"], Literal["all"]] = "all",
        **kwargs,
    ) -> torch.tensor:
        """
        Calcualtion of the metrics value. Direct use is not recommended, instead create an object and call it to keep parameters consistent throughout its use.

        Parameters
        ----------
        target: torch.tensor
            Target values from the training or validation dataset. Dimensions have to be (sample number, timestep, 1).
        predictions: torch.tensor
            Predicted values on the same dataset as target. Dimension have to be (sample number, timestep, label number).
        avg_over: str, default = "all"
            One of "time", "sample", "all", averages the the results over the coresponding axis.
        **kwargs
            Additional metric specific parameters, these can be set when initializing an object and are then used when calling the object.

        Returns
        -------
        torch.tensor
            Negative-log-Likelihood, which depending on the value of 'avg_over'
            is either a 0d-tensor (overall loss) or 1d-tensor over the horizon or the sample.
        """
        pass


def tensor_mean_helper(tens: torch.Tensor, every_dim_but: int):
    """
    Calculates the mean over every dimension but the specified.

    Parameters
    ----------
    tens: torch.Tensor
        Tensor with data to be averaged.
    every_dim_but: int
        Dimension which should NOT be averaged over.

    """
    return tens.transpose(0, every_dim_but).view(tens.size(every_dim_but), -1).mean(1)


def avg_over_dims(
    tens: torch.tensor,
    over_dims: List[Literal["sample"], Literal["time"], Literal["feature"]] = "all",
):
    """
    Calculates the mean over every specified dimension by name.

    Parameters
    ----------
    tens: torch.Tensor
        Tensor with data to be averaged.
    over_dims: List["sample", "time", "features"] || "all"
        list of all dimensions to average over or "all" for all dimensions.
    """

    if isinstance(over_dims, str):
        over_dims = (over_dims,)
    dim_index = {"sample": 0, "time": 1, "feature": 2}
    if over_dims == "all" or "all" in over_dims:
        dims = (0, 1, 2)
    else:
        dims = tuple((dim_index[name] for name in over_dims))
    return torch.mean(tens, dim=dims, keepdim=True, dtype=torch.float32)


class AutoEncoderLoss(Metric):
    """
    Loss wrapping an MSE to train an Autoencoder.
    To preserve naming in proloaf this loss ignores the target and
    instead calculates the error between `predictions` and the inputs to the encoder.
    The loss labels are the same as the ones of the wrapped loss

    Parameters
    ----------
    loss_metric: Metric
        Metric used to evaluate the reconstructed input data against the original input data.
    """

    def __init__(self, loss_metric=None):
        loss_metric = Mse() if loss_metric is None else loss_metric
        # if len(loss_metric.input_labels) != 1:
        #     raise ValueError(
        #         "The Autoencoder loss can only be evaluated with single feature losses like MSE."
        #     )
        super().__init__(metric=loss_metric)
        self.input_labels = loss_metric.input_labels

    def get_quantile_prediction(
        self,
        predictions: torch.tensor,
        inputs_enc: torch.Tensor,
        inputs_enc_aux: torch.Tensor,
        target: torch.Tensor = None,
        quantiles: Optional[List[float]] = None,
        **kwargs,
    ) -> QuantilePrediction:
        """
        Calculates the values for each quantile for the metric.
        For metrics using the normal distribution this will correspond to the confidence interval.

        Parameters
        ----------
        predictions: torch.tensor
            Predicted values on the same dataset as target. Dimension have to be (sample number, timestep, label number).
        avg_over: str
            One of "time", "sample", "all", averages the the results over the coresponding axis.
        inputs_enc: torch.tensor
            Input data used in the encoder, this is used to construct the target for the autoencoder decoder.
        inputs_enc_aux: torch.tensor
            Auxillary input data used in the encoder, this is used to construct the target for the autoencoder decoder.
        target: torch.tensor
            Target values from the training or validation dataset. Dimensions have to be (sample number, timestep, 1).
            Will be ignored for this loss and is only there for compatability with other metrics.
        Returns
        -------
        QuantilePrediciton
            Prediciton for the specified quantiles.

        """
        return self.options["metric"].get_quantile_prediction(
            predictions=predictions,
            quantiles=quantiles,
            target=torch.flip(
                torch.cat((inputs_enc, inputs_enc_aux), dim=2)[:, :-1], dims=(1,)
            ),
            **kwargs,
        )

    def from_quantiles(
        self,
        target: torch.Tensor,
        quantile_prediction: QuantilePrediction,
        inputs_enc: torch.Tensor,
        inputs_enc_aux: torch.Tensor,
        avg_over: Union[Literal["time"], Literal["sample"], Literal["all"]],
        **kwargs,
    ) -> torch.tensor:
        """Calculates the value of the metric based on the values for each quantile over the timeframe.

        Parameters
        ----------
        target: torch.tensor
            Target values from the training or validation dataset. Dimensions have to be (sample number, timestep, 1).
            Will be ignored for this loss and is only there for compatability with other metrics.
        quantile_prediction: QuantilePrediction
            A prediction for several quantiles. Has to contain atleast the median prediction and one additional one to estimate the std. deviation.
            The mean is estimated to be the median as it would be for a gaussian distribution.
        avg_over: str
            One of "time", "sample", "all", averages the the results over the coresponding axis.
        inputs_enc: torch.tensor
            Input data used in the encoder, this is used to construct the target for the autoencoder decoder.
        inputs_enc_aux: torch.tensor
            Auxillary input data used in the encoder, this is used to construct the target for the autoencoder decoder.

        Returns
        -------
        torch.tensor
            Value of the metric, which depending on the value of 'avg_over'
           A 3d-tensor with length 1 in dimensions that where averaged over ([sample,time,feature]).
        """
        flip_target = torch.flip(
            torch.cat((inputs_enc, inputs_enc_aux), dim=2)[:, :-1], dims=(1,)
        )
        return self.options["metric"].from_quantiles(
            target=flip_target,
            quantile_prediction=quantile_prediction,
            avg_over=avg_over,
            **kwargs,
        )

    @staticmethod
    def func(
        target: torch.Tensor,
        predictions: torch.Tensor,
        inputs_enc: torch.Tensor,
        inputs_enc_aux: torch.Tensor,
        metric: Metric,
        avg_over: Union[Literal["time"], Literal["sample"], Literal["all"]] = "all",
        **_,
    ) -> torch.tensor:
        flip_target = torch.flip(
            torch.cat((inputs_enc, inputs_enc_aux), dim=2)[:, :-1], dims=(1,)
        )
        return metric(
            torch.flip(
                torch.cat((inputs_enc, inputs_enc_aux), dim=2)[:, :-1], dims=(1,)
            ),
            predictions,
            avg_over,
        )


class DualModelLoss(Metric):
    def __init__(self, forecast_loss=None, reconstruction_loss=None, weight=0.5):
        """Loss for training a hybrid model comprised of a encoder and two decoders.
        One of the decoders will be evaluated using the forecast loss,
        the other will be evaluated as autoencoder using the MSE based autoencoder loss.
        The target contious to be the forecasting target,
        data transformation for the autoencoder part is done automatically.

        Parameters
        ----------
        forecast_loss: Metric, default = None
            Metric used as loss for the forecasting of the model. Default corresponds to Mse
        reconstruction_loss: Metric, default = None
            Metric used to evaluate the reconstructed input data against the original input data.
            Default corresponds to Mse
        """
        forecast_loss = Mse() if forecast_loss is None else forecast_loss
        reconstruction_loss = Mse()
        super().__init__(
            metrics=[forecast_loss, AutoEncoderLoss(reconstruction_loss)], weight=weight
        )
        self.input_labels = (
            forecast_loss.input_labels,
            reconstruction_loss.input_labels,
        )

    def get_quantile_prediction(
        self,
        predictions: torch.tensor,
        quantiles: Optional[List[float]] = None,
        **kwargs,
    ) -> Tuple[QuantilePrediction]:
        """Calculates the values for each quantile for the metric.
        For metrics using the normal distribution this will correspond to the confidence interval.

        Parameters
        ----------
        predictions: torch.tensor
            Predicted values on the same dataset as target. Dimension have to be (sample number, timestep, label number).
        avg_over: str
            One of "time", "sample", "all", averages the the results over the coresponding axis.
        inputs_enc: torch.tensor
            Input data used in the encoder, this is used to construct the target for the autoencoder decoder.
        inputs_enc_aux: torch.tensor
            Auxillary input data used in the encoder, this is used to construct the target for the autoencoder decoder.
        target: torch.tensor
            Target values from the training or validation dataset. Dimensions have to be (sample number, timestep, 1).
            Will be ignored for this loss and is only there for compatability with other metrics.
        Returns
        -------
        QuantilePrediciton
            Prediciton for the specified quantiles.

        """
        q_preds = tuple(
            (
                metric.get_quantile_prediction(
                    predictions=predictions[i], quantiles=quantiles, **kwargs
                )
                for i, metric in enumerate(self.options["metrics"])
            )
        )
        return q_preds  # 1. batch, 2. time, 3. feature, 4. quantile
        # return super().get_quantile_prediction(predictions, quantiles, **kwargs)

    def from_quantiles(
        self,
        target: torch.tensor,
        quantile_prediction: Tuple[QuantilePrediction],
        inputs_enc: torch.Tensor,
        inputs_enc_aux: torch.Tensor,
        avg_over: Union[Literal["time"], Literal["sample"], Literal["all"]] = "all",
        **_,
    ) -> torch.tensor:
        """
        Calculates the value of the metric based on the values for each quantile over the timeframe.

        Parameters
        ----------
        target: torch.tensor
            Target values from the training or validation dataset. Dimensions have to be (sample number, timestep, 1).
        quantile_prediction: QuantilePrediction
            A prediction for several quantiles. Has to contain atleast the median prediction and one additional one to estimate the std. deviation.
            The mean is estimated to be the median as it would be for a gaussian distribution.
        avg_over: str
            One of "time", "sample", "all", averages the the results over the coresponding axis.
        inputs_enc: torch.tensor
            Input data used in the encoder, this is used to construct the target for the autoencoder decoder.
        inputs_enc_aux: torch.tensor
            Auxillary input data used in the encoder, this is used to construct the target for the autoencoder decoder.

        Returns
        -------
        torch.tensor
            Value of the metric, which depending on the value of 'avg_over'
           A 3d-tensor with length 1 in dimensions that where averaged over ([sample,time,feature]).
        """
        w = self.options["weight"]
        w = (w, 1 - w)
        metric_values = tuple(
            (
                w[i]
                * metric.from_quantiles(
                    quantile_prediction=quantile_prediction[i],
                    target=target,
                    inputs_enc=inputs_enc,
                    inputs_enc_aux=inputs_enc_aux,
                    avg_over=avg_over,
                )
                for i, metric in enumerate(self.options["metrics"])
            )
        )
        return torch.sum(torch.cat(metric_values, dim=-1), dim=-1)

    @staticmethod
    def func(
        target: torch.Tensor,
        predictions: torch.Tensor,
        inputs_enc: torch.Tensor,
        inputs_enc_aux: torch.Tensor,
        avg_over: Union[Literal["time"], Literal["sample"], Literal["all"]] = "all",
        metrics: List[Metric] = [],
        weight: float = 0.5,
        **_,
    ) -> torch.tensor:
        weights = (weight, 1 - weight)

        return torch.sum(
            torch.cat(
                tuple(
                    (
                        weights[i]
                        * metric(
                            target=target,
                            predictions=predictions[i],
                            inputs_enc=inputs_enc,
                            inputs_enc_aux=inputs_enc_aux,
                            avg_over=avg_over,
                        )
                        for i, metric in enumerate(metrics)
                    )
                ),
                dim=-1,
            ),
            dim=-1,
        )


class NllGauss(Metric):
    """
    Gaussion-negativ-log-likelihood.

    Parameters
    -------
    alpha : float, default = global default (0.05 if not otherwise specified)
            Predicted probability of violating the bound of the prediction interval. For information on the global default see 'Metrics.set_global_default(...)'
    """

    def __init__(self, alpha=None):
        if alpha is None:
            alpha = Metric.alpha
        super().__init__(alpha=alpha)
        self.input_labels = ["expected_value", "log_variance"]

    def get_quantile_prediction(
        self,
        predictions: torch.tensor,
        quantiles: Optional[Iterable[float]] = None,
        **_,
    ) -> QuantilePrediction:
        """
        Calculates the values for each quantile for the metric.
        For metrics using the normal distribution this will correspond to the confidence interval.

        Parameters
        ----------
        predictions: torch.tensor
            Predicted values on the same dataset as target. Dimension have to be (sample number, timestep, label number).
        alpha : float, default = None
            Predicted probability of violating the bound of the prediction interval. If no alpha is specified the class instance alpha specified is used. To avoid confusion use of this parameter is discouraged.

        Returns
        -------
        QuantilePrediciton
            Prediciton for the specified quantiles.

        """
        if quantiles is None:
            alpha_half = self.options["alpha"] / 2.0
            quantiles = (1 - alpha_half, alpha_half, 0.5)

        sigma = (0.5 * predictions[..., 1]).exp()
        return QuantilePrediction.from_gauss_params(
            torch.stack((predictions[..., 0], sigma), dim=-1), quantiles
        )

    def from_quantiles(
        self,
        target: torch.tensor,
        quantile_prediction: QuantilePrediction,
        avg_over: Union[Literal["time"], Literal["sample"], Literal["all"]] = "all",
        **_,
    ) -> torch.tensor:
        """
        Calculates the value of the metric based on the values for each quantile over the timeframe.

        Parameters
        ----------
        target: torch.tensor
            Target values from the training or validation dataset. Dimensions have to be (sample number, timestep, 1).
        quantile_prediction: QuantilePrediction
            A prediction for several quantiles. Has to contain atleast the median prediction and one additional one to estimate the std. deviation.
            The mean is estimated to be the median as it would be for a gaussian distribution.
        avg_over: str
            One of "time", "sample", "all", averages the the results over the coresponding axis.
        alpha : float, default = None
            Predicted probability of violating the bound of the prediction interval.
            If no alpha is specified the class instance alpha specified is used. To avoid confusion use of this parameter is discouraged.

        Returns
        -------
        torch.tensor
            Value of the metric, which depending on the value of 'avg_over'
            is either a 0d-tensor (overall loss) or 1d-tensor over the horizon or the sample.
        """
        gauss_params = quantile_prediction.get_gauss_params()
        gauss_params[..., 1] = gauss_params[..., 1].log() * 2
        return self(
            target,
            gauss_params,
            avg_over=avg_over,
        )

    @staticmethod
    def func(
        target: torch.tensor,
        predictions: torch.tensor,
        avg_over: Union[Literal["time"], Literal["sample"], Literal["all"]] = "all",
        **_,
    ) -> torch.tensor:
        """
        Calculates gaussian negative log likelihood score.

        Parameters
        ----------
        target : torch.tensor
            The true values of the target variable, dimensions are (sample number, timestep, 1).
        predictions :  torch.tensor
            - predictions[...,0] = expected_value, a torch.tensor containing predicted expected values
            of the target variable
            - predictions[...,1] = log_variance, approx. equal to log(2*pi*sigma^2)
        avg_over: str, default = "all"
            One of "time", "sample", "all", averages the the results over the coresponding axis.

        Returns
        -------
        torch.tensor
            The gaussian negative log likelihood loss, which depending on the value of 'avg_over'
            is either a 0d-tensor (overall loss) or 1d-tensor over the horizon or the sample.
        """
        assert predictions.size()[-1] == 2
        expected_value = predictions[..., 0]
        log_variance = predictions[..., 1]
        # y, y_pred, var_pred must have the same shape
        assert (
            target.shape == expected_value.shape
        )  # target.shape = torch.Size([batchsize, horizon, # of target variables]) e.g.[64,40,1]
        assert target.shape == log_variance.shape

        squared_errors = (target - expected_value) ** 2
        return avg_over_dims(
            squared_errors / (2 * log_variance.exp()) + 0.5 * log_variance,
            over_dims=avg_over,
        )


class PinnballLoss(Metric):
    """Calculates pinball loss or quantile loss against the specified quantiles.

    Parameters
    -------
    quantiles: List[float]: None
        List of values between 0 and 1, the quantiles that predicted by the model.
        Defaults to None in wich case the quantiles are set to [1-alpha, alpha].
        For information on the global default see 'Metrics.set_global_default(...)'
    """

    def __init__(self, quantiles: List[float] = None):
        if quantiles is None:
            quantiles = [1.0 - Metric.alpha / 2, Metric.alpha / 2, 0.5]
        super().__init__(quantiles=quantiles)
        self.input_labels = [f"quant[{quant}]" for quant in quantiles]

    def get_quantile_prediction(
        self,
        predictions: torch.tensor,
        quantiles: Optional[Iterable[float]] = None,
        **_,
    ) -> QuantilePrediction:
        if quantiles is None:
            return QuantilePrediction(predictions, self.options["quantiles"])
        else:
            return QuantilePrediction(
                predictions, self.options["quantiles"]
            ).select_quantiles(quantiles, inplace=True)

    def from_quantiles(
        self,
        target: torch.tensor,
        quantile_prediction: QuantilePrediction,
        avg_over: Union[Literal["time"], Literal["sample"], Literal["all"]] = "all",
        **_,
    ):
        """
        Calculates the value of the metric based on the values for each quantile over the timeframe.

        Parameters
        ----------
        target: torch.tensor
            Target values from the training or validation dataset. Dimensions have to be (sample number, timestep, 1).
        quantile_prediction: QuantilePrediction
            A prediction for several quantiles.
        Returns
        -------
        torch.tensor
            Value of the metric, which depending on the value of 'avg_over'
            is either a 0d-tensor (overall loss) or 1d-tensor over the horizon or the sample.
        """
        # if alpha is not None:
        #     quantiles = [1 - alpha, alpha]
        # else:
        #     quantiles = None
        return self(
            target,
            quantile_prediction.values,
            quantiles=quantile_prediction.quantiles,
            avg_over=avg_over,
        )

    @staticmethod
    def func(
        target: torch.tensor,
        predictions: torch.tensor,
        quantiles: List[float],
        avg_over: Union[Literal["time"], Literal["sample"], Literal["all"]] = "all",
        **_,
    ):
        """
        Calculates pinball loss or quantile loss against the specified quantiles

        Parameters
        ----------
        target : torch.tensor
            The true values of the target variable. Dimensions are (sample number, timestep, 1).
        predictions : torch.tensor
            The predicted values for each quantile. Dimensions are (sample number, timestep, quantile).
        quantiles : List[float]
            Quantiles that we are estimating for
        avg_over: str
            One of "time", "sample", "all", averages the the results over the coresponding axis.
        Returns
        -------
        float
            The total quantile loss (the lower the better)

        Raises
        ------
        NotImplementedError
            When 'avg_over' is set to anything but "all", "time", or "sample".
        """
        errors = target.unsqueeze(-1) - predictions
        quantiles_tensor = torch.tensor([[[quantiles]]], device=predictions.device)

        upper = quantiles_tensor * errors
        lower = (quantiles_tensor - 1) * errors
        loss = torch.sum(torch.max(upper, lower), dim=-1)
        return avg_over_dims(loss, over_dims=avg_over)


class SmoothedPinnballLoss(Metric):
    """Calculates an approximated pinball loss or quantile loss against the specified quantiles.
    To avoid discontinous gradients it has been modified to return a quandratic error instead when close to the correct value
     as described in https://ieeexplore.ieee.org/document/8832203

    Parameters
    -------
    quantiles: List[float], default = None
        List of values between 0 and 1, the quantiles that predicted by the model.
        Defaults to None in wich case the quantiles are set to [1-alpha, alpha].
        For information on the global default see 'Metrics.set_global_default(...)'
    eps: float, default = 1e-6
        Determines the size around the target value that yields a quadratic loss
    """

    def __init__(
        self,
        quantiles: List[float] = None,
        eps: float = 1.0e-6,
    ):
        if quantiles is None:
            quantiles = [1.0 - Metric.alpha / 2, Metric.alpha / 2, 0.5]
        super().__init__(quantiles=quantiles, eps=eps)
        self.input_labels = [f"quant[{quant}]" for quant in quantiles]

    def get_quantile_prediction(
        self, predictions: torch.tensor, quantiles: Optional[List[float]] = None, **_
    ) -> QuantilePrediction:
        if quantiles is None:
            return QuantilePrediction(predictions, self.options["quantiles"])
        else:
            return QuantilePrediction(
                predictions, self.options.get("quantiles")
            ).select_quantiles(quantiles, inplace=True)

    def from_quantiles(
        self,
        target: torch.tensor,
        quantile_prediction: QuantilePrediction,
        avg_over: Union[Literal["time"], Literal["sample"], Literal["all"]] = "all",
        **kwargs,
    ):
        """
        Calculates the value of the metric based on the values for each quantile over the timeframe.

        Parameters
        ----------
        target: torch.tensor
            Target values from the training or validation dataset. Dimensions have to be (sample number, timestep, 1).
        quantile_prediction: QuantilePrediction
            A prediction for several quantiles.
        Returns
        -------
        torch.tensor
            Value of the metric, which depending on the value of 'avg_over'
            is either a 0d-tensor (overall loss) or 1d-tensor over the horizon or the sample.
        """
        # if alpha is not None:
        #     quantiles = [1 - alpha, alpha]
        # else:
        #     quantiles = None
        return self(
            target,
            quantile_prediction.values,
            quantiles=quantile_prediction.quantiles,
            avg_over=avg_over,
        )

    @staticmethod
    def func(
        target: torch.tensor,
        predictions: torch.tensor,
        quantiles: List[float],
        eps: float = 1e-6,
        avg_over: Literal["all"] = "all",
        **_,
    ):
        """
        Calculates smoothed pinball loss or quantile loss against the specified quantiles

        Parameters
        ----------
        target : torch.tensor
            The true values of the target variable. Dimensions are (sample number, timestep, 1).
        predictions : torch.tensor
            The predicted values for each quantile. Dimensions are (sample number, timestep, quantile).
        quantiles : List[float]
            Quantiles that we are estimating for
        avg_over: str, default = "all"
            One of "time", "sample", "all", averages the the results over the coresponding axis.
        Returns
        -------
        float
            The total quantile loss (the lower the better)

        Raises
        ------
        NotImplementedError
            When 'avg_over' is set to anything but "all", "time" or "sample".
        """

        loss = SmoothedPinnballLoss._huber_metric(
            predictions=predictions, target=target, eps=eps
        )

        mask_greater = predictions >= target
        mask_lesser = predictions < target
        quantiles_tensor = torch.tensor([[quantiles]], device=predictions.device)

        loss[mask_greater] = ((1 - quantiles_tensor) * loss)[mask_greater]
        loss[mask_lesser] = (quantiles_tensor * loss)[mask_lesser]

        loss = torch.sum(loss, dim=-1)
        return avg_over_dims(loss, over_dims=avg_over)

    @staticmethod
    def _huber_metric(predictions, target, eps):
        errors = target - predictions
        mask_in = torch.abs(errors) <= eps
        mask_out = torch.abs(errors) > eps
        errors[mask_in] = (errors[mask_in] ** 2) / (2 * eps)
        errors[mask_out] = torch.abs(errors[mask_out]) - eps / 2
        return errors


class CRPSGauss(Metric):
    """Normalized CRPS (continuous ranked probability score) of observations x
    relative to normally distributed forecasts with mean, mu, and standard deviation, sig.
    CRPS(N(mu, sig^2); x)

    Parameters
    -------
    alpha : float, default = global default (0.05 if not otherwise specified)
        Predicted probability of violating the bound of the prediction interval. For information on the global default see 'Metrics.set_global_default(...)'

    """

    def __init__(self, alpha: float = None):
        if alpha is None:
            alpha = Metric.alpha
        super().__init__(alpha=alpha)
        self.input_labels = ["expected_value", "log_variance"]

    def get_quantile_prediction(
        self, predictions: torch.tensor, alpha=None, **_
    ) -> QuantilePrediction:
        """
        Calculates the values for each quantile for the metric.
        For metrics using the normal distribution this will correspond to the confidence interval.

        Parameters
        ----------
        predictions: torch.tensor
            Predicted values on the same dataset as target. Dimension have to be (sample number, timestep, label number).
        alpha : float, default = None
            Predicted probability of violating the bound of the prediction interval. If no alpha is specified the class instance alpha specified is used. To avoid confusion use of this parameter is discouraged.

        Returns
        -------
        QuantilePrediciton
            Prediciton for the specified quantiles.

        """
        if alpha is None:
            alpha = self.options.get("alpha")
        expected_values = predictions[..., 0]  # expected_values:mu
        sigma = torch.exp(predictions[..., 1] * 0.5)
        return QuantilePrediction.from_gauss_params(
            torch.stack((expected_values, sigma)), [1 - alpha / 2.0, alpha / 2.0, 0.5]
        )

    def from_quantiles(
        self,
        target: torch.tensor,
        quantile_prediction: QuantilePrediction,
        avg_over: Union[Literal["time"], Literal["sample"], Literal["all"]] = "all",
    ):
        """
        Calculates the value of the metric based on the values for each quantile over the timeframe.

        Parameters
        ----------
        target: torch.tensor
            Target values from the training or validation dataset. Dimensions have to be (sample number, timestep, 1).
        quantile_prediction: QuantilePrediction
            A prediction for several quantiles. Has to contain atleast the median prediction and one additional one to estimate the std. deviation.
            The mean is estimated to be the median as it would be for a gaussian distribution.

        Returns
        -------
        torch.tensor
            Value of the metric, which depending on the value of 'avg_over'
            is either a 0d-tensor (overall loss) or 1d-tensor over the horizon or the sample.
        """
        gauss_params = quantile_prediction.get_gauss_params()
        gauss_params[..., 1] = gauss_params[..., 1].log() * 2
        return self(target, gauss_params, avg_over=avg_over)

    @staticmethod
    def func(
        target: torch.tensor,
        predictions: torch.tensor,
        avg_over: Union[Literal["time"], Literal["sample"], Literal["all"]] = "all",
        **_,
    ):
        """
        Computes normalized CRPS (continuous ranked probability score) of observations x
        relative to normally distributed forecasts with mean, mu, and standard deviation, sig.
        CRPS(N(mu, sig^2); x)

        Code Source: https://github.com/TheClimateCorporation/properscoring/blob/master/properscoring/_crps.py
        Formula taken from Equation (5):
        Calibrated Probablistic Forecasting Using Ensemble Model Output
        Statistics and Minimum CRPS Estimation. Gneiting, Raftery,
        Westveld, Goldman. Monthly Weather Review 2004
        http://journals.ametsoc.org/doi/pdf/10.1175/MWR2904.1

        Parameters
        ----------
        target : torch.tensor
            The true values of the target variable, dimensions are (sample number, timestep, 1).
        predictions :  torch.tensor
            - predictions[...,0] = expected_value, a torch.tensor containing predicted expected values
            of the target variable
            - predictions[...,1] = log_variance, approx. equal to log(2*pi*sigma^2)
        avg_over: str, default = "all"
            Only "all" is supported, .

        Returns
        -------
        torch.tensor
            The gaussian negative log likelihood loss, which depending on the value of 'avg_over'
            is either a 0d-tensor (overall loss) or 1d-tensor over the horizon or the sample.

        Raises
        ------
        NotImplementedError
            When 'avg_over' is set to anything but "all", as crps_gaussian does not support loss over the horizon or sample.
        """

        assert predictions.size()[2] == 2
        mu = predictions[..., 0]
        log_variance = predictions[..., 1]
        if avg_over != "all":
            raise NotImplementedError(
                "crps_gaussian does not support loss over the horizon or per sample."
            )
        sig = torch.exp(log_variance * 0.5)
        norm_dist = torch.distributions.normal.Normal(0, 1)
        # standadized x
        sx = (target - mu) / sig
        pdf = torch.exp(norm_dist.log_prob(sx))
        cdf = norm_dist.cdf(sx)
        pi_inv = 1.0 / np.sqrt(np.pi)
        # the actual crps
        crps = sig * (sx * (2 * cdf - 1) + 2 * pdf - pi_inv)
        return torch.mean(crps, keepdim=True)


class Residuals(Metric):
    """Residual error including the sign."""

    def __init__(self):
        super().__init__()
        self.input_labels = ["expected_value"]

    def get_quantile_prediction(
        self,
        target: torch.tensor,
        predictions: torch.tensor,
        quantiles: Optional[List[float]] = None,
        **_,
    ) -> QuantilePrediction:
        """
        Calculates the values for each quantile for the metric.
        For metrics using the normal distribution this will correspond to the confidence interval.

        Parameters
        ----------
        target: torch.tensor
            Target values from the training or validation dataset. Dimensions have to be (sample number, timestep, 1).
        predictions: torch.tensor
            Predicted values on the same dataset as target. Dimension have to be (sample number, timestep, 1).
        alpha : float, default = None
            Predicted probability of violating the bound of the prediction interval. If no alpha is specified the class instance alpha specified is used. To avoid confusion use of this parameter is discouraged.

        Returns
        -------
        QuantilePrediciton
            Prediciton for the specified quantiles.

        """
        if quantiles is None:
            alpha_half = self.options.get("alpha") / 2.0
            quantiles = (1 - alpha_half, alpha_half, 0.5)
        rmse = Rmse.func(target, predictions, avg_over="all")
        sigma = torch.full_like(predictions[..., 0], rmse.item())
        return QuantilePrediction.from_gauss_params(
            torch.stack((predictions[..., 0], sigma), dim=-1), quantiles
        )

    def from_quantiles(
        self,
        target: torch.tensor,
        quantile_prediction: QuantilePrediction,
        avg_over: Union[Literal["time"], Literal["sample"], Literal["all"]] = "all",
        **_,
    ) -> torch.tensor:
        """
        Calculates the value of the metric based on the values for each quantile over the timeframe.

        Parameters
        ----------
        target: torch.tensor
            Target values from the training or validation dataset. Dimensions have to be (sample number, timestep, 1).
        quantile_prediction: QuantilePrediction
            A prediction for several quantiles. Has to contain atleast the median prediction.
            The mean is estimated to be the median as it would be for a gaussian distribution.
        avg_over: str
            One of "time", "sample", "all", averages the the results over the coresponding axis.

        Returns
        -------
        torch.tensor
            Value of the metric, which depending on the value of 'avg_over'
            is either a 0d-tensor (overall loss) or 1d-tensor over the horizon or the sample.
        """
        return self(
            target, quantile_prediction.get_gauss_params()[..., 0:1], avg_over=avg_over
        )

    @staticmethod
    def func(
        target: torch.tensor,
        predictions: torch.tensor,
        avg_over: Union[Literal["time"], Literal["sample"], Literal["all"]] = "all",
        **_,
    ):

        """
        Calculates the mean of the prediction error

        Parameters
        ----------
        target : torch.tensor
            The true values of the target variable, dimensions are (sample number, timestep, 1).
        predictions :  torch.tensor
            - predictions[...,0] = expected_value, a torch.tensor containing predicted expected values
            of the target variable
        avg_over: str, default = "all"
            One of "time", "sample", "all", averages the the results over the coresponding axis.

        Returns
        -------
        torch.tensor
            difference from the target value, which depending on the value of 'avg_over'
            is either a 0d-tensor (overall average) or 1d-tensor over the horizon or the sample.

        Raises
        ------
        ValueError
            When the dimensions of the predictions and target are not compatible
        """
        predictions = predictions[..., 0]
        if predictions.shape != target.shape:
            raise ValueError(
                "dimensions of predictions and target need to be compatible"
            )

        loss = target - predictions
        return avg_over_dims(loss, over_dims=avg_over)


class Mse(Metric):
    """Mean squared error

    Parameters
    ----------
    alpha: float, default = global default
        p-value of the quantile prediction generated by this metric.
        Not relevant outside of the quantile predicitons.
    """

    def __init__(self, alpha: float = None):
        if alpha is None:
            alpha = Metric.alpha
        super().__init__(alpha=alpha)
        self.input_labels = ["expected_value"]

    def get_quantile_prediction(
        self,
        target: torch.tensor,
        predictions: torch.tensor,
        quantiles: Optional[List[float]] = None,
        **_,
    ) -> QuantilePrediction:
        """
        Calculates the values for each quantile for the metric.
        For metrics using the normal distribution this will correspond to the confidence interval.

        Parameters
        ----------
        target: torch.tensor
            Target values from the training or validation dataset. Dimensions have to be (sample number, timestep, 1).
        predictions: torch.tensor
            Predicted values on the same dataset as target. Dimension have to be (sample number, timestep, 1).
        alpha : float, default = None
            Predicted probability of violating the bound of the prediction interval. If no alpha is specified the class instance alpha specified is used. To avoid confusion use of this parameter is discouraged.

        Returns
        -------
        QuantilePrediciton
            Prediciton for the specified quantiles.

        """
        if quantiles is None:
            alpha_half = self.options.get("alpha") / 2.0
            quantiles = (1 - alpha_half, alpha_half, 0.5)
        sigma = Rmse.func(target, predictions, avg_over=("time", "sample"))
        sigma = sigma.unsqueeze(dim=-1).expand_as(predictions)
        return QuantilePrediction.from_gauss_params(
            torch.cat((predictions, sigma), dim=-1), quantiles
        )

    def from_quantiles(
        self,
        target: torch.Tensor,
        quantile_prediction: QuantilePrediction,
        avg_over: Union[Literal["time"], Literal["sample"], Literal["all"]] = "all",
        **_,
    ) -> torch.tensor:
        """
        Calculates the value of the metric based on the values for each quantile over the timeframe.

        Parameters
        ----------
        target: torch.tensor
            Target values from the training or validation dataset. Dimensions have to be (sample number, timestep, 1).
        quantile_prediction: QuantilePrediction
            A prediction for several quantiles. Has to contain at least the median prediction.
            The mean is estimated to be the median as it would be for a gaussian distribution.
        avg_over: str
            One of "time", "sample", "all", averages the the results over the corresponding axis.
        alpha : float, default = None
            Predicted probability of violating the bound of the prediction interval.
            If no alpha is specified the class instance alpha specified is used. To avoid confusion use of this parameter is discouraged.

        Returns
        -------
        torch.tensor
            Value of the metric, which depending on the value of 'avg_over'
            is either a 0d-tensor (overall loss) or 1d-tensor over the horizon or the sample.
        """
        return self(
            target, quantile_prediction.get_gauss_params()[..., 0:1], avg_over=avg_over
        )

    @staticmethod
    def func(
        target: torch.tensor,
        predictions: torch.tensor,
        avg_over: Union[Literal["time"], Literal["sample"], Literal["all"]] = "all",
        **_,
    ):
        """
        Calculate the mean squared error (MSE)

        Parameters
        ----------
        target : torch.tensor
            The true values of the target variable, dimensions are (sample number, timestep, 1).
        predictions :  torch.tensor
            - predictions[...,0] = expected_value, a torch.tensor containing predicted expected values
            of the target variable. Dimensions are (sample number, timestep, 1).
        avg_over: str, default = "all"
            One of "time", "sample", "all", averages the the results over the coresponding axis.

        Returns
        -------
        torch.tensor
            The mean squared error, which depending on the value of 'avg_over'
            is either a 3d-tensor (overall loss) or 1d-tensor over the horizon or the sample.

        Raises
        ------
        ValueError
            When the dimensions of the predictions and target are not compatible
        """
        predictions = predictions[..., 0]
        if predictions.shape != target.shape:
            raise ValueError(
                f"dimensions of predictions {predictions.shape} and target {target.shape} need to be compatible"
            )

        squared_errors = (target - predictions) ** 2
        return avg_over_dims(squared_errors, over_dims=avg_over)


class Rmse(Metric):
    """Root mean squared error

    Parameters
    ----------
    alpha: float, default = global default
        p-value of the quantile prediction generated by this metric.
        Not relevant outside of the quantile predicitons.
    """

    def __init__(self, alpha=None):
        if alpha is None:
            alpha = Metric.alpha
        super().__init__(alpha=alpha)
        self.input_labels = ["expected_value"]

    def get_quantile_prediction(
        self,
        target: torch.tensor,
        predictions: torch.tensor,
        quantiles: Optional[List[float]] = None,
        **_,
    ) -> QuantilePrediction:
        """
        Calculates the values for each quantile for the metric.
        For metrics using the normal distribution this will correspond to the confidence interval.

        Parameters
        ----------
        target: torch.tensor
            Target values from the training or validation dataset. Dimensions have to be (sample number, timestep, 1).
        predictions: torch.tensor
            Predicted values on the same dataset as target. Dimension have to be (sample number, timestep, 1).
        alpha : float, default = None
            Predicted probability of violating the bound of the prediction interval. If no alpha is specified the class instance alpha specified is used. To avoid confusion use of this parameter is discouraged.

        Returns
        -------
        QuantilePrediciton
            Prediciton for the specified quantiles.

        """
        if quantiles is None:
            alpha_half = self.options.get("alpha") / 2.0
            quantiles = (1 - alpha_half, alpha_half, 0.5)
        rmse = Rmse.func(target, predictions, avg_over="all")
        sigma = torch.full_like(predictions[..., 0], rmse.item())
        return QuantilePrediction.from_gauss_params(
            torch.stack((predictions[..., 0], sigma), dim=-1), quantiles
        )

    def from_quantiles(
        self,
        target: torch.tensor,
        quantile_prediction: QuantilePrediction,
        avg_over: Union[Literal["time"], Literal["sample"], Literal["all"]] = "all",
        **_,
    ) -> torch.tensor:
        """
        Calculates the value of the metric based on the values for each quantile over the timeframe.

        Parameters
        ----------
        target: torch.tensor
            Target values from the training or validation dataset. Dimensions have to be (sample number, timestep, 1).
        quantile_prediction: QuantilePrediction
            A prediction for several quantiles. Has to contain atleast the median prediction.
            The mean is estimated to be the median as it would be for a gaussian distribution.
        avg_over: str
            One of "time", "sample", "all", averages the the results over the coresponding axis.
        alpha : float, default = None
            Predicted probability of violating the bound of the prediction interval.
            If no alpha is specified the class instance alpha specified is used. To avoid confusion use of this parameter is discouraged.

        Returns
        -------
        torch.tensor
            Value of the metric, which depending on the value of 'avg_over'
            is either a 0d-tensor (overall loss) or 1d-tensor over the horizon or the sample.
        """
        return self(
            target, quantile_prediction.get_gauss_params()[..., 0:1], avg_over=avg_over
        )

    @staticmethod
    def func(
        target: torch.Tensor,
        predictions: torch.Tensor,
        avg_over: List[Literal["time"], Literal["sample"], Literal["all"]] = "all",
        **_,
    ):
        """
        Calculate the root mean squared error

                Parameters
        ----------
        target : torch.tensor
            The true values of the target variable, dimensions are (sample number, timestep, 1).
        predictions :  torch.tensor
            - predictions[...,0] = expected_value, a torch.tensor containing predicted expected values
            of the target variable. Dimensions are (sample number, timestep, 1).
        avg_over: str, default = "all"
            One of "time", "sample", "all", averages the the results over the coresponding axis.

        Returns
        -------
        torch.tensor
            The mean squared error, which depending on the value of 'avg_over'
            is either a 0d-tensor (overall loss) or 1d-tensor over the horizon or the sample.

        Raises
        ------
        ValueError
            When the dimensions of the predictions and target are not compatible

        """
        predictions = predictions[..., 0]
        if predictions.shape != target.shape:
            raise ValueError(
                f"dimensions of predictions {predictions.shape} and target {target.shape} need to be compatible"
            )

        squared_errors = (target - predictions) ** 2
        return avg_over_dims(squared_errors, over_dims=avg_over).sqrt()


class Mase(Metric):
    """
    Calculate the mean absolute scaled error (MASE)

    (https://en.wikipedia.org/wiki/Mean_absolute_scaled_error)
    For more clarity, please refer to the following paper
    https://www.nuffield.ox.ac.uk/economics/Papers/2019/2019W01_M4_forecasts.pdf

    Parameters
    ----------
    freq : int scalar
        The frequency of the season type being considered
    insample_target : torch.tensor, default = None
        Contains insample values (e.g. target values shifted by season frequency). If none is provided defaults to shifting the target by 'freq'.
    """

    def __init__(self, freq: int = 1, insample_target=None):
        super().__init__(freq=freq, insample_target=insample_target)
        self.input_labels = ["expected_value"]

    def get_quantile_prediction(
        self,
        target: torch.tensor,
        predictions: torch.tensor,
        quantiles: Optional[List[float]] = None,
        **_,
    ) -> QuantilePrediction:
        """
        Calculates the values for each quantile for the metric.
        For metrics using the normal distribution this will correspond to the confidence interval.

        Parameters
        ----------
        target: torch.tensor
            Target values from the training or validation dataset. Dimensions have to be (sample number, timestep, 1).
        predictions: torch.tensor
            Predicted values on the same dataset as target. Dimension have to be (sample number, timestep, 1).
        quantiles : List[float], default = None
            quantiles to be estimated from the original prediction. Defaults to (1-alpha/2, alpha/2, 0.5)

        Returns
        -------
        QuantilePrediciton
            Prediciton for the specified quantiles.

        """
        if quantiles is None:
            alpha_half = self.options.get("alpha") / 2.0
            quantiles = (1 - alpha_half, alpha_half, 0.5)
        rmse = Rmse.func(target, predictions, avg_over="all")
        sigma = torch.full_like(predictions[..., 0], rmse.item())
        return QuantilePrediction.from_gauss_params(
            torch.stack((predictions[..., 0], sigma), dim=-1), quantiles
        )

    def from_quantiles(
        self,
        target: torch.tensor,
        quantile_prediction: QuantilePrediction,
        avg_over: Union[Literal["time"], Literal["sample"], Literal["all"]] = "all",
        freq=None,
        **_,
    ) -> torch.tensor:
        """
        Calculates the value of the metric based on the values for each quantile over the timeframe.

        Parameters
        ----------
        target: torch.tensor
            Target values from the training or validation dataset. Dimensions have to be (sample number, timestep, 1).
        quantile_prediction: QuantilePrediction
            A prediction for several quantiles. Has to contain atleast the median prediction.
            The mean is estimated to be the median as it would be for a gaussian distribution.
        avg_over: str
            One of "time", "sample", "all", averages the the results over the coresponding axis.

        Returns
        -------
        torch.tensor
            Value of the metric, which depending on the value of 'avg_over'
            is either a 0d-tensor (overall loss) or 1d-tensor over the horizon or the sample.
        """
        if freq is None:
            freq = self.options.get("freq", 1)
        return self(
            target,
            quantile_prediction.get_gauss_params()[..., 0:1],
            avg_over=avg_over,
            freq=freq,
        )

    @staticmethod
    def func(
        target: torch.tensor,
        predictions: torch.tensor,
        freq=1,
        avg_over: Union[Literal["time"], Literal["sample"], Literal["all"]] = "all",
        insample_target=None,
        **_,
    ):
        """
        Calculate the mean absolute scaled error (MASE)

        (https://en.wikipedia.org/wiki/Mean_absolute_scaled_error)
        For more clarity, please refer to the following paper
        https://www.nuffield.ox.ac.uk/economics/Papers/2019/2019W01_M4_forecasts.pdf


        Parameters
        ----------
        target : torch.tensor
            The true values of the target variable
        predictions :  torch.tensor
            predictions[...,0] = y_hat_test, predicted expected values of the target variable (torch.tensor).
            Dimensions are (sample number, timestep, 1).
        freq : int
            The frequency of the season type being considered
        avg_over: str, default = "all"
            Only "all" is supported by Mase, averages the results over the coresponding axis.
        insample_target : torch.tensor, default = None
            Contains insample values (e.g. target values shifted by season frequency). If none is provided defaults to shifting the target by 'freq'.

        Returns
        -------
        torch.tensor
            A scalar with the overall MASE (lower the better)

        Raises
        ------
        NotImplementedError
            When 'avg_over' is set to anything but "all", as MASE does not support loss over the horizon or sample.
        """
        if avg_over != "all":
            raise NotImplementedError(
                "mase does not support loss over the horizon or per sample."
            )

        y_hat_test = predictions[..., 0]
        if insample_target is None:
            y_hat_naive = torch.roll(
                target, freq, 1
            )  # shift all values by frequency, so that at time t,
        # y_hat_naive returns the value of insample [t-freq], as the first values are 0-freq = negative,
        # all values at the beginning are filled with values of the end of the tensor. So to not falsify the evaluation,
        # exclude all terms before freq
        else:
            y_hat_naive = insample_target
        masep = torch.mean(torch.abs(target[:, freq:] - y_hat_naive[:, freq:]))
        # denominator is the mean absolute error of the "seasonal naive forecast method"

        return (
            avg_over_dims(
                torch.abs(target[:, freq:] - y_hat_test[:, freq:]), over_dims=avg_over
            )
            / masep
        )


class Sharpness(Metric):
    """
    Calculate the mean size of the intervals, called the sharpness (lower the better)

    Parameters
    ----------
    quantiles: Iterable[float], default = None
        Quantiles used in `get_quantile_prediciton` if no other specified.
        If None is provided it defaults to a confidence interval
        based on the global setting for alpha.
    """

    def __init__(self, quantiles: Iterable[float] = None):
        if quantiles is None:
            quantiles = (1 - Metric.alpha / 2, Metric.alpha / 2)
        super().__init__(quantiles=quantiles)
        self.input_labels = ["upper_limit", "lower_limit"]

    def get_quantile_prediction(
        self,
        target: torch.tensor,
        predictions: torch.tensor,
        quantiles: Optional[List[float]] = None,
        **_,
    ) -> QuantilePrediction:
        """
        Calculates the values for each quantile for the metric.
        For metrics using the normal distribution this will correspond to the confidence interval.

        Parameters
        ----------
        target: torch.tensor
            Target values from the training or validation dataset. Dimensions have to be (sample number, timestep, 1).
        predictions: torch.tensor
            Predicted values on the same dataset as target. Dimension have to be (sample number, timestep, 1).
        quantiles : List[float], default = None
            quantiles to be estimated from the original prediction. Defaults to (1-alpha/2, alpha/2, 0.5)

        Returns
        -------
        QuantilePrediciton
            Prediciton for the specified quantiles.
        """
        if quantiles is None:
            quantiles = self.options.get("quantiles")
        rmse = Rmse.func(target, predictions, avg_over="all")
        sigma = torch.full_like(predictions[..., 0], rmse.item())
        return QuantilePrediction.from_gauss_params(
            torch.stack((predictions[..., 0], sigma), dim=-1), quantiles
        )

    def from_quantiles(
        self,
        target: torch.tensor,
        quantile_prediction: QuantilePrediction,
        avg_over: Union[Literal["time"], Literal["sample"], Literal["all"]] = "all",
        **_,
    ) -> torch.tensor:
        """
        Calculates the value of the metric based on the values for each quantile over the timeframe.

        Parameters
        ----------
        target: torch.tensor
            Target values from the training or validation dataset. Dimensions have to be (sample number, timestep, 1).
        quantile_prediction: QuantilePrediction
            A prediction for several quantiles. Has to contain atleast 2 quantile predictions, if more are provided highest and lowest quantile are used.
        avg_over: str
            One of "time", "sample", "all", averages the the results over the coresponding axis.

        Returns
        -------
        torch.tensor
            Value of the metric, which depending on the value of 'avg_over'
            is either a 0d-tensor (overall loss) or 1d-tensor over the horizon or the sample.
        """
        # TODO constraints on the quantile prediction?
        idx_lower = np.argmin(quantile_prediction.quantiles)
        idx_upper = np.argmax(quantile_prediction.quantiles)

        return self(
            target,
            quantile_prediction.values[..., (idx_upper, idx_lower)],
            avg_over=avg_over,
        )

    @staticmethod
    def func(
        # target: torch.tensor,
        predictions: torch.tensor,
        avg_over: Union[Literal["time"], Literal["sample"], Literal["all"]] = "all",
        **_,
    ):
        """
        Calculate the mean size of the intervals, called the sharpness (lower the better)

        Parameters
        ----------
        predictions :  torch.tensor
            - predictions[...,0] = y_pred_upper, predicted upper limit of the target variable
            - predictions[...,1] = y_pred_lower, predicted lower limit of the target variable
        avg_over: str, default = "all"
            One of "time", "sample", "all", averages the the results over the coresponding axis.

        Returns
        -------
        torch.tensor
            The shaprness, which depending on the value of 'total' is either a scalar (overall sharpness)
            or 1d-array over the horizon, in which case it is expected to increase as we move
            along the horizon. Generally, lower is better.

        """
        # XXX Seems weird to me that quantiles do not contribute here, since obviously a [0.05,0.95] interval is wider than [0.2,0.8]
        assert predictions.size()[-1] == 2
        y_pred_upper = predictions[..., 0]
        y_pred_lower = predictions[..., 1]
        return avg_over_dims(y_pred_upper - y_pred_lower, over_dims=avg_over)
        # if avg_over == "all":
        #     return torch.mean(y_pred_upper - y_pred_lower)
        # elif avg_over == "sample":
        #     return torch.mean(y_pred_upper - y_pred_lower, dim=0)
        # elif avg_over == "time":
        #     return torch.mean(y_pred_upper - y_pred_lower, dim=1)


class Picp(Metric):
    """Calculate PICP (prediction interval coverage probability) or simply the % of true
    values in the predicted intervals. Higher is generaly better.
    """

    def __init__(self):
        super().__init__()
        self.input_labels = ["upper_limit", "lower_limit"]

    def from_quantiles(
        self,
        target: torch.tensor,
        quantile_prediction: QuantilePrediction,
        avg_over: Union[Literal["sample"], Literal["all"]] = "all",
        **_,
    ):
        """
        Calculates the value of the metric based on the values for each quantile over the timeframe.

        Parameters
        ----------
        target: torch.tensor
            Target values from the training or validation dataset. Dimensions have to be (sample number, timestep, 1).
        quantile_prediction: QuantilePrediction
            A prediction for several quantiles. Has to contain atleast 2 quantile predictions, if more are provided highest and lowest quantile are used.
        avg_over: str
            One of "time", "sample", "all", averages the the results over the coresponding axis.

        Returns
        -------
        torch.tensor
            Value of the metric, which depending on the value of 'avg_over'
            is either a 0d-tensor (overall loss) or 1d-tensor over the horizon or the sample.
        """
        idx_lower = np.argmin(quantile_prediction.quantiles)
        idx_upper = np.argmax(quantile_prediction.quantiles)

        return self(
            target,
            quantile_prediction.values[..., (idx_upper, idx_lower)],
            avg_over=avg_over,
        )

    @staticmethod
    def func(
        target: torch.tensor,
        predictions: torch.tensor,
        avg_over: Union[Literal["sample"], Literal["all"]] = "all",
        **_,
    ):
        """
        Calculate PICP (prediction interval coverage probability) or simply the % of true
        values in the predicted intervals. Higher is generaly better.

        Parameters
        ----------
        target : torch.tensor
            true values of the target variable
        predictions :  List[torch.tensor]
            - predictions[...,0] = y_pred_upper, predicted upper limit of the target variable (torch.tensor)
            - predictions[...,1] = y_pred_lower, predicted lower limit of the target variable (torch.tensor)
        avg_over: str, default = "all"
            One of "time", "sample", "all", averages the the results over the coresponding axis.

        Returns
        -------
        torch.tensor
            The PICP, which depending on the value of 'avg_over' is either a scalar (PICP in %, for
            significance level alpha = 0.05, PICP should >= 95%)
            or 1d-array over the horizon or per sample.
        """

        assert predictions.size()[-1] == 2

        y_pred_upper = predictions[..., 0]
        y_pred_lower = predictions[..., 1]

        in_interval = ((target > y_pred_lower) & (target <= y_pred_upper)).float()

        return avg_over_dims(in_interval, over_dims=avg_over)


class PicpLoss(Picp):
    @staticmethod
    def func(
        target: torch.tensor,
        predictions: torch.tensor,
        avg_over: Union[Literal["sample"], Literal["all"]] = "all",
        **_,
    ):
        return 1 - Picp.func(target, predictions, avg_over)


class Mis(Metric):
    """
    Calculate MIS (mean interval score) without scaling by seasonal difference

    This metric combines both the sharpness and PICP metrics into a scalar value
    For more,please refer to https://www.m4.unic.ac.cy/wp-content/uploads/2018/03/M4-Competitors-Guide.pdf

    Parameters
    ----------
    alpha : float
        The significance level for the prediction interval
    """

    def __init__(self, alpha=None):
        if alpha is None:
            alpha = Metric.alpha
        super().__init__(alpha=alpha)
        self.input_labels = ["upper_limit", "lower_limit"]

    def from_quantiles(
        self,
        target: torch.tensor,
        quantile_prediction: QuantilePrediction,
        avg_over: Union[Literal["time"], Literal["sample"], Literal["all"]] = "all",
        **_,
    ):
        """
        Calculates the value of the metric based on the values for each quantile over the timeframe.

        Parameters
        ----------
        target: torch.tensor
            Target values from the training or validation dataset. Dimensions have to be (sample number, timestep, 1).

        quantile_prediction: QuantilePrediction
            A prediction for several quantiles. Has to contain atleast 2 quantile predictions of symetric quantiles (e.g (0.95,0.05)).
            If more are provided greatest symetric interval is used.
        avg_over: str
            One of "time", "sample", "all", averages the the results over the coresponding axis.

        Returns
        -------
        torch.tensor
            Value of the metric, which depending on the value of 'avg_over'
            is either a 0d-tensor (overall loss) or 1d-tensor over the horizon or the sample.
        """
        alpha = None
        for quant in quantile_prediction.quantiles:
            if 1 - quant in quantile_prediction.quantiles:
                alpha = min(quant, 1 - quant) * 2
                # break
        if not alpha:
            raise ValueError(
                "Mean Interval Score is only available for symetric intervals"
            )

        return self(
            target,
            quantile_prediction.select_quantiles((1 - alpha / 2, alpha / 2)).values,
            alpha=alpha,
            avg_over=avg_over,
        )

    @staticmethod
    def func(
        target: torch.tensor,
        predictions: torch.tensor,
        alpha: float = None,
        avg_over: Union[Literal["time"], Literal["sample"], Literal["all"]] = "all",
        **_,
    ):
        """
        Calculate MIS (mean interval score) without scaling by seasonal difference

        This metric combines both the sharpness and PICP metrics into a scalar value
        For more,please refer to https://www.m4.unic.ac.cy/wp-content/uploads/2018/03/M4-Competitors-Guide.pdf

        Parameters
        ----------
        target : torch.tensor
            true values of the target variable
        predictions :  torch.tensor
            - predictions[...,0] = y_pred_upper, predicted upper limit of the target variable
            - predictions[...,1] = y_pred_lower, predicted lower limit of the target variable
        alpha : float
            The significance level for the prediction interval
        avg_over: str, default = "all"
            One of "time", "sample", "all", averages the the results over the coresponding axis.

        Returns
        -------
        torch.tensor
            The MIS, which depending on the value of 'total' is either a scalar (overall MIS)
            or 1d-array over the horizon, in which case it is expected to increase as we move
            along the horizon. Generally, lower is better.

        """
        if alpha is None:
            alpha = Metric.alpha
        assert predictions.size()[-1] == 2
        y_pred_upper = predictions[..., 0]
        y_pred_lower = predictions[..., 1]

        # calculate penalty for large prediction interval
        diff = torch.abs(y_pred_upper - y_pred_lower)

        # calculate under estimation penalty
        diff_lower = y_pred_lower - target
        diff_lower[diff_lower < 0] = 0
        diff_lower = diff_lower * (2 / alpha)

        # calculate over estimation penalty
        diff_upper = target - y_pred_upper
        diff_upper[diff_upper < 0] = 0
        diff_upper = diff_upper * (2 / alpha)

        # combine all the penalties
        return avg_over_dims(
            diff + diff_lower + diff_upper,
            over_dims=avg_over,
        )


class Rae(Metric):
    """
    Calculate the RAE (Relative Absolute Error) compared to a naive forecast that only
    assumes that the future will produce the average of the past observations. Lower is better.
    """

    def __init__(self):
        super().__init__()
        self.input_labels = ["expected_value"]

    def from_quantiles(
        self,
        target: torch.tensor,
        quantile_prediction: QuantilePrediction,
        avg_over: Union[Literal["time"], Literal["sample"], Literal["all"]] = "all",
        **_,
    ):
        """
        Calculates the value of the metric based on the values for each quantile over the timeframe.

        Parameters
        ----------
        target: torch.tensor
            Target values from the training or validation dataset. Dimensions have to be (sample number, timestep, 1).
        quantile_prediction: QuantilePrediction
            A prediction for several quantiles. Has to contain atleast the median prediction.
            The mean is estimated to be the median as it would be for a gaussian distribution.
        avg_over: str
            One of "time", "sample", "all", averages the the results over the coresponding axis.
        alpha : float, default = None
            Predicted probability of violating the bound of the prediction interval.
            If no alpha is specified the class instance alpha specified is used. To avoid confusion use of this parameter is discouraged.

        Returns
        -------
        torch.tensor
            Value of the metric, which depending on the value of 'avg_over'
            is either a 0d-tensor (overall loss) or 1d-tensor over the horizon or the sample.
        """
        return self(
            target, quantile_prediction.get_gauss_params()[..., 0:1], avg_over=avg_over
        )

    @staticmethod
    def func(
        target: torch.tensor,
        predictions: torch.tensor,
        avg_over: Union[Literal["time"], Literal["sample"], Literal["all"]] = "all",
        **_,
    ):
        """
        Calculate the RAE (Relative Absolute Error) compared to a naive forecast that only
        assumes that the future will produce the average of the past observations. Lower is better.

        Parameters
        ----------
        target : torch.tensor
            The true values of the target variable
        predictions :  torch.tensor
            Predicted values over samples and time. Dimension have to be (sample number, timestep,features,1).
        avg_over: str, default = "all"
            Only "all" is supported, averages the the results over the coresponding axis.


        Returns
        -------
        torch.tensor
            A 0d-Tensor with the overall RAE.

        Raises
        ------
        NotImplementedError
            When 'avg_over' is set to anything but "all", "time" or "sample"
        """

        y_hat_test = predictions[..., 0]
        y_hat_naive = torch.mean(target)
        res_test = torch.abs(target - y_hat_test)
        res_naive = torch.abs(target - y_hat_naive)
        # denominator is the mean absolute error of the periodicity dependent "naive forecast method"
        # on the test set -->outsample
        return avg_over_dims(res_test, over_dims=avg_over) / avg_over_dims(
            res_naive, over_dims=avg_over
        )


class Mae(Metric):
    """Calculates mean absolute error
    MAE is different from MAPE in that the average of mean error is normalized over the average of all the actual values
    """

    def __init__(self):
        super().__init__()
        self.input_labels = ["expected_value"]

    def from_quantiles(
        self,
        target: torch.tensor,
        quantile_prediction: QuantilePrediction,
        avg_over: Union[Literal["time"], Literal["sample"], Literal["all"]] = "all",
        **_,
    ):
        """
        Calculates the value of the metric based on the values for each quantile over the timeframe.

        Parameters
        ----------
        target: torch.tensor
            Target values from the training or validation dataset. Dimensions have to be (sample number, timestep, 1).
         quantile_prediction: QuantilePrediction
            A prediction for several quantiles. Has to contain atleast the median prediction.
            The mean is estimated to be the median as it would be for a gaussian distribution.
        avg_over: str
            One of "time", "sample", "all", averages the the results over the coresponding axis.

        Returns
        -------
        torch.tensor
            Value of the metric, which depending on the value of 'avg_over'
            is either a 0d-tensor (overall loss) or 1d-tensor over the horizon or the sample.
        """
        return self(
            target, quantile_prediction.get_gauss_params()[..., 0:1], avg_over=avg_over
        )

    @staticmethod
    def func(
        target: torch.tensor,
        predictions: torch.tensor,
        avg_over: Union[Literal["time"], Literal["sample"], Literal["all"]] = "all",
        **_,
    ):
        """
        Calculates mean absolute error
        MAE is different from MAPE in that the average of mean error is normalized over the average of all the actual values

        Parameters
        ----------
        target : torch.tensor
            true values of the target variable
        predictions :  torch.tensor
            Predicted values over samples and time. Dimension have to be (sample number, timestep,1).
        avg_over: str, default = "all"
            Averages the results over the coresponding axis.

        Returns
        -------
        torch.tensor
            A scalar with the overall mae (the lower the better)

        """

        return avg_over_dims(
            torch.abs(target - predictions[..., 0]), over_dims=avg_over
        )


_EXCLUDED = ["Metric", "QuantilePrediction"]
# dict {class_name:class}
_all_dict = {
    cls[0].lower(): cls[1]
    for cls in inspect.getmembers(sys.modules[__name__], inspect.isclass)
    if (cls[1].__module__ == __name__ and cls[0] not in _EXCLUDED)
}


def get_metric(metric_name: str, **options) -> Metric:
    cls = _all_dict[metric_name.lower()]
    return cls(**options)


# TODO remove after baselines is up to date.
# def results_table(models: Union[None, List[str]], results, save_to_disc: bool = False):
#     """
#     Put the models' scores for the given metrics in a DataFrame.

#     Parameters
#     ----------
#     TODO: update only for the case that results includes the metrics
#     models : List[str] or None
#         The names of the models to use as index e.g. "gc17ct_GRU_gnll_test_hp"
#     mse : ndarray
#         The value(s) for mean squared error
#     rmse : ndarray
#         The value(s) for root mean squared error
#     mase : ndarray
#         The value(s) for mean absolute squared error
#     rae : ndarray
#         The value(s) for relative absolute error
#     mae : ndarray
#         The value(s) for mean absolute error
#     sharpness : ndarray
#         The value(s) for sharpness
#     coverage : ndarray
#         The value(s) for PICP (prediction interval coverage probability or % of true
#         values in the predicted intervals)
#     mis : ndarray
#         The value(s) for mean interval score
#     quantile_score : ndarray
#         The value(s) for quantile score
#     save_to_disc : string, default = False
#         If not False, save the scores as a csv, to the path specified in the string

#     Returns
#     -------
#     pandas.DataFrame
#         A DataFrame containing the models' scores for the given metrics
#     """
#     results.index = [models]
#     if save_to_disc:
#         save_path = save_to_disc + models.replace("/", "_")
#         results.to_csv(save_path + ".csv", sep=";", index=True)

#     return results
