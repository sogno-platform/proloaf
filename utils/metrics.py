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
import sys
import numpy as np
import torch
from abc import ABC, abstractstaticmethod
from typing import List, Union
import inspect


class Metric(ABC):
    """
    Stores loss functions and how many parameters they have.

    Parameters
    ----------
    func : callable
        A callable loss function
    num_params : int
        Number of parameters
    """

    def __init__(self, **options):
        self.options: dict = options
        self.input_labels: List[str]
        self.id: str = self.__class__

    def __call__(
        self,
        target: torch.Tensor,
        predictions: List[torch.Tensor],
        total: bool = False,
    ):
        return self.func(
            target=target, predictions=predictions, total=total ** self.options
        )

    # @abstractmethod
    def get_prediction_intervall(self, predictions: List[torch.Tensor], **kwargs):
        raise NotImplementedError(
            f"get_prediciton is not available for {self.__class__}"
        )

    @abstractstaticmethod
    def func(
        target: torch.Tensor, predictions: List[torch.Tensor], total: bool, **kwargs
    ) -> torch.Tensor:
        pass


class NllGauss(Metric):
    def __init__(self):
        super().__init__()
        self.input_labels = ["expected_value", "log_variance"]

    def get_prediction_intervall(self, predictions: List[torch.Tensor], alpha: float):
        expected_values = predictions[:, :, 0:1]  # expected_values:mu
        sigma = torch.sqrt(predictions[:, :, -1:].exp())
        # TODO: make 95% prediction interval changeable
        y_pred_upper = expected_values + 1.96 * sigma
        y_pred_lower = expected_values - 1.96 * sigma
        return y_pred_upper, y_pred_lower, expected_values

    @staticmethod
    def func(target: torch.Tensor, predictions: List[torch.Tensor], total: bool = True):
        """
        Calculates gaussian negative log likelihood score

        Parameters
        ----------
        target : torch.Tensor
            The true values of the target variable
        predictions :  List[torch.Tensor]
            - predictions[0] = expected_value, a torch.Tensor containing predicted expected values
            of the target variable
            - predictions[1] = log_variance, approx. equal to log(2*pi*sigma^2)
        total : bool, default = True
            - When total is set to True, return overall gaussian negative log likelihood loss
            - When total is set to False, return gaussian negative log likelihood loss along the horizon

        Returns
        -------
        torch.Tensor
            The gaussian negative log likelihood loss, which depending on the value of 'total'
            is either a scalar (overall loss) or 1d-array over the horizon, in which case it is
            expected to increase as we move along the horizon. Generally, lower is better.
        """

        assert len(predictions) == 2
        expected_value = predictions[0]
        log_variance = predictions[1]

        # y, y_pred, var_pred must have the same shape
        assert (
            target.shape == expected_value.shape
        )  # target.shape = torch.Size([batchsize, horizon, # of target variables]) e.g.[64,40,1]
        assert target.shape == log_variance.shape

        squared_errors = (target - expected_value) ** 2
        if total:
            return torch.mean(
                squared_errors / (2 * log_variance.exp()) + 0.5 * log_variance
            )
        else:
            return torch.mean(
                squared_errors / (2 * log_variance.exp()) + 0.5 * log_variance, dim=0
            )


class PinnballLoss(Metric):
    def __init__(self, quantiles: List[float]):
        super().__init__(quantiles=quantiles)
        self.input_labels = ["expected_value"]

    @staticmethod
    def func(
        target: torch.Tensor,
        predictions: List[torch.Tensor],
        quantiles: List[float],
        total: bool = True,
    ):
        """
        Calculates pinball loss or quantile loss against the specified quantiles

        Parameters
        ----------
        target : torch.Tensor
            The true values of the target variable
        predictions : List[torch.Tensor]
            The predicted expected values of the target variable
        quantiles : List[float]
            Quantiles that we are estimating for
        total : bool, default = True
            Used in other loss functions to specify whether to return overall loss or loss over
            the horizon. Pinball_loss only supports the former.

        Returns
        -------
        float
            The total quantile loss (the lower the better)

        Raises
        ------
        NotImplementedError
            When 'total' is set to False, as pinball_loss does not support loss over the horizon
        """

        # assert (len(predictions) == (len(quantiles) + 1))
        # quantiles = options

        if not total:
            raise NotImplementedError(
                "Pinball_loss does not support loss over the horizon"
            )
        loss = 0.0

        # TODO doing this in a loop seems inefficient
        for i, quantile in enumerate(quantiles):
            assert 0 < quantile
            assert quantile < 1
            assert target.shape == predictions[i].shape
            errors = target - predictions[i]
            loss += torch.mean(torch.max(quantile * errors, (quantile - 1) * errors))

        return loss


class QuantileScore(Metric):
    def __init__(self, quantiles):
        super().__init__(quantiles=quantiles)
        self.input_labels = ["expected_value"]

    def get_prediction_intervall(self, predictions: List[torch.Tensor]):
        y_pred_lower = predictions[:, :, 0:1]
        y_pred_upper = predictions[:, :, 1:2]
        expected_values = predictions[:, :, -1:]
        return y_pred_upper, y_pred_lower, expected_values

    @staticmethod
    def func(
        target: torch.Tensor,
        predictions: List[torch.Tensor],
        quantiles: List[float],
        total: bool = True,
    ):
        """
        Build upon the pinball loss, using the MSE to adjust the mean.

        Parameters
        ----------
        target : torch.Tensor
            True values of the target variable
        predictions : List[torch.Tensor]
            The predicted expected values of the target variable
        quantiles : List[float]
            Quantiles that we are estimating for
        total : bool, default = True
            Used in other loss functions to specify whether to return overall loss or loss over
            the horizon. Quantile_score (as an extension of pinball_loss) only supports the former.

        Returns
        -------
        float
            The total pinball loss + the rmse loss
        """

        # the quantile score builds upon the pinball loss,
        # we use the MSE to adjust the mean. one could also use 0.5 as third quantile,
        # but further code adjustments would be necessary then
        loss1 = PinnballLoss.func(target, predictions, quantiles, total)
        # loss2 = pinball_loss(target, [predictions[2]], [0.5], total)
        loss2 = Rmse.func(target, [predictions[len(quantiles)]])

        return loss1 + loss2


class CRPSGauss(Metric):
    def __init__(self):
        super().__init__()
        self.input_labels = ["expected_value", "log_variance"]

    def get_prediction_intervall(self, predictions: List[torch.Tensor], alpha: float):
        expected_values = predictions[:, :, 0:1]  # expected_values:mu
        sigma = torch.sqrt(predictions[:, :, -1:].exp())
        # TODO: make 95% prediction interval changeable
        y_pred_upper = expected_values + 1.96 * sigma
        y_pred_lower = expected_values - 1.96 * sigma
        return y_pred_upper, y_pred_lower, expected_values

    @staticmethod
    def func(target: torch.Tensor, predictions: List[torch.Tensor], total: bool = True):
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
        target : scalar or torch.Tensor
            The observation or set of observations.
        predictions :  List[torch.Tensor]
            - predictions[0] = mu, the mean of the forecast normal distribution (scalar or torch.Tensor)
            - predictions[1] = log_variance, The standard deviation of the forecast distribution (scalar or torch.Tensor)
        total : bool, default = True
            Used in other loss functions to specify whether to return overall loss or loss over
            the horizon. This function only supports the former.

        Returns
        -------
        torch.Tensor
            A scalar with the overall CRPS score. (lower the better )

        Raises
        ------
        NotImplementedError
            When 'total' is set to False, as crps_gaussian does not support loss over the horizon
        """

        assert len(predictions) == 2
        mu = predictions[0]
        log_variance = predictions[1]

        if not total:
            raise NotImplementedError(
                "crps_gaussian does not support loss over the horizon"
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
        return torch.mean(crps)


class Residuals(Metric):
    def __init__(self):
        super().__init__()
        self.input_labels = ["expected_value"]

    @staticmethod
    def func(target: torch.Tensor, predictions: List[torch.Tensor], total: bool = True):
        """
        Calculates the mean of the prediction error

        Parameters
        ----------
        target : torch.Tensor
            The true values of the target variable
        predictions :  List[torch.Tensor]
            The predicted expected values of the target variable
        total : bool, default = True
            - When total is set to True, return the overall mean of the error
            - When total is set to False, return the mean of the error along the horizon

        Returns
        -------
        torch.Tensor
            The mean of the error, which depending on the value of 'total'
            is either a scalar (overall mean) or 1d-array over the horizon, in which case it is
            expected to increase as we move along the horizon. Generally, lower is better.

        Raises
        ------
        ValueError
            When the dimensions of the predictions and target are not compatible
        """

        if predictions[0].shape != target.shape:
            raise ValueError(
                "dimensions of predictions and target need to be compatible"
            )

        error = target - predictions[0]
        if total:
            return torch.mean(error)
        else:
            return torch.mean(error, dim=0)


class Mse(Metric):
    def __init__(self):
        super().__init__()
        self.input_labels = ["expected_value"]

    def get_prediction_intervall(self, predictions: List[torch.Tensor]):
        expected_values = predictions
        y_pred_lower = 0
        y_pred_upper = 1
        return y_pred_upper, y_pred_lower, expected_values

    @staticmethod
    def func(target: torch.Tensor, predictions: List[torch.Tensor], total: bool = True):
        """
        Calculate the mean squared error (MSE)

        Parameters
        ----------
        target : torch.Tensor
            The true values of the target variable
        predictions : torch.Tensor
            predicted expected values of the target variable
        total : bool, default = True
            - When total is set to True, return overall MSE
            - When total is set to False, return MSE along the horizon

        Returns
        -------
        torch.Tensor
            The MSE, which depending on the value of 'total' is either a scalar (overall loss)
            or 1d-array over the horizon, in which case it is expected to increase as we move
            along the horizon. Generally, lower is better.

        Raises
        ------
        ValueError
            When the dimensions of the predictions and target are not compatible
        """
        if predictions[0].shape != target.shape:
            raise ValueError(
                "dimensions of predictions and target need to be compatible"
            )

        squared_errors = (target - predictions[0]) ** 2
        # TODO: Implement multiple target MSE calculation
        # num_targets = [int(x) for x in target.shape][-1]
        # for i in range(num_targets):
        #    if predictions[i].shape != target[:,:,i].unsqueeze_(-1).shape:
        #        raise ValueError('dimensions of predictions and target need to be compatible')
        #    squared_errors = (target.unsqueeze_(-1) - predictions) ** 2

        if total:
            return torch.mean(squared_errors)
        else:
            return torch.mean(squared_errors, dim=0)


class Rmse(Metric):
    def __init__(self, **options):
        super().__init__(**options)
        self.input_labels = ["expected_value"]

    def get_prediction_intervall(
        self, predictions: List[torch.Tensor], target: torch.Tensor
    ):
        expected_values = predictions
        rmse = self(target, expected_values.unsqueeze(0))
        # In order to produce an interval covering roughly 95% of the error magnitudes,
        # the prediction interval is usually calculated using the model output ± 2 × RMSE.
        y_pred_lower = expected_values - 2 * rmse
        y_pred_upper = expected_values + 2 * rmse
        return y_pred_upper, y_pred_lower, expected_values

    @staticmethod
    def func(target: torch.Tensor, predictions: List[torch.Tensor], total: bool = True):
        """
        Calculate the root mean squared error

        Parameters
        ----------
        target : torch.Tensor
            true values of the target variable
        predictions : torch.Tensor
            predicted expected values of the target variable
        total : bool, default = True
            - When total is set to True, return overall rmse
            - When total is set to False, return rmse along the horizon

        Returns
        -------
        torch.Tensor
            The rmse, which depending on the value of 'total' is either a scalar (overall loss)
            or 1d-array over the horizon, in which case it is expected to increase as we move
            along the horizon. Generally, lower is better.

        Raises
        ------
        ValueError
            When the dimensions of the predictions and target are not compatible
        """

        if predictions[0].shape != target.shape:
            raise ValueError(
                "dimensions of predictions and target need to be compatible"
            )

        squared_errors = (target - predictions[0]) ** 2
        if total:
            return torch.mean(squared_errors).sqrt()
        else:
            return torch.mean(squared_errors, dim=0).sqrt()


class Mse(Metric):
    def __init__(self):
        super().__init__()
        self.input_labels = ["expected_value"]

    def get_prediction_intervall(self, predictions: List[torch.Tensor]):
        expected_values = predictions
        y_pred_lower = 0
        y_pred_upper = 1
        return y_pred_upper, y_pred_lower, expected_values

    @staticmethod
    def func(target: torch.Tensor, predictions: List[torch.Tensor], total: bool = True):
        """
        Calculate root mean absolute error (mean absolute percentage error in %)

        Parameters
        ----------
        target : torch.Tensor
            true values of the target variable
        predictions :  List[torch.Tensor]
            - predictions[0] = predicted expected values of the target variable (torch.Tensor)
        total : bool, default = True
            Used in other loss functions to specify whether to return overall loss or loss over
            the horizon. This function only supports the former.

        Returns
        -------
        torch.Tensor
            A scalar with the mean absolute percentage error in % (lower the better)

        Raises
        ------
        NotImplementedError
            When 'total' is set to False, as MAPE does not support loss over the horizon
        """

        if not total:
            raise NotImplementedError("MAPE does not support loss over the horizon")

        return torch.mean(torch.abs((target - predictions[0]) / target)) * 100


class Mase(Metric):
    def __init__(self, freq: int = 1, insample_target=None):
        super().__init__(freq=freq, insample_target=insample_target)
        self.input_labels = ["expected_value"]

    @staticmethod
    def func(
        target: torch.Tensor,
        predictions: List[torch.Tensor],
        freq=1,
        total: bool = True,
        insample_target=None,
    ):
        """
        Calculate the mean absolute scaled error (MASE)

        (https://en.wikipedia.org/wiki/Mean_absolute_scaled_error)
        For more clarity, please refer to the following paper
        https://www.nuffield.ox.ac.uk/economics/Papers/2019/2019W01_M4_forecasts.pdf


        Parameters
        ----------
        target : torch.Tensor
            The true values of the target variable
        predictions :  List[torch.Tensor]
            - predictions[0] = y_hat_test, predicted expected values of the target variable (torch.Tensor)
        freq : int scalar
            The frequency of the season type being considered
        total : bool, default = True
            Used in other loss functions to specify whether to return overall loss or loss over
            the horizon. This function only supports the former.
        insample_target : torch.Tensor, default = None
            Contains insample values (e.g. target values shifted by season frequency)

        Returns
        -------
        torch.Tensor
            A scalar with the overall MASE (lower the better)

        Raises
        ------
        NotImplementedError
            When 'total' is set to False, as MASE does not support loss over the horizon
        """

        if not total:
            raise NotImplementedError("mase does not support loss over the horizon")

        y_hat_test = predictions[0]
        if insample_target == None:
            y_hat_naive = torch.roll(
                target, freq, 0
            )  # shift all values by frequency, so that at time t,
        # y_hat_naive returns the value of insample [t-freq], as the first values are 0-freq = negative,
        # all values at the beginning are filled with values of the end of the tensor. So to not falsify the evaluation,
        # exclude all terms before freq
        else:
            y_hat_naive = insample_target

        masep = torch.mean(torch.abs(target[freq:] - y_hat_naive[freq:]))
        # denominator is the mean absolute error of the "seasonal naive forecast method"
        return torch.mean(torch.abs(target[freq:] - y_hat_test[freq:])) / masep


class Sharpness(Metric):
    def __init__(self):
        super().__init__()
        self.input_labels = ["upper_limit", "lower_limit"]

    @staticmethod
    def func(target: torch.Tensor, predictions: List[torch.Tensor], total: bool = True):
        """
        Calculate the mean size of the intervals, called the sharpness (lower the better)

        Parameters
        ----------
        predictions :  List[torch.Tensor]
            - predictions[0] = y_pred_upper, predicted upper limit of the target variable (torch.Tensor)
            - predictions[1] = y_pred_lower, predicted lower limit of the target variable (torch.Tensor)
        total : bool, default = True
            - When total is set to True, return overall sharpness
            - When total is set to False, return sharpness along the horizon

        Returns
        -------
        torch.Tensor
            The shaprness, which depending on the value of 'total' is either a scalar (overall sharpness)
            or 1d-array over the horizon, in which case it is expected to increase as we move
            along the horizon. Generally, lower is better.

        """

        assert len(predictions) == 2
        y_pred_upper = predictions[0]
        y_pred_lower = predictions[1]
        if total:
            return torch.mean(y_pred_upper - y_pred_lower)
        else:
            return torch.mean(y_pred_upper - y_pred_lower, dim=0)


class Picp(Metric):
    def __init__(self):
        super().__init__()
        self.input_labels = ["upper_limit", "lower_limit"]

    @staticmethod
    def func(target: torch.Tensor, predictions: List[torch.Tensor], total: bool = True):
        """
        Calculate PICP (prediction interval coverage probability) or simply the % of true
        values in the predicted intervals

        Parameters
        ----------
        target : torch.Tensor
            true values of the target variable
        predictions :  List[torch.Tensor]
            - predictions[0] = y_pred_upper, predicted upper limit of the target variable (torch.Tensor)
            - predictions[1] = y_pred_lower, predicted lower limit of the target variable (torch.Tensor)
        total : bool, default = True
            - When total is set to True, return overall PICP
            - When total is set to False, return PICP along the horizon

        Returns
        -------
        torch.Tensor
            The PICP, which depending on the value of 'total' is either a scalar (PICP in %, for
            significance level alpha = 0.05, PICP should >= 95%)
            or 1d-array over the horizon, in which case it is expected to decrease as we move
            along the horizon. Generally, higher is better.
        """

        # coverage_horizon = torch.zeros(target.shape[1], device= target.device,requires_grad=True)
        # for i in range(target.shape[1]):
        #     # for each step in forecast horizon, calcualte the % of true values in the predicted interval
        #     coverage_horizon[i] = (torch.sum((target[:, i] > y_pred_lower[:, i]) &
        #                             (target[:, i] <= y_pred_upper[:, i])) / target.shape[0]) * 100
        assert len(predictions) == 2
        # torch.set_printoptions(precision=5)
        y_pred_upper = predictions[0]
        y_pred_lower = predictions[1]
        coverage_horizon = (
            100.0
            * (torch.sum((target > y_pred_lower) & (target <= y_pred_upper), dim=0))
            / target.shape[0]
        )

        coverage_total = torch.sum(coverage_horizon) / target.shape[1]
        if total:
            return coverage_total
        else:
            return coverage_horizon


# TODO not very nice, if needed think about a different approach to the "reward" vs "loss" problem
# @metric_with_labels(["upper_limit", "lower_limit"])
# def picp_loss(target: torch.Tensor, predictions, total: bool =True):
#     """
#     Calculate 1 - PICP (see eval_metrics.picp for more details)

#     Parameters
#     ----------
#     target : torch.Tensor
#         The true values of the target variable
#     predictions :  List[torch.Tensor]
#         - predictions[0] = y_pred_upper, predicted upper limit of the target variable (torch.Tensor)
#         - predictions[1] = y_pred_lower, predicted lower limit of the target variable (torch.Tensor)
#     total : bool, default = True
#         - When total is set to True, return a scalar value for 1- PICP
#         - When total is set to False, return 1-PICP along the horizon

#     Returns
#     -------
#     torch.Tensor
#         Returns 1-PICP, either as a scalar or over the horizon
#     """
#     return 1 - picp(target: torch.Tensor, predictions, total)
class Mis(Metric):
    def __init__(self, alpha=0.05):
        super().__init__(alpha=alpha)
        self.input_labels = ["upper_limit", "lower_limit"]

    @staticmethod
    def func(
        target: torch.Tensor,
        predictions: List[torch.Tensor],
        alpha=0.05,
        total: bool = True,
    ):
        """
        Calculate MIS (mean interval score) without scaling by seasonal difference

        This metric combines both the sharpness and PICP metrics into a scalar value
        For more,please refer to https://www.m4.unic.ac.cy/wp-content/uploads/2018/03/M4-Competitors-Guide.pdf

        Parameters
        ----------
        target : torch.Tensor
            true values of the target variable
        predictions :  List[torch.Tensor]
            - predictions[0] = y_pred_upper, predicted upper limit of the target variable (torch.Tensor)
            - predictions[1] = y_pred_lower, predicted lower limit of the target variable (torch.Tensor)
        alpha : float
            The significance level for the prediction interval
        total : bool, default = True
            - When total is set to True, return overall MIS
            - When total is set to False, return MIS along the horizon

        Returns
        -------
        torch.Tensor
            The MIS, which depending on the value of 'total' is either a scalar (overall MIS)
            or 1d-array over the horizon, in which case it is expected to increase as we move
            along the horizon. Generally, lower is better.

        """

        assert len(predictions) == 2
        y_pred_upper = predictions[0]
        y_pred_lower = predictions[1]

        mis_horizon = torch.zeros(target.shape[1])

        for i in range(target.shape[1]):
            # calculate penalty for large prediction interval
            large_PI_penalty = torch.sum(y_pred_upper[:, i] - y_pred_lower[:, i])

            # calculate under estimation penalty
            diff_lower = y_pred_lower[:, i] - target[:, i]
            under_est_penalty = (2 / alpha) * torch.sum(diff_lower[diff_lower > 0])

            # calcualte over estimation penalty
            diff_upper = target[:, i] - y_pred_upper[:, i]
            over_est_penalty = (2 / alpha) * torch.sum(diff_upper[diff_upper > 0])

            # combine all the penalties
            mis_horizon[i] = (
                large_PI_penalty + under_est_penalty + over_est_penalty
            ) / target.shape[0]

        mis_total = torch.sum(mis_horizon) / target.shape[1]

        if total:
            return mis_total
        else:
            return mis_horizon


class Rae(Metric):
    def __init__(self):
        super().__init__()
        self.input_labels = ["expected_value"]

    @staticmethod
    def func(target: torch.Tensor, predictions: List[torch.Tensor], total: bool = True):
        """
        Calculate the RAE (Relative Absolute Error) compared to a naive forecast that only
        assumes that the future will produce the average of the past observations

        Parameters
        ----------
        target : torch.Tensor
            The true values of the target variable
        predictions :  List[torch.Tensor]
            - predictions[0] = y_hat_test, predicted expected values of the target variable (torch.Tensor)
        total : bool, default = True
            Used in other loss functions to specify whether to return overall loss or loss over
            the horizon. This function only supports the former.

        Returns
        -------
        torch.Tensor
            A scalar with the overall RAE (the lower the better)

        Raises
        ------
        NotImplementedError
            When 'total' is set to False, as rae does not support loss over the horizon
        """

        y_hat_test = predictions[0]
        y_hat_naive = torch.mean(target)

        if not total:
            raise NotImplementedError("rae does not support loss over the horizon")

        # denominator is the mean absolute error of the preidicity dependent "naive forecast method"
        # on the test set -->outsample
        return torch.mean(torch.abs(target - y_hat_test)) / torch.mean(
            torch.abs(target - y_hat_naive)
        )


class Mae(Metric):
    def __init__(self):
        super().__init__()
        self.input_labels = ["expected_value"]

    @staticmethod
    def func(target: torch.Tensor, predictions: List[torch.Tensor], total: bool = True):
        """
        Calculates mean absolute error
        MAE is different from MAPE in that the average of mean error is normalized over the average of all the actual values

        Parameters
        ----------
        target : torch.Tensor
            true values of the target variable
        predictions :  List[torch.Tensor]
            - predictions[0] = y_hat_test, predicted expected values of the target variable (torch.Tensor)
        total : bool, default = True
            Used in other loss functions to specify whether to return overall loss or loss over
            the horizon. This function only supports the former.

        Returns
        -------
        torch.Tensor
            A scalar with the overall mae (the lower the better)

        Raises
        ------
        NotImplementedError
            When 'total' is set to False, as mae does not support loss over the horizon
        """

        if not total:
            raise NotImplementedError("mae does not support loss over the horizon")

        y_hat_test = predictions[0]

        return torch.mean(torch.abs(target - y_hat_test))


# dict {class_name:class}
_all_dict = {
    cls[0].lower(): cls[1]
    for cls in inspect.getmembers(sys.modules[__name__], inspect.isclass)
    if cls[1].__module__ == __name__
}


def get_metric(metric_name: str, options: dict = {}) -> Metric:
    cls = _all_dict[metric_name.lower()]
    return cls(**options)


def results_table(models: Union[None, List[str]], results, save_to_disc: bool = False):
    """
    Put the models' scores for the given metrics in a DataFrame.

    Parameters
    ----------
    TODO: update only for the case that results includes the metrics
    models : List[str] or None
        The names of the models to use as index e.g. "gc17ct_GRU_gnll_test_hp"
    mse : ndarray
        The value(s) for mean squared error
    rmse : ndarray
        The value(s) for root mean squared error
    mase : ndarray
        The value(s) for mean absolute squared error
    rae : ndarray
        The value(s) for relative absolute error
    mae : ndarray
        The value(s) for mean absolute error
    sharpness : ndarray
        The value(s) for sharpness
    coverage : ndarray
        The value(s) for PICP (prediction interval coverage probability or % of true
        values in the predicted intervals)
    mis : ndarray
        The value(s) for mean interval score
    quantile_score : ndarray
        The value(s) for quantile score
    save_to_disc : string, default = False
        If not False, save the scores as a csv, to the path specified in the string

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the models' scores for the given metrics
    """
    results.index = [models]
    if save_to_disc:
        save_path = save_to_disc + models.replace("/", "_")
        results.to_csv(save_path + ".csv", sep=";", index=True)

    return results


# def fetch_metrics(
#     target,
#     expected_values,
#     y_pred_upper,
#     y_pred_lower,
#     analyzed_metrics=["mse"],
#     total=True,
# ):
#     # Note:
#     # if total = False, the metric is returned averaged over the test period per prediction time step
#     # else (if total = True), the metric is returned averaged over the test period and every prediction step
#     # (=forecast horizon)
#     # calculate the metrics
#     mse_result = mse(target, [expected_values], total=total).detach().numpy()
#     try:
#         results = pd.DataFrame(mse_result, columns=["mse"])
#     except:
#         results = pd.DataFrame([mse_result], columns=["mse"])
#     if "rmse" in analyzed_metrics:
#         rmse_result = rmse(target, [expected_values], total=total).detach().numpy()
#         results["rmse"] = rmse_result
#     if "sharpness" in analyzed_metrics:
#         sharpness_result = (
#             sharpness([y_pred_upper, y_pred_lower], total=total).detach().numpy()
#         )
#         results["sharpness"] = sharpness_result
#     if "picp" in analyzed_metrics:
#         coverage_result = (
#             picp(target, [y_pred_upper, y_pred_lower], total=total).detach().numpy()
#         )
#         results["picp"] = coverage_result
#     if "mis" in analyzed_metrics:
#         mis_result = (
#             mis(target, [y_pred_upper, y_pred_lower], total=total).detach().numpy()
#         )
#         results["mis"] = mis_result

#     if total:
#         # only add these metrics if total is true as the performance per time_step is not implemented yet
#         if "mase" in analyzed_metrics:
#             mase_result = (
#                 mase(target, [expected_values], freq=7 * 24, total=total)
#                 .detach()
#                 .numpy()
#             )
#             results["mase"] = mase_result
#         if "rae" in analyzed_metrics:
#             rae_result = rae(target, [expected_values], total=total).detach().numpy()
#             results["rae"] = rae_result
#         if "mae" in analyzed_metrics:
#             mae_result = mae(target, [expected_values], total=total).detach().numpy()
#             results["mae"] = mae_result
#         if "qs" in analyzed_metrics:
#             qs_result = (
#                 pinball_loss(
#                     target,
#                     [y_pred_upper, y_pred_lower],
#                     [0.025, 0.975],  # equals a 95% prediction interval
#                     total=total,
#                 )
#                 .detach()
#                 .numpy()
#             )
#             results["qs"] = qs_result
#         # TODO: implement this case in each of the metric functions
#     return results