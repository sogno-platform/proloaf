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
Provides structures for storing and loading data (e.g. training, validation or test data)
"""
import numpy as np
import pandas as pd
import torch
from typing import List, Union, Tuple, Callable, Iterable
import proloaf.datahandler as dh
from time import sleep
from torch.utils.data.dataloader import DataLoader
from proloaf.cli import create_event_logger

logger = create_event_logger(__name__)


class TimeSeriesData(torch.utils.data.Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        history_horizon: int = None,
        forecast_horizon: int = None,
        encoder_features: Iterable[str] = None,
        decoder_features: Iterable[str] = None,
        target_id: Union[str, Iterable[str]] = None,
        preparation_steps: Iterable[Callable[[pd.DataFrame], pd.DataFrame]] = None,
        transform: Callable = None,
        device: Union[int, str] = "cpu",
        **_,
    ):
        self.data = df

        self.encoder_features = encoder_features
        self.history_horizon = history_horizon
        self.decoder_features = decoder_features
        self.forecast_horizon = forecast_horizon
        self.target_id = target_id
        # self.transform = transform
        self.device = device
        self.tensor_prepared = False
        self.frame_prepared = False
        self.preparation_steps = preparation_steps
        # print(self.data[self.history_features])

        # print(f"{self.data[self.future_features].columns = }")
        # self.to_tensor()

    @property
    def encoder_features(self):
        return self._encoder_features.copy()

    @encoder_features.setter
    def encoder_features(self, val):
        try:
            if set(val) != set(self._encoder_features):
                self._encoder_features = val
                self.tensor_prepared = False
        except AttributeError:
            self._encoder_features = val
            self.tensor_prepared = False

    @property
    def decoder_features(self):
        return self._decoder_features.copy()

    @decoder_features.setter
    def decoder_features(self, val):
        try:
            if set(val) != set(self._decoder_features):
                self._decoder_features = val
                self.tensor_prepared = False
        except AttributeError:
            self._decoder_features = val
            self.tensor_prepared = False

    @property
    def target_id(self):
        return self._target_features.copy()

    @target_id.setter
    def target_id(self, val):
        target_features = (val,) if isinstance(val, str) else val
        try:
            if set(target_features) != set(self._target_features):
                self._target_features = target_features
                self.tensor_prepared = False
        except AttributeError:
            self._target_features = target_features
            self.tensor_prepared = False

    def __len__(self):
        # TODO add Warning when not enough data available
        return max(0, len(self.data) - self.history_horizon - self.forecast_horizon) + 1

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        if self._index >= len(self):
            raise StopIteration
        idx = self._index
        self._index += 1
        return self[idx]

    def add_preparation_step(self, step: Callable[[pd.DataFrame], pd.DataFrame]):
        self.preparation_steps.append(step)
        self.frame_prepared = False
        self.tensor_prepared = False
        return self

    def clear_preparation_steps(self):
        if self.frame_prepared:
            raise AttributeError(
                "The preparationsteps were already applied to the Dataframe and can not be removed anymore"
            )
        self.preparation_steps = None
        self.tensor_prepared = False
        return self

    @staticmethod
    def _apply_prep_to_frame(df, steps):
        if isinstance(steps, str):
            raise TypeError("Preparation steps can not be identified by strings.")
        elif isinstance(steps, Iterable):
            for step in steps:
                df = step(df)
        elif isinstance(steps, Callable):
            df = steps(df)
        else:
            raise TypeError(
                "preparation_steps should be a callable or an interable of callables."
            )
        return df

    def apply_prep_to_frame(self):
        if self.preparation_steps is None or self.frame_prepared:
            self.frame_prepared = True
            return self
        self.data = self._apply_prep_to_frame(self.data, self.preparation_steps)
        self.frame_prepared = True
        return self

    def get_as_frame(self, idx) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if not isinstance(idx, int):
            idx = self.data.get_loc(idx)
        hist = self.data.iloc[idx : idx + self.history_horizon]
        fut = self.data.iloc[
            idx
            + self.history_horizon : idx
            + self.history_horizon
            + self.forecast_horizon
        ]

        return (
            hist.filter(items=self.encoder_features, axis="columns"),
            fut.filter(items=self.decoder_features, axis="columns"),
            fut.filter(items=self.target_features, axis="columns"),
        )

    def __getitem__(self, idx):
        return (
            self.hist[idx : idx + self.history_horizon],
            self.fut[
                idx
                + self.history_horizon : idx
                + self.history_horizon
                + self.forecast_horizon
            ],
            self.target[
                idx
                + self.history_horizon : idx
                + self.history_horizon
                + self.forecast_horizon
            ],
        )

    def to(self, device: Union[str, int]):
        self.device = device
        if self.tensor_prepared:
            if self.hist is not None:
                self.hist.to(device)
            if self.fut is not None:
                self.fut.to(device)
            if self.target is not None:
                self.target.to(device)
        return self

    def make_data_loader(
        self,
        history_horizon: int = None,
        forecast_horizon: int = None,
        encoder_features: Iterable[str] = None,
        decoder_features: Iterable[str] = None,
        target_id: Union[str, Iterable[str]] = None,
        batch_size: int = None,
        shuffle: bool = True,
        drop_last: bool = True,
    ):
        if history_horizon is not None:
            self.history_horizon = history_horizon
        if forecast_horizon is not None:
            self.forecast_horizon = forecast_horizon
        if encoder_features is not None:
            self.encoder_features = encoder_features
        if decoder_features is not None:
            self.decoder_features = decoder_features
        if target_id is not None:
            self.target_id = target_id

        if batch_size is None:
            batch_size = len(self)

        self.to_tensor()

        return TensorDataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
        )

    # def to(self, device: Union[str, int]):
    #     self._device = device

    def to_tensor(
        self,
        # encoder_features: List[Union[str, int]] = None,
        # decoder_features: List[Union[str, int]] = None,
        # target_id: Union[str, int] = None,
    ):
        if self.tensor_prepared:
            logger.info("tensor prepared")
            return self

        if not self.frame_prepared:
            logger.info("frame not prepared")
            df = self.data.copy()
            df = self._apply_prep_to_frame(self.data, self.preparation_steps)
        else:
            df = self.data

        self.hist = (
            torch.from_numpy(df[self._encoder_features].to_numpy())
            .float()
            .to(self.device)
        )
        self.fut = (
            torch.from_numpy(df[self._decoder_features].to_numpy())
            .float()
            .to(self.device)
        )
        self.target = (
            torch.from_numpy(df[self._target_features].to_numpy())
            .float()
            .to(self.device)
        )
        self.tensor_prepared = True
        return self


# class CustomTensorData:
#     """
#     A custom data structure that stores the inputs (for the Encoder and Decoder) and
#     target data as torch.Tensors.

#     Parameters
#     ----------
#     inputs1 : ndarray
#         Encoder inputs
#     inputs2 : ndarray
#         Decoder inputs
#     targets : ndarray
#         Targets (actual values)
#     """

#     def __init__(self, inputs1, inputs2, targets):
#         self.inputs1 = torch.Tensor(inputs1).float()
#         self.inputs2 = torch.Tensor(inputs2).float()
#         self.targets = torch.Tensor(targets).float()

#     def __getitem__(self, index):
#         return (
#             self.inputs1[index, :, :],
#             self.inputs2[index, :, :],
#             self.targets[index, :, :],
#         )

#     def __len__(self):
#         return self.targets.shape[0]

#     def to(self, device):
#         """
#         See torch.Tensor.to() - does dtype and/or device conversion
#         """

#         self.inputs1 = self.inputs1.to(device)
#         self.inputs2 = self.inputs2.to(device)
#         self.targets = self.targets.to(device)
#         return self


# class CustomTensorDataLoader:
#     """
#     An iterable data structure that stores CustomTensorData (optionally shuffled into random
#     permutations) in batches. Each iteration returns one batch.

#     Parameters
#     ----------
#     dataset : CustomTensorData
#         Input and target data stored in Tensors
#     batch_size : int, default = 1
#         The size of a batch. One training run operates on a single batch of data.
#     shuffle : bool, default = False
#         If True, shuffle the dataset into a random permutation
#     drop_last : bool, default = True
#         If True, sort the data set into as many batches of the specified size as possible.
#         Ignore any remaining data that don't make up a full batch.
#         If False, raise an error.

#     Raises
#     ------
#     NotImplementedError
#         Raised if drop_last is set to False.
#     StopIteration
#         Raised when the end of the iterator is reached.
#     """

#     def __init__(
#         self, dataset: CustomTensorData, batch_size=1, shuffle=False, drop_last=True
#     ):
#         if not drop_last:
#             raise NotImplementedError

#         self.dataset = dataset
#         self.batch_size = batch_size
#         self.shuffle = shuffle
#         self.drop_last = drop_last
#         n = self.batch_size * (len(self.dataset) // self.batch_size)
#         if self.shuffle:
#             permutation = np.random.permutation(len(self.dataset))[:n]
#         else:
#             permutation = np.arange(n)
#         self.permutation = np.reshape(permutation, (len(self), self.batch_size))

#     def __iter__(self):
#         n = self.batch_size * (len(self.dataset) // self.batch_size)
#         if self.shuffle:
#             permutation = np.random.permutation(len(self.dataset))[:n]
#         else:
#             permutation = np.arange(n)
#         self.permutation = np.reshape(permutation, (len(self), self.batch_size))
#         self.batch_index = 0
#         return self

#     def __next__(self):
#         if self.batch_index < len(self):
#             indices = self.permutation[self.batch_index]
#             self.batch_index += 1
#             return self.dataset[indices]
#         else:
#             raise StopIteration

#     def __getitem__(self, index):
#         indices = self.permutation[index]
#         return self.dataset[indices]

#     def get_sample(self, index):
#         batch_num = index // self.batch_size
#         item_num = index % self.batch_size
#         index = self.permutation[batch_num][item_num]
#         return self.dataset[index]

#     def __len__(self):
#         return len(self.dataset) // self.batch_size

#     def to(self, device: str):
#         """
#         See torch.Tensor.to() - does dtype and/or device conversion
#         """

#         self.dataset.to(device)
#         return self

#     def number_features1(self):
#         """
#         Return the number of features in the Encoder inputs

#         Returns
#         -------
#         int
#             The number of entries in the axis that represents the features of the Encoder inputs
#         """

#         return self.dataset.inputs1.shape[2]

#     def number_features2(self):
#         """
#         Return the number of features in the Decoder inputs

#         Returns
#         -------
#         int
#             The number of entries in the axis that represents the features of the Decoder inputs
#         """

#         return self.dataset.inputs2.shape[2]


class TensorDataLoader(DataLoader):
    def to(self, device):
        if hasattr(self.dataset, "to"):
            self.dataset.to(device)
        else:
            raise AttributeError("Dataset does not have an attribute 'to'")
        return self


def make_dataloader(
    df,
    target_id,
    encoder_features,
    decoder_features,
    history_horizon,
    forecast_horizon,
    batch_size=None,
    shuffle=True,
    drop_last=True,
    anchor_key=0,
    **_,
):
    """
    Store the given data in a CustomTensorDataLoader

    Parameters
    ----------
    df : pandas.DataFrame
        The data to be stored in a CustomTensorDataLoader
    target_id : string
        The name of feature to forecast, e.g. "demand"
    encoder_features : string list
        A list containing desired encoder feature names as strings
    decoder_features : string list
        A list containing desired decoder feature names as strings
    history_horizon : int
        The length of the history horizon in hours
    forecast_horizon : int
        The length of the forecast horizon in hours
    batch_size : int, default = 1
        The size of a batch. One training run operates on a single batch of data.
    shuffle : bool, default = True
        If True, shuffle the dataset into a random permutation
    drop_last : bool, default = True
        If True, sort the data set into as many batches of the specified size as possible.
        Ignore any remaining data that don't make up a full batch.
        If False, raise an error.

    Returns
    -------
    CustomTensorDataLoader
        An iterable data structure containing the provided input and target data
    """
    raise DeprecationWarning(
        "Proloaf has changed the way it is handling data. The Dataloader created by this function is deprecated."
    )
    # anchor_key='09:00:00'
    # rel_anchor_key='09:00:00'
    x_enc = dh.extract(
        df[encoder_features].iloc[:-forecast_horizon, :], history_horizon
    )
    # shape input data that is known for the Future, here take perfect hourly temp-forecast
    x_dec = dh.extract(df[decoder_features].iloc[history_horizon:, :], forecast_horizon)
    # shape y, depending on number of targets given
    y = dh.extract(df[target_id].iloc[history_horizon:, :], forecast_horizon)

    if len(x_enc) > len(y):
        x_enc = x_enc[:-1]
    elif len(x_enc) < len(y):
        x_dec = x_dec[1:]
        y = y[1:]
    custom_tensor_data = CustomTensorData(x_enc, x_dec, y)
    if batch_size is None:
        batch_size = len(custom_tensor_data)
    return CustomTensorDataLoader(
        custom_tensor_data, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
    )
