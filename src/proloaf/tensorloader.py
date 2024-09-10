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
from __future__ import annotations
from typing import Optional, Union, Tuple, Callable, Iterable
import pandas as pd
import torch
from torch.utils.data import Dataset, dataloader
from proloaf.event_logging import create_event_logger

logger = create_event_logger(__name__)


class TimeSeriesData(Dataset):
    """Contains timeseries data in pandas dataframe in addition to a Torch.Tensor based representation used in deep learning.

    Parameters
    ----------
    df : pandas.DataFrame
        Inital input data, this should already be split into training, validation and testing if needed.
    history_horizon: int (optional)
        Number of datapoints used as "past" in a prediction. If not set some methods will not be able to run. It can set through the `make_dataloader` method later on.
    forecast_horizon: int (optional)
        Number of datapoints used as "future" in a prediction. If not set some methods will not be able to run. It can set through the `make_dataloader` method later on.
    encoder_features: Iterable[str] (optional)
        List of the column names available to the encoder, that means it is data from the "past". This can be set later, but it has to be set before trying to access elements of a TimeSeriesData object.
    decoder_features: Iterable[str] (optional)
        List of the column names available to the encoder, that means it is data from the "future". This can be set later, but it has to be set before trying to access elements of a TimeSeriesData object.
    aux_features: Iterable[str] (optional),
        Auxillary features both encoder and decoder use, for documentation only. This can be set later, but it has to be set before trying to access elements of a TimeSeriesData object.
    target_id: Union[str, Iterable[str]] (optional)
        Column name(s) that is(are) regarded as the output in training. This can be set later, but it has to be set before trying to access elements of a TimeSeriesData object.
    preparation_steps: Iterable[Callable[[pd.DataFrame], pd.DataFrame]] (optional)
        Functions or callables that transform the dataframe in any way before it is fed to the model. Both input and output need to be a pandas.Dataframe.
    device: Union[int, str], default = "cpu"
        Device ony which the data should be loaded, can be changed later.

    Attributes
    ----------
    self.data: pandas.DataFrame
        Timeseries data in a single DataFrame
    self.encoder_features: Iterable[str]
        See Parameters
    self.history_horizon: Iterable[str]
        See Parameters
    self.decoder_features
        See Parameters
    self.forecast_horizon
        See Parameters
    self.aux_features
        See Parameters
    self.target_id
        See Parameters
    self.device
        See Parameters
    self.tensor_prepared: bool
        Tells whether all preparation steps have been applied to the Tensor representation of the data. This needs to be True before accessing data by indexing (see `to_tensor()`).
    self.frame_prepared: bool
        Tells whether all preparation steps have been applied to the DataFrame representation of the data.
    self.preparation_steps:
        See Parameters
    """

    def __init__(
        self,
        df: pd.DataFrame,
        history_horizon: Optional[int] = None,
        forecast_horizon: Optional[int] = None,
        encoder_features: Optional[Iterable[str]] = None,
        decoder_features: Optional[Iterable[str]] = None,
        aux_features: Optional[Iterable[str]] = None,
        target_id: Union[str, Iterable[str], None] = None,
        preparation_steps: Optional[Iterable[Callable[[pd.DataFrame], pd.DataFrame]]] = None,
        device: Union[int, str] = "cpu",
        **_,
    ):
        self.data = df

        self.encoder_features = encoder_features
        self.history_horizon = history_horizon
        self.aux_features = aux_features
        self.decoder_features = decoder_features
        self.forecast_horizon = forecast_horizon
        self.target_id = target_id

        self.device = device
        self.tensor_prepared = False
        self.frame_prepared = False
        self.preparation_steps = preparation_steps

    @property
    def encoder_features(self):
        if self._encoder_features is not None:
            return self._encoder_features.copy()
        return []

    @encoder_features.setter
    def encoder_features(self, val):
        try:
            if self._encoder_features is None or set(val) != set(self._encoder_features):
                self._encoder_features = val
                self.tensor_prepared = False
        except AttributeError:
            self._encoder_features = val
            self.tensor_prepared = False

    @property
    def decoder_features(self):
        if self._decoder_features is not None:
            return self._decoder_features.copy()
        return []

    @decoder_features.setter
    def decoder_features(self, val):
        try:
            if self._decoder_features is None or set(val) != set(self._decoder_features):
                self._decoder_features = val
                self.tensor_prepared = False
        except AttributeError:
            self._decoder_features = val
            self.tensor_prepared = False

    @property
    def aux_features(self):
        if self._aux_features is not None:
            return self._aux_features.copy()
        return []

    @aux_features.setter
    def aux_features(self, val):
        try:
            if self._aux_features is None or set(val) != set(self._aux_features):
                self._aux_features = val
                self.tensor_prepared = False
        except AttributeError:
            self._aux_features = val
            self.tensor_prepared = False

    # TODO rename target_id to target_features and treat it as iterable like the other features
    @property
    def target_id(self):
        if self._target_features is not None:
            return self._target_features.copy()
        return []

    @target_id.setter
    def target_id(self, val):
        target_features = (val,) if isinstance(val, str) else val
        try:
            if self._target_features is None or set(target_features) != set(self._target_features):
                self._target_features = target_features
                self.tensor_prepared = False
        except AttributeError:
            self._target_features = target_features
            self.tensor_prepared = False

    def __len__(self):
        # TODO add Warning when not enough data available
        if not self.history_horizon or not self.forecast_horizon:
            raise AttributeError(
                "Number of samples depends on both history- and forecast_horizon which have not been set."
            )
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
        """Adds a preparation step to the end of the existing ones.

        Parameters
        ----------
        preparation_steps: Callable[[pd.DataFrame], pd.DataFrame] (optional),
            Function or callable that transform the dataframe in any way before it is fed to the model. Both input and output need to be a pandas.Dataframe.

        """
        self.preparation_steps.append(step)
        self.frame_prepared = False
        self.tensor_prepared = False
        return self

    def clear_preparation_steps(self):
        """Remove all preparation steps.

        Returns
        -------
        self: TimeSeriesData
            The object itself

        Raises
        ------
        AttributeError
            If the steps are already applied to the underlying dataframe, they can not be reverted an thus not be removed.
        """
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
        elif steps is None:
            return df
        else:
            raise TypeError("preparation_steps should be a callable or an interable of callables.")
        return df

    def apply_prep_to_frame(self):
        """Applies the preparation steps to the Dataframe if they have not been applied already.

        Returns
        -------
        self: TimeSeriesData
            The object itself

        """
        if self.preparation_steps is None or self.frame_prepared:
            self.frame_prepared = True
            return self
        self.data = self._apply_prep_to_frame(self.data, self.preparation_steps)
        self.frame_prepared = True
        return self

    def get_as_frame(
        self, idx
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
        """Gives a sample as it would be given to the model but in pandas.DataFrames instead of torch.Tensors.

        Parameters
        ----------
        idx: Union[int,Any]
            Starting index of the sample either as int or as the type of the index of the internal DataFrame.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
            6 DataFrames (data for encoder, auxillary data for encoder, Data for decoder, auxillary data for decoder, last historic target, Targets)

        """
        if not isinstance(idx, int):
            idx = self.data.index.get_loc(idx)
        assert isinstance(idx, int)
        assert self.history_horizon is not None
        assert self.forecast_horizon is not None
        hist = self.data.iloc[idx : idx + self.history_horizon]
        fut = self.data.iloc[idx + self.history_horizon : idx + self.history_horizon + self.forecast_horizon]
        return (
            hist.filter(items=self.encoder_features, axis="columns"),
            hist.filter(items=self.aux_features, axis="columns"),
            fut.filter(items=self.decoder_features, axis="columns"),
            fut.filter(items=self.aux_features, axis="columns"),
            hist.filter(items=self.target_id, axis="columns")[-1],
            fut.filter(items=self.target_id, axis="columns"),
        )

    def __getitem__(self, idx):
        """Gets a sample ready for input into the model.

        Parameters
        ----------
        idx: int
            Starting index of the sample either as int or as the type of the index of the internal DataFrame.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            6 Tensor (data for encoder, auxillary data for encoder, Data for decoder, auxillary data for decoder, last historic target, Targets)

        """
        return (
            self.encoder_tensor[idx : idx + self.history_horizon],
            self.aux_tensor[idx : idx + self.history_horizon],
            self.decoder_tensor[idx + self.history_horizon : idx + self.history_horizon + self.forecast_horizon],
            self.aux_tensor[idx + self.history_horizon : idx + self.history_horizon + self.forecast_horizon],
            self.target_tensor[idx + self.history_horizon - 1 : idx + self.history_horizon],
            self.target_tensor[idx + self.history_horizon : idx + self.history_horizon + self.forecast_horizon],
        )

    def to(self, device: Union[str, int]):
        """Moves Tensor data to specified device.
        Will not move data if it has to be reformated anyways but will do that when reformating.

        Parameters
        ----------
        device: Union[str, int]
            Device to wich the data is moved.

        """
        self.device = device
        if self.tensor_prepared:
            if self.encoder_tensor is not None:
                self.encoder_tensor.to(device)
            if self.decoder_tensor is not None:
                self.decoder_tensor.to(device)
            if self.target_tensor is not None:
                self.target_tensor.to(device)
        return self

    def make_data_loader(
        self,
        history_horizon: int = None,
        forecast_horizon: int = None,
        encoder_features: Iterable[str] = None,
        decoder_features: Iterable[str] = None,
        aux_features: Iterable[str] = None,
        target_id: Union[str, Iterable[str]] = None,
        batch_size: int = None,
        shuffle: bool = True,
        drop_last: bool = True,
    ) -> TensorDataLoader:
        """Creates a DataLoader, which in essence is an iterator over batches of of data.

        history_horizon: int (optional)
            Number of datapoints used as "past" in a prediction. If provided will change the .
        forecast_horizon: int (optional)
            Number of datapoints used as "future" in a prediction. If not set some methods will not be able to run. It can set through the `make_dataloader` method later on.
        encoder_features: Iterable[str] (optional)
            List of the column names available to the encoder, that means it is data from the "past". This can be set later, but it has to be set before trying to access elements of a TimeSeriesData object.
        decoder_features: Iterable[str] (optional),
            List of the column names available to the encoder, that means it is data from the "past". This can be set later, but it has to be set before trying to access elements of a TimeSeriesData object.
        aux_features: Iterable[str] (optional),
            Auxillary features both encoder and decoder use, for documentation only. This can be set later, but it has to be set before trying to access elements of a TimeSeriesData object.
        target_id: Union[str, Iterable[str]] (optional),
            Column name(s) that is(are) regarded as the output in training. This can be set later, but it has to be set before trying to access elements of a TimeSeriesData object.
        batch_size: int (optional)
            Number of samples per batch, if not provided it will contain all samples in a single batch.
        shuffle: bool, default = True
            Wheter samples and batches are in random order.
        drop_last: bool, default = True
            Wheter to drop data if a batch can not be filled.

        Returns
        -------
        TensorDataLoader
            torch.DataLoader iterator over batches of sample timeseries'.

        """
        if history_horizon is not None:
            self.history_horizon = history_horizon
        if forecast_horizon is not None:
            self.forecast_horizon = forecast_horizon
        if encoder_features is not None:
            self.encoder_features = encoder_features
        if decoder_features is not None:
            self.decoder_features = decoder_features
        if aux_features is not None:
            self.aux_features = aux_features
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

    def to_tensor(
        self,
    ):
        """Creats/updates the torch.Tensor representation of the data if necessary.

        Returns
        -------
        TimeSeriesData
            The data object itself
        """
        if self.tensor_prepared:
            logger.debug("tensor already prepared")
            return self

        if not self.frame_prepared:
            logger.debug("frame not prepared")
            df = self.data.copy()
            df = self._apply_prep_to_frame(self.data, self.preparation_steps)
        else:
            df = self.data

        self.encoder_tensor = (
            torch.from_numpy(df.filter(items=self.encoder_features, axis="columns").to_numpy().astype(float))
            .float()
            .to(self.device)
        )
        # print(df.filter(items=self.decoder_features, axis="columns").to_numpy().astype(float))
        self.decoder_tensor = (
            torch.from_numpy(df.filter(items=self.decoder_features, axis="columns").to_numpy().astype(float))
            .float()
            .to(self.device)
        )
        self.aux_tensor = (
            torch.from_numpy(df.filter(items=self.aux_features, axis="columns").to_numpy().astype(float))
            .float()
            .to(self.device)
        )
        self.target_tensor = (
            torch.from_numpy(df.filter(items=self.target_id, axis="columns").to_numpy().astype(float))
            .float()
            .to(self.device)
        )
        self.tensor_prepared = True
        return self


class TensorDataLoader(dataloader.DataLoader):
    """torch.DataLoader with an additional `to(device)´ method"""

    def to(self, device):
        """Move underlying dataset to a different device.

        device: Union[int, str]
            Device ony which the data should be loaded, can be changed later.

        """
        if hasattr(self.dataset, "to"):
            self.dataset.to(device)
        else:
            raise AttributeError("Dataset does not have an attribute 'to'")
        return self
