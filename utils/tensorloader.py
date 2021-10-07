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
import torch
import sklearn

import utils.datahandler as dh


class CustomTensorData:
    """
    A custom data structure that stores the inputs (for the Encoder and Decoder) and
    target data as torch.Tensors.

    Parameters
    ----------
    inputs1 : ndarray
        Encoder inputs
    inputs2 : ndarray
        Decoder inputs
    targets : ndarray
        Targets (actual values)
    """

    def __init__(self, inputs1, inputs2, targets):
        self.inputs1 = torch.Tensor(inputs1).float()
        self.inputs2 = torch.Tensor(inputs2).float()
        self.targets = torch.Tensor(targets).float()

    def __getitem__(self, index):
        return (
            self.inputs1[index, :, :],
            self.inputs2[index, :, :],
            self.targets[index, :, :],
        )

    def __len__(self):
        return self.targets.shape[0]

    def to(self, device):
        """
        See torch.Tensor.to() - does dtype and/or device conversion
        """

        self.inputs1 = self.inputs1.to(device)
        self.inputs2 = self.inputs2.to(device)
        self.targets = self.targets.to(device)
        return self


class CustomTensorDataLoader:
    """
    An iterable data structure that stores CustomTensorData (optionally shuffled into random
    permutations) in batches. Each iteration returns one batch.

    Parameters
    ----------
    dataset : CustomTensorData
        Input and target data stored in Tensors
    batch_size : int, default = 1
        The size of a batch. One training run operates on a single batch of data.
    shuffle : bool, default = False
        If True, shuffle the dataset into a random permutation
    drop_last : bool, default = True
        If True, sort the data set into as many batches of the specified size as possible.
        Ignore any remaining data that don't make up a full batch.
        If False, raise an error.

    Raises
    ------
    NotImplementedError
        Raised if drop_last is set to False.
    StopIteration
        Raised when the end of the iterator is reached.
    """

    def __init__(
        self, dataset: CustomTensorData, batch_size=1, shuffle=False, drop_last=True
    ):
        if not drop_last:
            raise NotImplementedError

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        n = self.batch_size * (len(self.dataset) // self.batch_size)
        if self.shuffle:
            permutation = np.random.permutation(len(self.dataset))[:n]
        else:
            permutation = np.arange(n)
        self.permutation = np.reshape(permutation, (len(self), self.batch_size))
        self.batch_index = 0
        return self

    def __next__(self):
        if self.batch_index < len(self):
            indices = self.permutation[self.batch_index]
            self.batch_index += 1
            return self.dataset[indices]
        else:
            raise StopIteration

    def __getitem__(self, index):
        indices = self.permutation[self.batch_index]
        return self.dataset[indices]

    def __len__(self):
        return len(self.dataset) // self.batch_size

    def to(self, device: str):
        """
        See torch.Tensor.to() - does dtype and/or device conversion
        """

        self.dataset.to(device)
        return self

    def number_features1(self):
        """
        Return the number of features in the Encoder inputs

        Returns
        -------
        int
            The number of entries in the axis that represents the features of the Encoder inputs
        """

        return self.dataset.inputs1.shape[2]

    def number_features2(self):
        """
        Return the number of features in the Decoder inputs

        Returns
        -------
        int
            The number of entries in the axis that represents the features of the Decoder inputs
        """

        return self.dataset.inputs2.shape[2]


def make_dataloader(
    df,
    target_id,
    encoder_features,
    decoder_features,
    history_horizon,
    forecast_horizon,
    batch_size=1,
    shuffle=True,
    drop_last=True,
    anchor_key=0,
    **_
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
    return CustomTensorDataLoader(
        custom_tensor_data, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
    )
