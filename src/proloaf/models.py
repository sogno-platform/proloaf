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
Holds the functions and classes of the general model architecture
"""
import importlib
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from proloaf.event_logging import create_event_logger

logger = create_event_logger(__name__)


class Encoder(nn.Module):
    """
    Encodes the history horizon part of the input data which is fed as initial state in decoder

    Parameters
    ----------
    input_size : int
        Number of features at any point in the sequence
    core_size : int
        Size of the produced state vector that encodes the recurred data
    core_layers : int, default = 1
        Number of layers of the core_net
    dropout_core : float, default = 0.0
        Internal dropout ratio
    core_net : string, default = 'torch.nn.GRU'
        The name of the desired core net
    """

    def __init__(
        self,
        input_size: int,
        core_size: int,
        core_layers: int = 1,
        dropout_core: float = 0.0,
        core_net="torch.nn.GRU",
    ):

        super(Encoder, self).__init__()
        parse = core_net.rsplit(".", 1)
        mod = importlib.__import__(parse[0], fromlist=parse[1])
        net_constructor = getattr(mod, parse[1])
        try:
            self.core = net_constructor(
                input_size=input_size,
                hidden_size=core_size,
                batch_first=True,
                num_layers=core_layers,
                dropout=dropout_core,
            )
        except:
            logger.critical(
                "The given core_net could not be constructed. It needs to have arguments with similar names to torch.nn.GRU:\n",
                "input_size: int, number of features at any point in the sequence.\n",
                "hidden_size: int, size of the produced state vector that encodes the recurred data.\n",
                "batch_first: bool, specifying that the data is of shape (batch_size, sequence_length, num_features), always True.\n",
                "num_layers: int, number of layers of the core_net.\n",
                "dropout: float, internal dropout ratio.\n"
            )
            exit()

    def forward(self, inputs):
        """
        Calculate the hidden state for the given inputs

        Parameters
        ----------
        inputs : torch.Tensor
            Tensor containing the features of the input sequence

        Returns
        -------
        torch.Tensor
            Tensor containing the hidden state for the given inputs
        """
        _, state = self.core(inputs)
        return state


class Decoder(nn.Module):
    """
    Predicts the target variable for forecast length using the future meta data as the input

    Parameters
    ----------
    input_size : int
        Number of features at any point in the sequence
    hidden_size : int
        The size of each output sample for fc1 and each input sample for fc2
    core_size : int
        Size of the produced state vector that encodes the recurred data
    out_size : int, default = 1
        Dimension of Decoder output (number of predictions)
    core_layers : int, default = 1
        Number of layers of the core_net
    dropout_fc : float, default = 0.0
        The dropout probability for the decoder
    dropout_core : float, default = 0.0
        The dropout probability for the core_net
    core_net : string, default = 'torch.nn.GRU'
        The name of the desired core net
    relu_leak : float, default = 0.01
        The value of the negative slope for the LeakyReLU
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        core_size: int,
        out_size: int = 1,
        core_layers: int = 1,
        dropout_fc: float = 0.0,
        dropout_core: float = 0.0,
        core_net="torch.nn.GRU",
        relu_leak=0.01,
    ):

        super(Decoder, self).__init__()

        parse = core_net.rsplit(".", 1)
        mod = importlib.__import__(parse[0], fromlist=parse[1])
        net_constructor = getattr(mod, parse[1])

        try:
            self.core = net_constructor(
                input_size=input_size,
                hidden_size=core_size,
                batch_first=True,
                num_layers=core_layers,
                dropout=dropout_core,
            )
        except:
            logger.critical(
                "The given core_net could not be constructed. It needs to have arguments with similar names to torch.nn.GRU:\n",
                "input_size: int, number of features at any point in the sequence.\n",
                "hidden_size: int, size of the produced state vector that encodes the recurred data.\n",
                "batch_first: bool, specifying that the data is of shape (batch_size, sequence_length, num_features), always True.\n",
                "num_layers: int, number of layers of the core_net.\n",
                "dropout: float, internal dropout ratio.\n"
            )

            exit()
        self.fc1 = nn.Linear(core_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout_fc)
        self.fc2 = nn.Linear(hidden_size, out_size)

        self.relu_leak = relu_leak

    def forward(self, inputs, states):
        """
        Calculate the prediction for the given inputs

        Parameters
        ----------
        inputs : torch.Tensor
            Tensor containing the features of the input sequence
        states : torch.Tensor
            Tensor containing the initial hidden state for each element in the batch

        Returns
        -------
        list
            The prediction
        torch.Tensor
            Tensor containing the hidden state for the given inputs
        """
        out, new_states = self.core(inputs, states)
        # for decoder seq_length = forecast horizon
        h = out[:, :, :]
        h = self.fc1(h)
        # Here output_feature_size chosen as hidden size
        h = self.dropout(h)
        h = F.leaky_relu(h, negative_slope=self.relu_leak)
        output = self.fc2(h)
        # output = torch.stack([fc(h).squeeze(dim=2) for fc in self.fc2], dim=2)
        return output, new_states


class EncoderDecoder(nn.Module):
    """
    Implements Sequence-to-Sequence model

    Provides a single interface that accepts inputs for both the Encoder and Decoder and returns the resulting prediction

    Parameters
    ----------
    input_size1 : int
        Encoder input size
    input_size2 : int
        Decoder input size
    out_size : int, default = 1
        Dimension of Decoder output (number of predictions)
    rel_linear_hidden_size : float, default = 1
        The relative linear hidden size, as a fraction of the total number of features in the training data
    rel_core_hidden_size : c
        The relative core hidden size, as a fraction of the total number of features in the training data
    core_layers : int, default = 1
        Number of layers of the core_net
    dropout_fc : float, default = 0.0
        The dropout probability for the decoder
    dropout_core : float, default = 0.0
        The dropout probability for the core_net
    core_net : string, default = 'torch.nn.GRU'
        The name of the desired core net
    relu_leak : float, default = 0.01
        The value of the negative slope for the LeakyReLU
    """

    def __init__(
        self,
        enc_size: int,
        dec_size: int,
        out_size: int = 1,
        rel_linear_hidden_size: float = 1.0,
        rel_core_hidden_size: float = 1.0,
        core_layers: int = 1,
        dropout_fc: float = 0.0,
        dropout_core: float = 0.0,
        core_net="torch.nn.GRU",
        relu_leak=0.01,
    ):
        linear_hidden_size = int(rel_linear_hidden_size * (enc_size + dec_size))
        core_hidden_size = int(rel_core_hidden_size * (enc_size + dec_size))

        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder(
            enc_size, core_hidden_size, core_layers, dropout_core, core_net
        )
        self.decoder = Decoder(
            dec_size,
            linear_hidden_size,
            core_hidden_size,
            out_size,
            core_layers,
            dropout_fc,
            dropout_core,
            core_net,
            relu_leak,
        )

    def forward(self, inputs_encoder, inputs_decoder):
        """
        Calculate predictions given Encoder and Decoder inputs

        Receives inputs and passes the relevant ones to the Encoder. The Encoder's output (hidden states)
        is then passed to the Decoder along with the Decoder's inputs, and the Decoder's predictions
        and hidden states are returned

        Parameters
        ----------
        inputs_encoder : torch.Tensor
            A tensor containing the encoder inputs
        inputs_decoder : torch.Tensor
            A tensor containing the decoder inputs

        Returns
        -------
        list
            The prediction
        torch.Tensor
            Tensor containing the hidden state for the given inputs
        """
        states_encoder = self.encoder(inputs_encoder)
        outputs_decoder, states_decoder = self.decoder(inputs_decoder, states_encoder)
        return outputs_decoder, states_decoder


class Transformer(nn.Module):
    def __init__(
        self,
        feature_size: int = 0,
        num_layers: int = 3,
        dropout: float = 0.0,
        n_heads: int = 0,
        out_size: int = 1,
        encoder_features: list = [],
        decoder_features: list = [],
    ):
        super(Transformer, self).__init__()

        self.encoder_features = encoder_features
        self.decoder_features = decoder_features

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_size,
            nhead=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_layers
        )
        self.decoder = nn.Linear(feature_size, out_size)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def prepare_input(self, inputs_encoder, inputs_decoder, device):
        diff_enc_dec = len(self.encoder_features) - len(self.decoder_features)

        inputs_decoder = torch.cat(
            (
                torch.zeros(
                    inputs_decoder.shape[0], inputs_decoder.shape[1], diff_enc_dec
                ).to(device),
                inputs_decoder,
            ),
            2,
        ).to(device)

        inputs = torch.cat((inputs_encoder, inputs_decoder), 1).to(device)

        return inputs

    def forward(self, inputs_encoder, inputs_decoder):
        device = next(self.parameters()).device

        inputs = self.prepare_input(inputs_encoder, inputs_decoder, device)

        mask = self._generate_square_subsequent_mask(inputs.shape[1]).to(device)

        attention = self.transformer_encoder(inputs, mask)
        output = self.decoder(attention)

        return output[:, -inputs_decoder.shape[1] :, :], attention
