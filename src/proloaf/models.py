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
from typing import Tuple
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
        except Exception as exc:
            logger.critical(
                "The given core_net could not be constructed. It needs to have arguments with similar names to torch.nn.GRU:\n",
                "input_size: int, number of features at any point in the sequence.\n",
                "hidden_size: int, size of the produced state vector that encodes the recurred data.\n",
                "batch_first: bool, specifying that the data is of shape (batch_size, sequence_length, num_features), always True.\n",
                "num_layers: int, number of layers of the core_net.\n",
                "dropout: float, internal dropout ratio.\n",
            )
            raise exc

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
        Dimension of Decoder output (number of predicted features)
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

        self.empty_input = input_size == 0
        if self.empty_input:
            input_size = 1

        try:
            self.core = net_constructor(
                input_size=input_size,  # + out_size,
                hidden_size=core_size,
                batch_first=True,
                num_layers=core_layers,
                dropout=dropout_core,
            )
        except Exception as exc:
            logger.critical(
                "The given core_net could not be constructed. It needs to have arguments with similar names to torch.nn.GRU:\n",
                "input_size: int, number of features at any point in the sequence.\n",
                "hidden_size: int, size of the produced state vector that encodes the recurred data.\n",
                "batch_first: bool, specifying that the data is of shape (batch_size, sequence_length, num_features), always True.\n",
                "num_layers: int, number of layers of the core_net.\n",
                "dropout: float, internal dropout ratio.\n",
            )

            raise exc
        self.fc1 = nn.Linear(core_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout_fc)
        self.fc2 = nn.Linear(hidden_size, out_size)

        self.relu_leak = relu_leak

    def forward(self, inputs, states, last_value):
        """
        Calculate the prediction for the given inputs

        Parameters
        ----------
        inputs : torch.Tensor
            Tensor containing the features of the input sequence
        states : torch.Tensor
            Tensor containing the initial hidden state for each element in the batch
        last_value : torch.Tensor
            Last known target value(s) in history horizon

        Returns
        -------
        torch.Tensor
            The prediction
        torch.Tensor
            Tensor containing the hidden state for the given inputs
        """
        dev = next(self.parameters()).device
        if self.empty_input:
            inputs = torch.zeros((inputs.size(0), inputs.size(1), 1), device=dev)
        # interpret hiddenstate from encoder as output
        out = last_value
        # for i in range(inputs.size(1)):
        out, states = self.core(inputs, states)
        out = self.fc1(out)
        # Here output_feature_size chosen as hidden size
        out = self.dropout(out)
        out = F.leaky_relu(out, negative_slope=self.relu_leak)
        out = self.fc2(out)

        return out, states


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
        Dimension of Decoder output (number of predicted features)
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
        n_target_features: int = 1,
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
        self.out_size = out_size
        self.n_target_features = n_target_features
        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder(
            enc_size, core_hidden_size, core_layers, dropout_core, core_net
        )
        self.decoder = Decoder(
            dec_size,
            linear_hidden_size,
            core_hidden_size,
            out_size * n_target_features,
            core_layers,
            dropout_fc,
            dropout_core,
            core_net,
            relu_leak,
        )

    def forward(
        self, inputs_enc, inputs_enc_aux, inputs_dec, inputs_dec_aux, last_value
    ):
        """
        Calculate predictions given Encoder and Decoder inputs

        Receives inputs and passes the relevant ones to the Encoder. The Encoder's output (hidden states)
        is then passed to the Decoder along with the Decoder's inputs, and the Decoder's predictions
        and hidden states are returned

        Parameters
        ----------
        inputs_enc : torch.Tensor
            Contains the input data for the encoder
        inputs_enc_aux : torch.Tensor
            Contains auxillary input data for the encoder
        inputs_dec : torch.Tensor
            Contains the input data for the decoder
        inputs_dec_aux : torch.Tensor
            Contains auxillary input data for the decoder
        last_value : torch.Tensor
            Last known target value(s) in history horizon

        Returns
        -------
        torch.Tensor
            The prediction
        torch.Tensor
            Tensor containing the hidden state for the given inputs
        """
        inputs_enc = torch.cat((inputs_enc, inputs_enc_aux), dim=2)
        states_encoder = self.encoder(inputs_enc)
        inputs_dec = torch.cat((inputs_dec, inputs_dec_aux), dim=2)
        outputs_decoder, states_decoder = self.decoder(
            inputs_dec, states_encoder, last_value
        )
        return (
            outputs_decoder.reshape(
                outputs_decoder.size(0),
                outputs_decoder.size(1),
                self.n_target_features,
                self.out_size,
            ),
            states_decoder,
        )


class AutoEncoder(nn.Module):
    """
    Implements a RNN Autoencoder. Output length is he same as input -1.

    Provides a single interface that accepts inputs for the Encoderand returns the resulting reconstruction.

    Parameters
    ----------
    enc_size : int
        Encoder input size
    dec_size : int, default = 0
        Decoder input size
    rel_linear_hidden_size : float, default = 1
        The relative linear hidden size, as a fraction of the total number of features in the training data
    rel_core_hidden_size :
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
        dec_size: int = 0,
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
        self.enc_size = enc_size
        self.out_size = out_size
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(
            enc_size, core_hidden_size, core_layers, dropout_core, core_net
        )
        self.decoder = Decoder(
            input_size=dec_size,
            hidden_size=linear_hidden_size,
            core_size=core_hidden_size,
            out_size=enc_size * out_size,
            core_layers=core_layers,
            dropout_fc=dropout_fc,
            dropout_core=dropout_core,
            core_net=core_net,
            relu_leak=relu_leak,
        )

    def forward(
        self,
        inputs_enc: torch.Tensor,
        inputs_enc_aux: torch.Tensor,
        inputs_dec: torch.Tensor,
        inputs_dec_aux: torch.Tensor,
        last_value: torch.Tensor,
    ):
        """
        Calculate predictions given Encoder and Decoder inputs

        Receives inputs and passes the relevant ones to the Encoder. The Encoder's output (hidden states)
        is then passed to the Decoder along with the Decoder's inputs, and the Decoder's predictions
        and hidden states are returned

        Parameters
        ----------
        inputs_enc : torch.Tensor
            Contains the input data for the encoder
        inputs_enc_aux : torch.Tensor
            Contains auxillary input data for the encoder

        Returns
        -------
        torch.Tensor
            The prediction
        torch.Tensor
            Tensor containing the hidden state for the given inputs
        """
        inputs_enc = torch.cat((inputs_enc, inputs_enc_aux), dim=2)
        states_encoder = self.encoder(inputs_enc)
        # input to decoder is nothing but with length of encoder input
        inputs_dec = torch.empty(
            inputs_enc.size(0), inputs_enc.size(1), 0, device=inputs_enc.device
        )
        # it should recieve encoder aux inputs since it tries to reconstruct the same timeframe
        # inputs_dec = torch.cat((inputs_dec, inputs_enc_aux), dim=2) # XXX not provideing aux_features to decoder
        outputs_decoder, states_decoder = self.decoder(
            inputs=torch.flip(
                inputs_dec[:, :-1], dims=(1,)
            ),  # everything but last value
            last_value=inputs_enc[:, -1:],  # only last value
            states=states_encoder,
        )
        return (
            outputs_decoder.reshape(
                outputs_decoder.size(0),
                outputs_decoder.size(1),
                self.enc_size,
                self.out_size,
            ),
            states_decoder,
        )


class DualModel(nn.Module):
    """
    Implements Hybrid-Model consisting of one RNN encoder and two decoders.
    One decoder performs reconstruction of the input values like an Autoencoder,
    the other one creates a predicton

    Provides a single interface that accepts inputs for both the Encoder and Decoder and returns the resulting prediction.

    Parameters
    ----------
    enc_size : int
        Encoder input size
    dec_size : int, default = 0
        Decoder input size
    out_size : int, default = 1
        Dimension of Decoder output (number of predicted features)
    rel_linear_hidden_size : float, default = 1
        The relative linear hidden size, as a fraction of the total number of features in the training data
    rel_core_hidden_size :
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
        dec_size: int = 0,
        out_size: Tuple[int, int] = (1, 1),
        n_target_features: int = 1,
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
        self.out_size = out_size
        self.enc_size = enc_size
        self.n_target_features = n_target_features

        super(DualModel, self).__init__()
        self.encoder = Encoder(
            enc_size, core_hidden_size, core_layers, dropout_core, core_net
        )
        self.auto_decoder = Decoder(
            input_size=0,
            hidden_size=linear_hidden_size,
            core_size=core_hidden_size,
            out_size=enc_size * out_size[1],
            core_layers=core_layers,
            dropout_fc=dropout_fc,
            dropout_core=dropout_core,
            core_net=core_net,
            relu_leak=relu_leak,
        )
        self.future_decoder = Decoder(
            input_size=dec_size,
            hidden_size=linear_hidden_size,
            core_size=core_hidden_size,
            out_size=n_target_features * out_size[0],
            core_layers=core_layers,
            dropout_fc=dropout_fc,
            dropout_core=dropout_core,
            core_net=core_net,
            relu_leak=relu_leak,
        )

    def forward(
        self,
        inputs_enc: torch.Tensor,
        inputs_enc_aux: torch.Tensor,
        inputs_dec: torch.Tensor,
        inputs_dec_aux: torch.Tensor,
        last_value: torch.Tensor,
    ):
        """
        Calculate predictions given Encoder and Decoder inputs

        Receives inputs and passes the relevant ones to the Encoder. The Encoder's output (hidden states)
        is then passed to the Decoder along with the Decoder's inputs, and the Decoder's predictions
        and hidden states are returned

        Parameters
        ----------
        inputs_enc : torch.Tensor
            Contains the input data for the encoder
        inputs_enc_aux : torch.Tensor
            Contains auxillary input data for the encoder
        inputs_dec : torch.Tensor
            Contains the input data for the decoder
        inputs_dec_aux : torch.Tensor
            Contains auxillary input data for the decoder
        last_value : torch.Tensor
            Last known target value(s) in history horizon

        Returns
        -------
        Tuple[torch.Tensor,torch.Tensor]
            (output of the prediction, output of reconstruction)
        Tuple[torch.Tensor,torch.Tensor]
            (Hidden state of prediction decoder, Hiddenstate of reconstruction decoder)
        """
        inputs_enc = torch.cat((inputs_enc, inputs_enc_aux), dim=2)
        states_enc = self.encoder(inputs_enc)

        inputs_dec_auto = torch.empty(
            inputs_enc.size(0), inputs_enc.size(1), 0, device=inputs_enc.device
        )
        # it should recieve encoder aux inputs since it tries to reconstruct the same timeframe
        # inputs_dec_auto = torch.cat((inputs_dec_auto, inputs_enc_aux), dim=2)
        inputs_dec_future = torch.cat((inputs_dec, inputs_dec_aux), dim=2)
        outputs_auto_dec, states_auto_dec = self.auto_decoder(
            inputs=torch.flip(inputs_dec_auto[:, :-1], dims=(1,)),
            last_value=inputs_enc[:, -1:],
            states=states_enc,
        )
        outputs_future_dec, states_future_dec = self.future_decoder(
            inputs=inputs_dec_future, last_value=last_value, states=states_enc
        )
        return (
            (
                outputs_future_dec.reshape(
                    outputs_future_dec.size(0),
                    outputs_future_dec.size(1),
                    self.n_target_features,
                    self.out_size[0],
                ),
                outputs_auto_dec.reshape(
                    outputs_auto_dec.size(0),
                    outputs_auto_dec.size(1),
                    self.enc_size,
                    self.out_size[1],
                ),
            ),
            (states_future_dec, states_auto_dec),
        )


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
        dim_feedforward: int = 1024,
    ):
        super(Transformer, self).__init__()

        self.encoder_features = encoder_features
        self.decoder_features = decoder_features

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_size,
            nhead=n_heads,
            dropout=dropout,
            batch_first=True,
            dim_feedforward=dim_feedforward,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
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

    def prepare_input(self, inputs_enc, inputs_dec, device):
        diff_enc_dec = len(self.encoder_features) - len(self.decoder_features)
        if diff_enc_dec > 0:
            inputs_dec = torch.cat(
                (
                    torch.zeros(
                        inputs_dec.shape[0],
                        inputs_dec.shape[1],
                        diff_enc_dec,
                    ).to(device),
                    inputs_dec,
                ),
                2,
            ).to(device)
        if diff_enc_dec < 0:
            inputs_enc = torch.cat(
                (
                    torch.zeros(
                        inputs_enc.shape[0],
                        inputs_enc.shape[1],
                        -diff_enc_dec,
                    ).to(device),
                    inputs_enc,
                ),
                2,
            ).to(device)

        inputs = torch.cat((inputs_enc, inputs_dec), 1).to(device)

        return inputs

    def forward(
        self,
        inputs_enc: torch.Tensor,
        inputs_enc_aux: torch.Tensor,
        inputs_dec: torch.Tensor,
        inputs_dec_aux: torch.Tensor,
        last_value: torch.Tensor,
    ):
        device = next(self.parameters()).device
        inputs_enc = torch.cat((inputs_enc, inputs_enc_aux), dim=2)
        inputs_dec = torch.cat((inputs_dec, inputs_dec_aux), dim=2)
        inputs = self.prepare_input(inputs_enc, inputs_dec, device)

        mask = self._generate_square_subsequent_mask(inputs.shape[1]).to(device)
        attention = self.transformer_encoder(inputs, mask)
        output = self.decoder(attention).unsqueeze(2)

        return output[:, -inputs_dec.shape[1] :, ...], attention
