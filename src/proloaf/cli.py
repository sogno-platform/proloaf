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
Provides various functions to run command line interface.
"""

import sys
import argparse

class flag_and_store(argparse._StoreAction):
    """Action for the argparse to set a flag and store a variable
    amount of arguments.
    
    For more info see argparse documentation.

    """
    def __init__(self, option_strings:str, dest:str, dest_const:str, const:str, nargs=0, **kwargs):
        self.dest_const = dest_const
        self.flag = const
        super(flag_and_store, self).__init__(
            option_strings, dest, const=None, nargs=nargs, **kwargs
        )
        self.nargs = nargs

    def __call__(self, parser, namespace, values, option_strings=None):
        setattr(namespace, self.dest_const, self.flag)
        super().__call__(parser, namespace, values, option_strings)


def parse_basic(args=sys.argv[1:]):
    """
    Parse command line arguments and store them in attributes of a Python object.

    Accepts (only) one of the following options:

    - "-s", "--station", the name of the station to be trained for. 'opsd' by default.
    - "-c", "--config", the path to the config file relative to the project root

    Parameters
    ----------
    args : list
        A list containing all command line arguments starting with argv[1].

    Returns
    -------
    Namespace
        An object that stores the given values of parsed arguments as attributes
    """

    parser = argparse.ArgumentParser()
    # TODO this should be a required argument (also remove default below)
    ident = parser.add_mutually_exclusive_group()  # required=True)
    ident.add_argument(
        "-s",
        "--station",
        help="station to be trained for (e.g. opsd)",
        default="opsd",
    )
    ident.add_argument(
        "-c", "--config", help="path to the config file relative to the project root"
    )
    return parser.parse_args(args)


def parse_with_loss(args=sys.argv[1:]):
    """
    Parse command line arguments, including arguments for the loss function, and store them in attributes of a Python object.

    Accepts the following options:

    - "--ci", Enables execution mode optimized for GitLab's CI
    - "--logname", Name of the run, displayed in Tensorboard

    AND (only) one of the following options:

    - "-s", "--station", the name of the station to be trained for. 'opsd' by default.
    - "-c", "--config", the path to the config file relative to the project root

    AND (only) one of the following options (which gets stored in the 'loss' attribute as a callable):

    - "--nll_gauss", Enables training with nll guassian loss
    - "--quantiles", Enables training with pinball loss and MSE with q1 and q2 being the upper and lower quantiles.
    Requires at least 2 float arguments e.g. "--quantiles q1, q2, ..."
    - "--crps","--crps_gaussian", Enables training with crps gaussian loss
    - "--mse", "--mean_squared_error", Enables training with mean squared error
    - "--rmse", "--root_mean_squared_error", Enables training with root mean squared error
    - "--mape", Enables training with root mean absolute error (mean absolute percentage error in %)
    - "--sharpness", Enables training with sharpness
    - "--mis", "--mean_interval_score", Enables training with mean interval score. Requires a single float as an
    argument e.g. "--mis alpha" where alpha is a float value.

    Parameters
    ----------
    args : list
        A list containing all command line arguments starting with argv[1].

    Returns
    -------
    Namespace
        An object that stores the following attributes:

        - ci : True/False (see function comment)
        - logname : A string (see function comment), '' by default
        - station : A string (see function comment) e.g. 'opsd' or 'gefcom2017/nh_data'
        - config : A string (see function comment), None by default
        - num_pred : An int with the number of predictions
        - quantiles : A list of floats, if '--quantiles' was used, otherwise None
        - alpha : A float, if '--mis' was used, otherwise None
        - loss : The specified callable loss function from proloaf.models.py
    dict
        Contains extra options if the loss functions mis or quantile score are used.

        - if '--mis a' is used, where 'a' is a float: the dict has a single entry i.e. {'alpha': a }
        - if "--quantiles q1 q2 ..." is used, where q1, q2,... are a series of (at least 2) floats: the dict contains a
        list with the entries q1, q2, ... i.e. {'quantiles': [q1, q2, ...]}
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ci",
        help="Enables execution mode optimized for GitLab's CI",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--logname",
        help="Name of the run, displayed in Tensorboard",
        type=str,
        action="store",
        default="",
    )

    # TODO this should be a required argument (also remove default below)
    ident = parser.add_mutually_exclusive_group()  # required=True)
    ident.add_argument(
        "-s",
        "--station",
        help="station to be trained for (e.g. opsd)",
        default="opsd",
    )
    ident.add_argument(
        "-c", "--config", help="path to the config file relative to the project root"
    )

    losses = parser.add_mutually_exclusive_group()
    # losses.add_argument("--hyper", help="turn hyperparam-tuning on/off next time (int: 1=on, else=off)", type=int, default=0)
    # losses.add_argument("-o", "--overwrite", help = "overwrite config with new training parameter (int: 1=True=default/else=False)", type=int, default=0)
    losses.add_argument(
        "--nll_gauss",
        dest="loss",
        const="NllGauss",
        help="train with nll guassian loss",
        action="store_const",
    )
    losses.add_argument(
        "--quantiles",
        dest_const="loss",
        dest="loss_arguments",
        metavar=" q1 q2",
        nargs="+",
        type=float,
        const="PinnballLoss",
        help="train with pinball loss and MSE with q1 and q2 being the upper and lower quantiles",
        action=flag_and_store,
    )

    losses.add_argument(
        "--smoothed_quantiles",
        dest_const="loss",
        dest="loss_arguments",
        metavar=" q1 q2",
        nargs="+",
        type=float,
        const="SmoothedPinnballLoss",
        help="train with pinball loss and MSE with q1 and q2 being the upper and lower quantiles",
        action=flag_and_store,
    )

    losses.add_argument(
        "--crps",
        "--crps_gaussian",
        dest="loss",
        const="CRPSGauss",
        help="train with crps gaussian loss",
        action="store_const",
    )
    losses.add_argument(
        "--mse",
        "--mean_squared_error",
        dest="loss",
        const="Mse",
        help="train with mean squared error",
        action="store_const",
    )
    losses.add_argument(
        "--rmse",
        "--root_mean_squared_error",
        dest="loss",
        const="Rmse",
        help="train with root mean squared error",
        action="store_const",
    )
    losses.add_argument(
        "--mape",
        dest="loss",
        const="Mape",
        help="train with root mean absolute error",
        action="store_const",
    )
    # losses.add_argument("--mase", dest_const='loss', dest='num_pred', nargs=0, type=int, const=[metrics.mase,2], help = "train with root mean absolute scaled error", action=flag_and_store)
    # losses.add_argument(
    #     "--picp",
    #     dest="loss",
    #     const="PicpLoss",
    #     help="train with sharpness",
    #     action="store_const",
    # )
    losses.add_argument(
        "--sharpness",
        dest="loss",
        const="Sharpness",
        help="train with sharpness",
        action="store_const",
    )
    losses.add_argument(
        "--mis",
        "--mean_interval_score",
        dest_const="loss",
        dest="loss_arguments",
        nargs=1,
        metavar="A",
        type=float,
        const="Mis",
        help="train with mean interval score. 'A' corresponds to the inverse weight",
        action=flag_and_store,
    )
    # TODO write mis as default argument as soon as evaluate works with it
    parser.set_defaults(loss="NllGauss")
    ret = parser.parse_args(args)

    if ret.loss.lower() == "mis":
        options = dict(alpha=ret.loss_arguments)
    elif ret.loss.lower() == "quantilescore":
        options = dict(quantiles=ret.loss_arguments)
    else:
        options = {}
    return ret, options


def query_true_false(question, default="yes"):
    """
    Ask a yes/no question via raw_input() and return their answer.

    Parameters
    ----------
    question : string
        A string with the question that is presented to the user
    default : string or None default = "yes"
        The presumed answer if the user just hits <Enter>. Valid string values are "yes" or "no". None means an
        answer is required of the user.

    Returns
    -------
    bool
        True for "yes" or False for "no".

    """

    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")