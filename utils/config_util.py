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
Provides various functions for reading, writing and working with config files.

Enables design and modification of the RNN via config files only.
"""
import os
import json
import sys
import argparse
import shutil
import plf_util.eval_metrics as metrics

def read_config(model_name = None, config_path = None, main_path=''):
    """
    Read the config file for the given model

    If no config_path is given, uses main_path to locate the 'targets' folder, which
    contains separate directories for different model names, in which the config files to be read are stored.

    Parameters
    ----------
    model_name : string, default = None
        The name of the model for which the config file should be read
    config_path : string, default = None
        The path to the config file relative to the project root, or else None
    main_path : string, default = ''
        The path to the project root

    Returns
    -------
    dict
        A Dictionary containing the parameters read from the config file

    """
    if config_path is None:
        config_path = os.path.join('targets',  model_name, 'config.json')
    with open(os.path.join(main_path,config_path),'r') as input:
        return json.load(input)

def write_config(config, model_name = None, config_path = None, main_path=''):
    """
    Write the given data to the specified config file.

    If no config_path is given, uses main_path to locate the 'targets' folder, which
    contains separate directories for different model names, in which the config files to be
    written are stored.

    Parameters
    ----------
    config : dict
        A dictionary containing the data to be written
    model_name : string, default = None
        The name of the model for which the config file should be written
    config_path : string, default = None
        The path to the config file relative to the project root, or else None
    main_path : string, default = ''
        The path to the folder containing the 'targets' folder

    Returns
    -------
    None
        json.dump() has no return value.
    """
    if config_path is None:
        config_path = os.path.join(main_path, 'targets',  model_name, 'config.json')
    with open(os.path.join(main_path,config_path),'w') as output:
        return json.dump(config, output, indent=4)

class flag_and_store(argparse._StoreAction):
    def __init__(self, option_strings, dest, dest_const, const, nargs = 0, **kwargs):
        self.dest_const = dest_const
        if isinstance(const, list):
            self.val = const[1:]
            self.flag = const[0]
        else:
            self.val = []
            self.flag = const
        if isinstance(nargs, int):
            nargs_store = nargs+len(self.val)
        else:
            nargs_store = nargs
        super(flag_and_store, self).__init__(option_strings, dest, const = None, nargs = nargs_store, **kwargs)
        self.nargs = nargs

    def __call__(self, parser, namespace, values, option_strings = None):
        setattr(namespace, self.dest_const,self.flag)
        if isinstance(values,list):
            self.val.extend(values)
        elif values is not None:
            self.val.append(values)

        if len(self.val) == 1:
            self.val = self.val[0]
        super().__call__(parser, namespace, self.val, option_strings)

def parse_basic(args = sys.argv[1:]):
    """
    Parse command line arguments and store them in attributes of a Python object.

    Accepts (only) one of the following options:
    - "-s", "--station", the name of the station to be trained for. 'gefcom2017/nh_data' by default.
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
    ident = parser.add_mutually_exclusive_group()#required=True)
    ident.add_argument("-s","--station", help = "station to be trained for (e.g. gefcom2017/nh_data)", default='gefcom2017/nh_data')
    ident.add_argument("-c", "--config", help = "path to the config file relative to the project root")
    return parser.parse_args(args)

def parse_with_loss(args = sys.argv[1:]):
    """
    Parse command line arguments, including arguments for the loss function, and store them in attributes of a Python object.

    Accepts the following options:
    - "--ci", Enables execution mode optimized for GitLab's CI
    - "--logname", Name of the run, displayed in Tensorboard
    AND (only) one of the following options:
    - "-s", "--station", the name of the station to be trained for. 'gefcom2017/nh_data' by default.
    - "-c", "--config", the path to the config file relative to the project root
    AND (only) one of the following options (which gets stored as the 'loss' attribute):
    - "--nll_gauss", Enables training with nll guassian loss
    - "--quantiles", Enables training with pinball loss and MSE with q1 and q2 being the upper and lower quantiles
    - "--crps","--crps_gaussian", Enables training with crps gaussian loss
    - "--mse", "--mean_squared_error", Enables training with mean squared error
    - "--rmse", "--root_mean_squared_error", Enables training
    - "--mape", Enables training with root mean squared error
    - "--sharpness", Enables training with sharpness
    - "--mis", "--mean_interval_score", Enables training with mean interval score

    Parameters
    ----------
    args : list
        A list containing all command line arguments starting with argv[1].

    Returns
    -------
    Namespace
        An object that stores the given values of parsed arguments as attributes
    dict
        Contains extra options if the loss functions mis or quantile score are used
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("--ci", help="Enables execution mode optimized for GitLab's CI", action='store_true', default=False)
    parser.add_argument("--logname", help="Name of the run, displayed in Tensorboard", type=str, action='store', default="")

    # TODO this should be a required argument (also remove default below)
    ident = parser.add_mutually_exclusive_group()#required=True)
    ident.add_argument("-s","--station", help = "station to be trained for (e.g. gefcom2017/nh_data)", default='gefcom2017/nh_data')
    ident.add_argument("-c", "--config", help = "path to the config file relative to the project root")

    losses = parser.add_mutually_exclusive_group()
    #losses.add_argument("--hyper", help="turn hyperparam-tuning on/off next time (int: 1=on, else=off)", type=int, default=0)
    #losses.add_argument("-o", "--overwrite", help = "overwrite config with new training parameter (int: 1=True=default/else=False)", type=int, default=0)
    losses.add_argument("--nll_gauss", dest_const='loss', dest='num_pred', nargs=0, type=int, const=[metrics.nll_gauss,2], help = "train with nll guassian loss", action=flag_and_store)
    losses.add_argument("--quantiles", dest_const='loss', dest='quantiles',metavar=' q1 q2', nargs='+', type=float, const=metrics.quantile_score, help = "train with pinball loss and MSE with q1 and q2 being the upper and lower quantiles", action=flag_and_store)
    losses.add_argument("--crps","--crps_gaussian", dest_const='loss', dest='num_pred', nargs=0, type=int, const=[metrics.crps_gaussian,2], help = "train with crps gaussian loss", action=flag_and_store)
    losses.add_argument("--mse", "--mean_squared_error", dest_const='loss', dest='num_pred', nargs=0, type=int, const=[metrics.mse,1], help = "train with mean squared error", action=flag_and_store)
    losses.add_argument("--rmse", "--root_mean_squared_error", dest_const='loss', dest='num_pred', nargs=0, type=int, const=[metrics.rmse,1], help = "train with root mean squared error", action=flag_and_store)
    losses.add_argument("--mape", dest_const='loss', dest='num_pred', nargs=0, type=int, const=[metrics.mape,1], help = "train with root mean absolute error", action=flag_and_store)
    # losses.add_argument("--mase", dest_const='loss', dest='num_pred', nargs=0, type=int, const=[metrics.mase,2], help = "train with root mean absolute scaled error", action=flag_and_store)
    # losses.add_argument("--picp", dest_const='loss', dest='num_pred', nargs=0, type=int, const=[metrics.picp_loss,2], help = "train with 1 - prediction intervall coverage",action=flag_and_store)
    losses.add_argument("--sharpness", dest_const='loss', dest='num_pred', nargs=0, type=int, const=[metrics.sharpness,1], help = "train with sharpness", action=flag_and_store)
    losses.add_argument("--mis", "--mean_interval_score", dest_const='loss', dest='alpha', nargs=1,metavar = 'A', type=float, const=metrics.mis, help = "train with mean interval score. 'A' corresponds to the inverse weight", action=flag_and_store)
    #TODO write mis as default argument as soon as evaluate works with it
    parser.set_defaults(loss=metrics.nll_gauss, num_pred=2)
    ret = parser.parse_args(args)

    if ret.loss == metrics.mis:
        ret.num_pred = 2
        options = dict(alpha=ret.alpha)
    if ret.loss == metrics.quantile_score:
        ret.num_pred = len(ret.quantiles) + 1
        options = dict(quantiles = ret.quantiles)
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

    ## Source: https://stackoverflow.com/questions/3041986/apt-command-line-interface-like-yes-no-input
    # valid = {"yes": False, "y": False, "ye": False,
    #         "no": True, "n": True}

    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
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
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")

def clean_up_tensorboard_dir(run_dir):
    """
    Move hyperparameter logs out of subfolders

    Iterate over all subdirectories of run_dir, moving any files they contain into run_dir.
    Neither sub-subdirectories nor their contents are moved. Delete them along with the subdirectories.

    Parameters
    ----------
    run_dir : string
        The path to the run directory

    Returns
    -------
    No return value
    """
    # move hparam logs out of subfolders
    subdir_list = [x for x in os.listdir(run_dir) if os.path.isdir(os.path.join(run_dir,x))]  # gets a list of all subdirectories in the run directory

    for dir in subdir_list:
        subdir = os.path.join(run_dir, dir)  # complete path from root to current subdir
        files =  [x for x in os.listdir(subdir) if os.path.isfile(os.path.join(subdir, x))]     # gets all files in the current subdir
        for f in files:
            shutil.move(os.path.join(subdir, f), run_dir)   # moves the file out of the subdir
        shutil.rmtree(subdir)  # removes the now empty subdir
        # !! only files located directly in the subdir are moved, sub-subdirs are not iterated and deleted with all their content !!
