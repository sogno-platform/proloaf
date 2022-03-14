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