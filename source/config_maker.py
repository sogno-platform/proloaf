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

# Config-maker can be used to modify or generate config files run either with the path to the config file or the station name
# to erase the old config file before writing a new one do "python3 config_maker.py --new <stationname> "

import json
import sys
import os

main_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(main_path)

par = {}
if __name__ == '__main__':
    #read configs

    CONFIG_PATH = sys.argv[2]

    if(sys.argv[1] == '--mod'):
        with open(sys.argv[2],'r') as input:
            CONFIG_PATH = sys.argv[2]
            par = json.load(input)

    #when modifying a config file only select the parameters that are changed or creat new ones, the other will stay the same
    elif(sys.argv[1] == '--new'):
        print(CONFIG_PATH + ' has been reset before writing')
        par = {}
    else:
        print('No option selected either use either --mod or --new (WARNING: This  erases the old config file)')

    # INSERT MODIFICATIONS FOR THE CONFIG HERE
    # ============================================
    par["dropout_fc"] = 0.3
    par["dropout_core"] = 0.2
    par["core_model"] = 'gru'

    with open(CONFIG_PATH,'w') as output:
        json.dump(par, output, indent = 4)
