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
This script changes parameters in the config such that the training will run very quickly but the model will not be very good.
That is helpful if the trainingrun is purely technically but everything else should be realistic.
"""

import json
import sys
import os
from proloaf.event_logging import create_event_logger



logger = create_event_logger(__name__)

if __name__ == '__main__':
    # Read configs
    main_path = os.path.dirname(os.path.realpath(__file__))
    config_path = sys.argv[1]
    with open(config_path, 'r') as input:
        main_config = json.load(input)
    tuning_path = main_config.get("exploration_path")
    main_config["max_epochs"] = 2
    tuning_config = None
    if tuning_path:
        try:
            with open(tuning_path, 'r') as input:
                tuning_config = json.load(input)
        except RuntimeError:
            logger.info("tuning config could not be loaded")
    if tuning_config:
        main_config["exploration"] = True
        tuning_config["number_of_tests"] = 2
        with open(tuning_path, 'w') as output:
            json.dump(tuning_config, output, indent=4)

    with open(config_path, 'w') as output:
        json.dump(main_config, output, indent=4)
    

