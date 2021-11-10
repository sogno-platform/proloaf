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
Standard module setup.py script - Holds the functions and classes of the general model architecture
"""
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

from setuptools import find_packages

if __name__ == "__main__":
    setup(
        name="proloaf",
        version="0.2.0",
        packages=["proloaf"],
        package_dir={"": "src"},
        license="Apache License 2.0",
        install_requires=[
            "numpy",
            "pandas",
            "matplotlib",
            "torch",
            "scikit-learn",
            "datetime",
            "pyparsing>=2.2.1,<3",
            "arch",
            "optuna",
            "pmdarima",
            "statsmodels",
            "tensorboard",
        ],
    )
