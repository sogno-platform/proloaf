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
    with open("README.md") as f:
        long_description = f.read()

    setup(
        name="proloaf",
        version="0.2.1",
        author="Gonca Gürses-Tran",
        author_email="gguerses@eonerc.rwth-aachen.de",
        url="https://github.com/sogno-platform/proloaf",
        license="Apache License 2.0",
        description="A Probabilistic Load Forecasting Project.",
        long_description=long_description,
        long_description_content_type="text/markdown",
        packages=["proloaf"],
        classifiers=[
            "Topic :: Scientific/Engineering",
            "License :: OSI Approved :: Apache Software License",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3.8",
        ],
        package_dir={"": "src"},
        python_requires=">=3.8",
        install_requires=[
            "numpy",
            "pandas",
            "matplotlib",
            "torch>1.9.0",
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
