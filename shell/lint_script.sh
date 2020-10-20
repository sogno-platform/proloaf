#! /bin/bash

flake8 --output-file ${1}_lint.txt source/${1}.py
flake8_junit ${1}_lint.txt ${1}_lint.xml
