#! /bin/bash

if [ -z "$STATION" ]
then
    echo "No station information given, contain the station name in the \$STATION variable"
else
    echo "Starting training for ${STATION}"
    python3 source/fc_train.py -s ${STATION} --ci
fi
