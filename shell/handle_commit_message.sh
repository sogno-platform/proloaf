#! /bin/bash

if [ -z "$STATION" ]
then
    HOLD=${CI_COMMIT_MESSAGE#* [}
    HOLD=${HOLD%]*}
    if [ "$HOLD" == "$CI_COMMIT_MESSAGE" ]
    then
        echo "No path to station defined, skiping..."
    else
        echo "Starting training for ${HOLD}"
        python3 source/fc_train.py -s ${HOLD} --ci
    fi
else
    echo "Starting training for ${STATION}"
    python3 source/fc_train.py -s ${STATION} --ci
fi
