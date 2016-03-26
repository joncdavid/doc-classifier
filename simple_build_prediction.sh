#!/usr/bin/env bash
# filename: simple_build_prediction.sh
#

MLEFILE=$1
MAPFILE=$2
PREDICTIONFILE=$3
BETA=$4

MLE_F=$MLEFILE.beta.$BETA
MAP_F=$MAPFILE.beta.$BETA
PRED_F=$PREDICTIONFILE.beta.$BETA

python3 build_predictions.py $MLE_F $MAP_F $PRED_F
