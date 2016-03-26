#!/usr/bin/env bash
# filename: simple_build_model.sh
#

DATAFILE=$1
LABELFILE=$2
MLEFILE=$3
MAPFILE=$4
BETA=$5

MLE_F=$MLEFILE.beta.$BETA
MAP_F=$MAPFILE.beta.$BETA

ARG_BETA=--beta=$BETA
ARG_DATA=--data=$DATAFILE
ARG_LABL=--label=$LABELFILE
ARG_MLE=--mle=$MLE_F
ARG_MAP=--map=$MAP_F

python3 build_models.py $ARG_BETA $ARG_DATA $ARG_LABL $ARG_MLE $ARG_MAP
