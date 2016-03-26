#!/usr/bin/env bash
# filename: simple_build_model.sh
#

MLEFILE=$1
MAPFILE=$2
EVIDENCEFILE=$3
RANKFILE=$4
BETA=$5

MLE_F=$MLEFILE.beta.$BETA
MAP_F=$MAPFILE.beta.$BETA
EVI_F=$EVIDENCEFILE.beta.$BETA
RNK_F=$RANKFILE.beta.$BETA

python3 build_ranks.py $MLE_F $MAP_F $EVI_F $RNK_F
