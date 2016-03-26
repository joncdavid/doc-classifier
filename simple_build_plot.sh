#!/usr/bin/env bash
# filename: simple_build_beta_plot
#


function add_data_point {
    PLOTFILE=$1
    BETA=$2
    echo $BETA
    ACC_FILE=./model_results/model.accuracy.beta.$BETA
    echo "$BETA $(cat $ACC_FILE)" >> $PLOTFILE
}


##====---- main ----=================================================
PLOTDATAFILE=./model_results/beta_vs_acc.data

rm $PLOTDATAFILE
add_data_point $PLOTDATAFILE 0.00001
add_data_point $PLOTDATAFILE 0.0001
add_data_point $PLOTDATAFILE 0.001
add_data_point $PLOTDATAFILE 0.01
add_data_point $PLOTDATAFILE 0.1
add_data_point $PLOTDATAFILE 0.2
add_data_point $PLOTDATAFILE 0.4
add_data_point $PLOTDATAFILE 0.6
add_data_point $PLOTDATAFILE 0.8
add_data_point $PLOTDATAFILE 1.0


## Use octave to generate plot figure.
octave build_beta_plot.m;
