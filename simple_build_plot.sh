#!/usr/bin/env bash
# filename: simple_build_beta_plot
#


function add_data_point {
    PLOTFILE=$1
    BETA=$2
    ACC_FILE=./model_results/model.accuracy.beta.$BETA
    echo "$BETA $(cat $ACC_FILE)" >> $PLOTFILE
}


##====---- main ----=================================================
PLOTDATAFILE=./model_results/beta_vs_acc.data


rm $PLOTDATAFILE

# Generate plot data
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


# Build plot
python3 build_plot.py $PLOTDATAFILE
cp ./beta-vs-acc.png ./model_results/
echo ""
echo "Plot can be found in ./model_results/beta-vs-acc.png"
echo ""

#octave build_beta_plot.m;
