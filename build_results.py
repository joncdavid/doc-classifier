#!/usr/env python3
# filename: build_results.py
# authors: Jon David and Jarrett Decker

import sys

from confusionmatrix import *


#DEFAULT_RESULTS_DIR_PREFIX = "./model_results/"

def build(true_labelfilename, predicted_labelfilename,
          accuracy_outfilename, confusionmatrix_outfilename):
    """Compares predictions against true values and saves results."""
    CM = ConfusionMatrix(20, true_labelfilename,
                         predicted_labelfilename)
    confusion_matrix = CM.construct_confusion(true_labelfilename,
                                              predicted_labelfilename)
    accuracy = CM.calc_accuracy(confusion_matrix)

    # Write confusion matrix to file.
    #matrix_fname = DEFAULT_RESULTS_DIR_PREFIX + confusionmatrix_outfilename
    #matrix_fname = DEFAULT_RESULTS_DIR_PREFIX + confusionmatrix_outfilename
    CM.print_conf_matrix(confusionmatrix_outfilename, confusion_matrix)

    # Write accuracy to file.
    #accuracy_fname = DEFAULT_RESULTS_DIR_PREFIX + accuracy_outfilename
    accuracy_file = open(accuracy_outfilename, 'w')
    print("{}".format(accuracy), file=accuracy_file)
    accuracy_file.close()
    
def main():
    """Defines the main function for this module."""

    if len(sys.argv) < 5:
        print("Must specify:")
        print("\t(1) [input] filename of true/actual labels.")
        print("\t(2) [input] filename of predicted labels.")
        print("\t(3) [output] filename of accuracy results.")
        print("\t(4) [output] filename of confusion matrix.")
        print("\nExample:")
        print("\tpython3 build_predictions.py <true_label> <pred_labels>" + \
              " <output-accuracy-file> <output-confusion-file>")
        print("\nExiting.\n")
        sys.exit(2)
        
    true_labelfilename = sys.argv[1]
    predicted_labelfilename = sys.argv[2]
    accuracy_outfilename = sys.argv[3]
    confusionmatrix_outfilename = sys.argv[4]

    build(true_labelfilename, predicted_labelfilename,
          accuracy_outfilename, confusionmatrix_outfilename)
    
##==-- Main --==##
main()
