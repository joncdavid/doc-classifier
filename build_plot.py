#!/usr/env python3
# filename: build_plot
# authors: Jon David and Jarrett Decker

import sys
import numpy as np
import matplotlib.pyplot as plt


def plot(plotdatafile):
    D = np.loadtxt(plotdatafile)
    x = D[:,0]
    y = D[:,1]
    plt.plot(x,y)
    plt.xlabel("log scale of beta-values in range (0,1.0]")
    plt.ylabel("accuracy")
    plt.title("Naive Bayes Classifier Accuracy")
    plt.semilogx()
    plt.grid(True)
    
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
