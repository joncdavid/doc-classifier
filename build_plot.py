#!/usr/env python3
# filename: build_plot
# authors: Jon David and Jarrett Decker

import sys
import numpy as np
import matplotlib.pyplot as plt


def build(plotdatafile):
    D = np.loadtxt(plotdatafile)
    x = D[:,0]
    y = D[:,1]
    plt.plot(x,y)
    plt.xlabel("log scale of beta-values in range (0,1.0]")
    plt.ylabel("accuracy")
    plt.title("Naive Bayes Classifier Accuracy")
    plt.semilogx()
    plt.grid(True)
    plt.savefig("beta-vs-acc.png", bbox_inches="tight")
    
    
def main():
    """Defines the main function for this module."""

    if len(sys.argv) < 2:
        print("Must specify:")
        print("\t(1) [input] filename of plot data (betas vs accuracies)")
        print("\nExample:")
        print("\tpython3 build_plot.py <plotdatafile>")
        print("\nExiting.\n")
        sys.exit(2)
        
    plotdatafile = sys.argv[1]
    build(plotdatafile)
    
##==-- Main --==##
main()
