#!/usr/env python3
# filename: build_predictions.py
# authors: Jon David and Jarrett Decker

import sys

from classifier import *


#DEFAULT_MODEL_DIR_PREFIX = "./models/"
DEFAULT_PREDICTION_DIR_PREFIX = "./model_predictions/"

def build(mleinfile, mapinfile, predfile):
    """Classifies test data file and makes predictions."""
    pstr = "\tBuilding predictions..." + \
      "\n\t\tMLEfile:{}\n\t\tMAPfile:{}\n\t\tPredictionsFile:{}\n"
    #predfile = DEFAULT_PREDICTION_DIR_PREFIX + predictionoutfile
    print(pstr.format(mleinfile, mapinfile, predfile))
    
    c = NaiveBayesClassifier(mlefile=mleinfile,
                             mapfile=mapinfile)
    predictions = c.classify(mleinfile, mapinfile)

    pred_f = open(predfile, 'w')
    for prediction in predictions.tolist():
        print(int(prediction), file=pred_f)
    pred_f.close()

def main():
    """Defines the main function for this module."""

    if len(sys.argv) < 4:
        print("Must specify:")
        print("\t(1) [input] MLE filename")
        print("\t(2) [input] MAP filename")
        print("\t(3) [output] Predictions filename")
        print("Example:")
        print("\tpython3 build_predictions.py <mlefile> <mapfile> <predfile>")
        print("\nExiting.\n")
        sys.exit(2)
        
    mleinfile = sys.argv[1]
    mapinfile = sys.argv[2]
    predoutfile = sys.argv[3]
    
    build(mleinfile, mapinfile, predoutfile)

    
##==-- Main --==##
main()
