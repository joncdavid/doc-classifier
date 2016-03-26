#!/usr/env python3
# filename: test_classifier.py
# authors: Jon David and Jarrett Decker

from confusionmatrix import *

DEFAULT_TEST_PATH = "./test_output/"

def test_construct_confusion(CM):
    print("\tTesting construct_confusion()...")
    fname = DEFAULT_TEST_PATH + "test_construct_confusion.txt"
    confusion_matrix = CM.construct_confusion()
    CM.print_conf_matrix(fname, confusion_matrix)

def test_calc_accuracy(CM):
    print("\tTesting calc_accuracy()...")
    accuracy = CM.calc_accuracy(CM.confusion_matrix)
    fname = DEFAULT_TEST_PATH + "test_accuracy.txt"
    test_f = open(fname, 'w')
    print("Accuracy:{}".format(accuracy), file=test_f)
    test_f.close()

    
##==-- Main --==##
CM = ConfusionMatrix(20)
test_construct_confusion(CM)
test_calc_accuracy(CM)
