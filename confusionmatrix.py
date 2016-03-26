#!/usr/env python3
# filename: confusionmatrix.py
# authors: Jon David and Jarrett Decker
# description:
#   Representation of a confusion matrix.
#

# from filereaders import *
from collections import *

from training import *
import vocabulary as vb
import newsgroups as ng
import numpy as np


class ConfusionMatrix(object):
    DEFAULT_LABEL_FILE = "./data/test.label"
    DEFAULT_PRED_LABEL_FILE = "./test_output/test_classify.txt"

    def __init__(self,
                 num_labels,
                 labelfile=DEFAULT_LABEL_FILE,
                 predictionfile=DEFAULT_PRED_LABEL_FILE):
        """Initializes this NaiveBayesClassifier."""
        self.num_labels = num_labels
        self.labelfile = labelfile
        self.predictionfile = predictionfile
        self.confusion_matrix = self.construct_confusion(self.labelfile,
                                                         self.predictionfile)
        self.accuracy = self.calc_accuracy(self.confusion_matrix)
        
    def construct_confusion(self,
                            labels_filename=DEFAULT_LABEL_FILE,
                            predictions_filename=DEFAULT_PRED_LABEL_FILE):
        """ Creates matrix containing predicted labels vs true labels"""
        num_labels = self.num_labels
        true_label_f = open(labels_filename, 'r')
        pred_label_f = open(predictions_filename, 'r')
        conf_matrix = np.zeros((num_labels,num_labels), dtype=int)

        t_l = true_label_f.readline().strip()
        p_l = pred_label_f.readline().strip()
        line_num = 1
        while t_l != '':
            conf_matrix[int(t_l) - 1][int(p_l) - 1] += 1
            t_l = true_label_f.readline().strip()
            p_l = pred_label_f.readline().strip()

        return conf_matrix

    def calc_accuracy(self, confusion_matrix):
        """Calculates accuracy."""
        i = len(confusion_matrix)
        correct=0
        total=0
        for x in range(0, i):
            for y in range(0, i):
                total += confusion_matrix[x][y]
                if x == y:
                    correct += confusion_matrix[x][y]
        return float(correct) / total

    def print_conf_matrix(self, outfilename, conf_matrix):
        """Prints the confusion to the outfile."""
        output_file = open(outfilename, "w")
        n = self.num_labels
        for i in range(0, n):
            for j in range(0, n):
                print("\t{}".format(conf_matrix[i][j]),
                      file=output_file, end="")
            print(file=output_file)
        output_file.close()
        
