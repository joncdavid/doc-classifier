#!/usr/env python3
# filename: classifier.py
# authors: Jon David and Jarrett Decker
# description:
#   Reads test data and generates predictions.
#

# from filereaders import *
from collections import *

from training import *
import vocabulary as vb
import newsgroups as ng
import numpy as np


class NaiveBayesClassifier(object):
    """Naives Bayes Classifier."""
    DEFAULT_DATA_FILE = "./data/test.data"
    DEFAULT_LABEL_FILE = "./data/test.label"
    DEFAULT_MLE_FILENAME = "./models/mle.model"
    DEFAULT_MAP_FILENAME = "./models/map.model"

    def __init__(self,
                 datafile=DEFAULT_DATA_FILE,
                 labelfile=DEFAULT_LABEL_FILE):
        """Initializes this NaiveBayesClassifier."""
        self.trainer = Trainer(datafile, labelfile)
        self.trainer.getdata()
        self.word_hist = defaultdict(int)
        self.word_hist = self.populate_word_histogram()
        self.word_vec = \
          self.const_word_vector(self.word_hist)
        self.MLE_vec = None
        self.MAP_matrix = None
        self.load_models()

    def populate_word_histogram(self):
        """Populates word histogram from dataarray."""
        num_rows, num_cols = self.trainer.dataarray.shape
        for w_col in range(0, num_cols):
            count = sum( self.trainer.dataarray[:,w_col] )
            word_id = w_col+1
            self.word_hist[word_id] += count
        return self.word_hist

    def const_word_vector(self, word_histogram):
        """Constructs a vector of word counts, where
        the index represents the word_id."""
        num_words = self.trainer.vocab.size
        word_vec = np.zeros(num_words)
        for word_id in range(1,num_words):
            word_vec[word_id-1] = word_histogram[word_id]
        return word_vec

    def load_models(self,
                    mle_file=DEFAULT_MLE_FILENAME,
                    map_file=DEFAULT_MAP_FILENAME):
        """Loads models from saved file."""
        self.MLE_vec = np.loadtxt(mle_file)
        self.MAP_matrix = np.loadtxt(map_file)
        
    def classify(self, word_vector, mle_vec, map_matrix):
        #################################################
        #################################################
        # Current implementation is WRONG
        # We need to produce a vector of predictions,
        # where the vector's index represents the
        # test document's ID.
        #------------------------------------------------
        # The error is with word_vector.
        # Currently word_vector is a count for all dataarray
        # But it should be a count for only a single doc.
        #################################################
        #################################################
        """Classifies a given document based only on
        its word histogram."""
        #Y_prediction =
        #  argmax( log2(P(Y_k)) +
        #          sum_i(# of X_i) * log2(P(X_i|Y_k)) ).
        d = word_vector + mle_vec.times(map_matrix)
        predicted_newsgroup_id = d.argmax(d)+1
        return predicted_newsgroup_id

