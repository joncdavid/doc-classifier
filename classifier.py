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
                 labelfile=DEFAULT_LABEL_FILE,
                 mlefile=DEFAULT_MLE_FILENAME,
                 mapfile=DEFAULT_MAP_FILENAME):
        """Initializes this NaiveBayesClassifier."""
        self.vocab = vb.Vocabulary()
        self.newsgroups = ng.NewsGroups()
        self.num_docs = 0
        self.datafile = datafile
        self.labelfile= labelfile
        self.mlefile = mlefile
        self.mapfile = mapfile
        self.word_hist = defaultdict(int)
        self.MLE_vec = None
        self.MAP_matrix = None
        self.load_models(self.mlefile, self.mapfile)

    def const_word_histograms(self):
        """Constructs word histogram from datafile."""
        #doc_word_dict is a dictionary of
        # (doc_id, word_histogram) pairs.
        f = open(self.datafile, 'r')
        doc_word_dict = {}
        for line in f:
            doc_id_str, word_id_str, count_str = line.strip().split()
            doc_id = int(doc_id_str)
            word_id = int(word_id_str)
            count = int(count_str)
            if doc_id not in doc_word_dict:
                doc_word_dict[doc_id] = defaultdict(int)
            word_hist = doc_word_dict[doc_id]
            word_hist[word_id] += count
        self.num_docs = len(doc_word_dict)
        return doc_word_dict

    def const_word_vectors(self, word_histograms):
        """Constructs a vector of word counts, where
        the index represents the word_id.
        Returns a dictionary of word vectors,
        whose key is the doc_id."""
        doc_word_vec_dict = {}
        for doc_id in range(1, self.num_docs+1):
            w_histogram = word_histograms[doc_id]
            v = np.zeros(self.vocab.size)
            for word_id in range(1, self.vocab.size+1):
                if word_id in w_histogram:
                    v[word_id-1] = w_histogram[word_id]
            doc_word_vec_dict[doc_id] = v

        #doc_word_vec_dict is a dictionary whose keys,values
        # are (doc_id, word_vector).
        #A word_vector is a 1D array of counts per word,
        # where the index represents the word_id-1..
        return doc_word_vec_dict
            
    def load_models(self,
                    mle_file=DEFAULT_MLE_FILENAME,
                    map_file=DEFAULT_MAP_FILENAME):
        """Loads models from saved file."""
        self.MLE_vec = np.loadtxt(mle_file)
        self.MAP_matrix = np.loadtxt(map_file)

    def classify(self, mlemodelfile, mapmodelfile):
        self.load_models(mlemodelfile, mapmodelfile)
        mle_vec = self.MLE_vec
        map_matrix = self.MAP_matrix
        word_hist = self.const_word_histograms()
        word_vec_dict = self.const_word_vectors(word_hist)
        return self._classify(word_vec_dict, mle_vec, map_matrix)
        
    def _classify(self, word_vector_dict, mle_vec, map_matrix):
        """Classifies documents based only on models, and the
        its input word histograms. Returns an array of
        predicted labels (i.e., NewsGroupIDs)"""
        predictions = np.zeros(self.num_docs)
        for doc_id in range(1, self.num_docs+1):
            word_vector = word_vector_dict[doc_id]
            a = mle_vec.T
            b = word_vector.T
            c = map_matrix

            a_log2 = np.log2(a)
            c_log2 = np.log2(c)
            d = a_log2 + b.dot(c_log2)

            predicted_newsgroup_id = d.argmax()+1
            predictions[doc_id-1] = predicted_newsgroup_id
        return predictions

    def confusionmatrix(self,
                        true_label_file=DEFAULT_LABEL_FILE,
                        pred_label_file = "./test_output/test_classify.txt"):
        """ Creates matrix containing predicted labels vs true labels"""
        true_label = open(true_label_file, 'r')
        pred_label = open(pred_label_file, 'r')
        confusion_matrix = np.zeros((self.newsgroups.size,
                                     self.newsgroups.size), dtype=int)
        
        t_l = true_label.readline().strip()
        p_l = pred_label.readline().strip()
        while t_l != '':
            confusion_matrix[int(t_l) - 1][int(p_l) - 1] += 1
            t_l = true_label.readline().strip()
            p_l = pred_label.readline().strip()
        return confusion_matrix

    def accuracy(self, confusion_matrix):
        """Calculates the accuracy of this model."""
        i = len(confusion_matrix)
        correct = 0
        total = 0
        for x in range(0, i):
            for y in range(0, i):
                total += confusion_matrix[x][y]
                if x == y:
                    correct += confusion_matrix[x][y]
        return float(correct / total)
