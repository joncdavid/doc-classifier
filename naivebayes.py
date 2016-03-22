#!/usr/env python3
# filename: naivebayes.py
# authors: Jon David and Jarrett Decker
# description:
#   Naive Bayes Classifier and related functions
#


from collections import *

class NaiveBayesClassifier:
    def __init__(self, vocab_obj, newsgroups_obj):
        self.vocab = vocab_obj
        self.newsgroups = newsgroups_obj

        self.label_histogram = defaultdict(int)
        self.num_documents = 0
        self.mle_dict = defaultdict(float)

        return

    def get_MLE(self, doclabel):
        """Get MLE for P(Y), where Y=doclabel."""
        return self.mle_dict[doclabel]

    def calc_all_MLE(self, labelfile):
        f = open(labelfile, 'r')
        self.num_documents = 0
        for line in f.readlines():
            label_id = line.strip()
            label = self.newsgroups.id_to_group_dict[label_id]
            self.mle_dict[label] += 1
            self.num_documents += 1

        for label, count in self.label_histogram.items():
            self.mle_dict[label] = (float)count / self.num_documents

    def get_MAP(self, word, doclabel):
        """Get MAP for P(X_i|Y_k), where X_i in Vocabulary, and Y_k in NewsGroups."""
        return self.map_dict[X_i][Y_k]
