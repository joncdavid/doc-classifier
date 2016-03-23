#!/usr/env python3
# filename: training.py
# authors: Jon David and Jarrett Decker
# description:
#   Reads training data and generates nparray used in classification
#

# from filereaders import *
from collections import *
import vocabulary as vb
import newsgroups as ng
import numpy as np

class Trainer:
    DEFAULT_DATA_FILE = "./data/train.data"
    DEFAULT_LABEL_FILE = "./data/train.label"

    def __init__(self, datafile=DEFAULT_DATA_FILE,
                 labelfile=DEFAULT_LABEL_FILE):
        self.datafile = datafile
        self.labelfile = labelfile
        self.vocab = vb.Vocabulary()
        self.newsgroups = ng.NewsGroups()
        self.num_docs = 0
        self.label_hist = defaultdict(int) #counts num docs labeled for each label.
        
        vocabsize = self.vocab.size
        newsgroupssize = self.newsgroups.size
        self.dataarray = np.zeros((newsgroupssize, vocabsize), dtype=int)
        self.labeldict = defaultdict()

    def getlabels(self):
        """Reads labels file and stores in labeldict."""
        with open(self.labelfile, 'r') as labelf:
            linenum = 1
            for line in labelf:
                label = line.strip().lower()
                self.labeldict[linenum] = label
                linenum += 1

    def getdata(self):
        """Reads data file and stores in dataarray."""
        self.getlabels()
        with open(self.datafile, 'r') as dataf:
            for line in dataf:
                data = line.split()
                doc_id = int(data[0])
                word_id = int(data[1])
                word_count = int(data[2])
                
                doc_label_id = int(self.labeldict[doc_id])
                self.num_docs += 1
                self.label_hist[doc_label_id] += 1                
                self.dataarray[doc_label_id-1, word_id-1] = word_count

    def train(self):
        self.getdata()
        MLE_vec = self.calc_vector_MLE()
        MAP_matrix = self.calc_matrix_MAP()
        self.generate_model()
        self.save_model("bayesclassifier.model")

    def calc_vector_MLE(self):
        """Calculates MLE for P(Y_k),
        for all Y_k in NewsGroups, and returns a vector,
        whose index represents newsgroup_id."""
        #P(Y_k) = (# of docs labeled Y_k) /
        #         (total # of docs)
        #D: why couldn't the input indices start with 0...
        mle_vec = np.zeros(self.newsgroups.size)
        total_count = 0
        for ng_id in range(1, self.newsgroups.size+1):
            count = self.label_hist[ng_id]
            mle_vec[ng_id-1] = float(count)/self.num_docs
            total_count += count

        return mle_vec

    def calc_matrix_MAP(self):
        """Calculates P(X_i|Y_k),
        for all X_i in Y_k, for all Y_k in NewsGroups,
        and returns a matrix whose indices are
        newsgroup_id, word_id."""
        #P(X_i|Y_k) =
        #  (count of X_i in Y_k) + (alpha-1) /
        #  (total words in Y_k) + ((alpha-1)*|V|)
        
        beta = 1 * self.vocab.size
        alpha = 1 + beta
        gamma = alpha - 1
        map_matrix = np.zeros((self.vocab.size,
                               self.newsgroups.size))
        for ng_id in range(1, self.newsgroups.size+1):
            total_words = sum(self.dataarray[ng_id-1,:])
            for word_id in range(1, self.vocab.size+1):
                count = self.dataarray[ng_id-1, word_id-1]
                P = float(count+gamma) / \
                    (total_words + gamma*self.vocab.size)
                map_matrix[word_id-1, ng_id-1] = P
                
        return map_matrix

    def generate_model(self):
        """Generates model as MLE_vec, and MAP_matrix."""
        return 0

    def save_model(self, savefilename):
        """Saves model in savefilename."""
        return 0

    #note: create separate class called classifier
    #class NBClassifier:
    #  def classify(self, X_new):
    #    return argmax(log(MLE_vec) + X_new*MAP_matrix)