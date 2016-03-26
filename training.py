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
    DEFAULT_MLE_FILENAME = "./models/mle.model"
    DEFAULT_MAP_FILENAME = "./models/map.model"
    DEFAULT_EVIDENCE_FILENAME = "./models/evidence.model"

    def __init__(self,
                 datafile=DEFAULT_DATA_FILE,
                 labelfile=DEFAULT_LABEL_FILE,
                 stop_words=False):
        self.datafile = datafile
        self.labelfile = labelfile
        self.vocab = vb.Vocabulary(stop_words=stop_words)
        self.newsgroups = ng.NewsGroups()
        self.num_docs = 0
        self.label_hist = defaultdict(int) #counts num docs labeled for each label.
        
        vocabsize = self.vocab.size
        newsgroupssize = self.newsgroups.size
        self.dataarray = np.zeros((newsgroupssize, vocabsize), dtype=int)
        self.labeldict = defaultdict()
        for i in range(1,21):
            self.labeldict[i] = 0

    def train(self):
        """The single function to rule them all."""
        MLE_vec, MAP_matrix, EVIDENCE_vec = self.generate_model()
        self.save_model(MLE_vec, MAP_matrix, EVIDENCE_vec)

    def calc_vector_EVIDENCE(self):
        """Calculates Evidence for P(X),
        for all X in Vocabulary, and returns a vector,
        whose index represents vocabulary_id-1."""
        evi_vec = np.zeros(self.vocab.size)
        with open(self.datafile, 'r') as dataf:
            for line in dataf:
                data = line.strip().split()
                word_id = int(data[1])
                word_count = int(data[2])
                evi_vec[int(word_id)-1] += word_count
        evi_vec = evi_vec / sum(evi_vec)
        return evi_vec

        
    def calc_vector_MLE(self):
        """Calculates MLE for P(Y_k),
        for all Y_k in NewsGroups, and returns a vector,
        whose index represents newsgroup_id-1."""
        #P(Y_k) = (# of docs labeled Y_k) /
        #         (total # of docs)
        mle_vec = np.zeros(self.newsgroups.size)
        doc_id = 1
        with open(self.labelfile, 'r') as labelf:
            for line in labelf:
                label = line.strip().lower()
                mle_vec[int(label)-1] += 1

                #sets label for doc_id in labeldict
                self.labeldict[doc_id] = label
                doc_id += 1
        mle_vec = mle_vec / sum(mle_vec)
        return mle_vec

    def calc_matrix_MAP(self, beta=None):
        """Calculates P(X_i|Y_k),
        for all X_i in Y_k, for all Y_k in NewsGroups,
        and returns a matrix whose indices are
        newsgroup_id, word_id."""
        #P(X_i|Y_k) =
        #  (count of X_i in Y_k) + (alpha-1) /
        #  (total words in Y_k) + ((alpha-1)*|V|)
        if not beta:
            beta = float(1.0 / self.vocab.size)
        alpha = 1 + beta
        gamma = alpha - 1

        input_matrix = np.zeros((self.vocab.size,
                                 self.newsgroups.size))
        with open(self.datafile, 'r') as dataf:
            for line in dataf:
                data = line.strip().split()
                doc_id = int(data[0])
                word_id = int(data[1])
                w_count = int(data[2])

                label_id = int(self.labeldict[doc_id])
                input_matrix[word_id-1, label_id-1] += w_count

        total_words = np.zeros(self.newsgroups.size)
        for j in range(0, self.newsgroups.size):
            total_words[j] = sum(input_matrix[:,j])
            
        map_matrix = np.zeros((self.vocab.size,
                               self.newsgroups.size))
        for i in range(0, self.vocab.size):
            for j in range(0, self.newsgroups.size):
                map_matrix[i,j] = \
                  ( input_matrix[i,j] + gamma ) / \
                  ( total_words[j] + (gamma*self.vocab.size) )
        for word_id in self.vocab.stop_words_id_list:
            for j in range(0, self.newsgroups.size):
                map_matrix[word_id-1][j] = 0.00000000000001

        return map_matrix

    def generate_model(self, betavalue=None):
        """Generates model as MLE_vec, and MAP_matrix."""
        MLE_vec = self.calc_vector_MLE()
        MAP_matrix = self.calc_matrix_MAP(betavalue)
        EVIDENCE_vec = self.calc_vector_EVIDENCE()
        return MLE_vec, MAP_matrix, EVIDENCE_vec

    def save_model(self, MLE_vector, MAP_matrix, EVIDENCE_vector,
                   mlefilename=DEFAULT_MLE_FILENAME,
                   mapfilename=DEFAULT_MAP_FILENAME,
                   evidencefilename=DEFAULT_EVIDENCE_FILENAME):
        """Saves model in savefilename."""
        np.savetxt(mlefilename, MLE_vector)
        np.savetxt(mapfilename, MAP_matrix)
        np.savetxt(evidencefilename, EVIDENCE_vector)

    def get_word_ranking(self, MAP_matrix, MLE_matrix):
        """Rank words in the vocabulary based off of information gain"""

        """first remove stopwords from the training data"""
        stopwords = open("./data/stopwords.txt", "r")
        word = stopwords.readline().strip()
        while(word != ""):
            for i in range (0, self.vocab.size):
                if word == self.vocab.get_word(i+1):
                    for j in range(0, self.newsgroups.size):
                        MAP_matrix[i][j] = 0
            word = stopwords.readline().strip()

        """Set up all the various values of probabilities we need"""
        P_XY_ = np.zeros((self.vocab.size, self.newsgroups.size), dtype=float)
        for i in range(0, self.vocab.size):
            for j in range(0, self.newsgroups.size):
                P_XY_[i][j] = MAP_matrix[i][j] * MLE_matrix[j]
        P_Y_ = MLE_matrix
        #print("P_XY_ shape:{}".format(P_XY_.shape))
        #print("P_Y_ shape:{}".format(P_Y_.shape))
        P_X_ = np.zeros(self.vocab.size, dtype=float)
        P_notX_ = np.zeros(self.vocab.size, dtype=float)
        P_notXY_ = np.zeros((self.vocab.size, self.newsgroups.size), dtype=float)
        for i in range(0, self.vocab.size):
            P_X_[i] = sum(P_XY_[i,:])
            P_notX_[i] = 1 - P_X_[i]
            for j in range(0, self.newsgroups.size):
                P_notXY_[i][j] = MLE_matrix[j] * (1 - MAP_matrix[i][j])
        #print("P_X_ shape:{}".format(P_X_.shape))
        #print("P_notXY_ shape:{}".format(P_notXY_.shape))
        #print("P_notX_ shape:{}".format(P_notX_.shape))

        """Calculate the information gain of each of the words"""
        I_X_ = np.zeros(self.vocab.size, dtype=float)
        for i in range(0, self.vocab.size):
            tmp1 = 0
            tmp2 = 0
            tmp3 = 0
            for j in range(0, self.newsgroups.size):
                if(P_XY_[i][j] == 0 or P_X_[i] == 0 or P_notXY_[i][j] == 0 or P_notX_[i] == 0):
                    tmp1 = 0
                    tmp2 = 0
                    tmp3 = 0
                    continue
                tmp1 += P_Y_[j] * np.log2(P_Y_[j])
                tmp2 += (P_XY_[i][j] / P_X_[i]) * np.log2(P_XY_[i][j] / P_X_[i])
                tmp3 += (P_notXY_[i][j] / P_notX_[i]) * np.log2(P_notXY_[i][j] / P_notX_[i])
            I_X_[i] = -tmp1 + P_X_[i] * tmp2 + P_notX_[i] * tmp3
        #print("I_X_ shape:{}".format(I_X_.shape))

        """Write the word ranking list to file, starting with the max value"""
        output_file = open("./model_results/top_words.txt", "w")
        for x in range(0, 100):
            # print(self.vocab.get_word(np.argmax(I_X_)+1))
            print(self.vocab.get_word(np.argmax(I_X_)+1), file=output_file)
            I_X_[np.argmax(I_X_)] = -9999999999999
