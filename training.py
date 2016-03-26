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

    def __init__(self,
                 datafile=DEFAULT_DATA_FILE,
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
        for i in range(1,21):
            self.labeldict[i] = 0



    #def getlabels(self):
    #    """Reads labels file and stores in labeldict."""
    #    with open(self.labelfile, 'r') as labelf:
    #        linenum = 1
    #        for line in labelf:
    #            label = line.strip().lower()
    #            self.labeldict[linenum] = label
    #            linenum += 1

    #def getdata(self):
    #    """Reads data file and stores in dataarray."""
    #    self.getlabels()
    #    with open(self.datafile, 'r') as dataf:
    #        for line in dataf:
    #            data = line.split()
    #            doc_id = int(data[0])
    #            word_id = int(data[1])
    #            word_count = int(data[2])
                
    #            doc_label_id = int(self.labeldict[doc_id])
    #            self.num_docs += 1
    #            self.label_hist[doc_label_id] += 1
    #            self.dataarray[doc_label_id-1, word_id-1] = word_count
                
    def train(self):
        """The single function to rule them all."""
        #self.getdata()
        MLE_vec, MAP_matrix = self.generate_model()
        self.save_model(MLE_vec, MAP_matrix)

    def calc_vector_MLE(self):
        """Calculates MLE for P(Y_k),
        for all Y_k in NewsGroups, and returns a vector,
        whose index represents newsgroup_id."""
        #P(Y_k) = (# of docs labeled Y_k) /
        #         (total # of docs)
        #D: why couldn't the input indices start with 0...
        # mle_vec = np.zeros(self.newsgroups.size)
        # for ng_id in range(1, self.newsgroups.size+1):
        #     mle_vec[ng_id-1] = self.label_hist[ng_id]
        # total_doc_count = sum(self.label_hist.values())
        # mle_vec = mle_vec / total_doc_count
        # return mle_vec

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

        return map_matrix

    def generate_model(self, betavalue=None):
        """Generates model as MLE_vec, and MAP_matrix."""
        MLE_vec = self.calc_vector_MLE()
        MAP_matrix = self.calc_matrix_MAP(betavalue)
        return MLE_vec, MAP_matrix

    def save_model(self, MLE_vector, MAP_matrix,
                   mlefilename=DEFAULT_MLE_FILENAME,
                   mapfilename=DEFAULT_MAP_FILENAME):
        """Saves model in savefilename."""
        np.savetxt(mlefilename, MLE_vector)
        np.savetxt(mapfilename, MAP_matrix)


    def get_word_ranking(self, MAP_matrix):
        rank = np.zeros((self.vocab.size, self.newsgroups.size),  dtype=float)
        top_ranks = np.zeros((100, 2), dtype=float)
        for i in range(0, self.vocab.size):
            total = sum(MAP_matrix[i,:])
            # print("total = ", total)
            for j in range(0, self.newsgroups.size):
                rank[i][j] = float(MAP_matrix[i][j]) / total
                # print("rank[",i,"][",j,"] = ", rank[i][j], "MAP[",i,"][",j,"] = ",MAP_matrix[i][j])
                if rank[i,j] > np.min(top_ranks[:,0]):
                    k = np.argmin(top_ranks[:,0])
                    # print("Old TR[",k,"][0] = ", top_ranks[k][0]," TR[",k,"][1] = ", top_ranks[k][1] )
                    top_ranks[k][1] = i
                    top_ranks[k][0] = rank[i][j]
                    # print("New TR[",k,"][0] = ", top_ranks[k][0]," TR[",k,"][1] = ", top_ranks[k][1] )

        # print(top_ranks)
        words = []
        for h in range (0, 100):
            word = self.vocab.get_word(top_ranks[h][1])
            words.append(word)
        # print(words)
        return words

