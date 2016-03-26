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

    def train(self):
        """The single function to rule them all."""
        MLE_vec, MAP_matrix = self.generate_model()
        self.save_model(MLE_vec, MAP_matrix)

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
        rank = np.zeros((self.vocab.size, self.newsgroups.size),  dtype=float)
        top_ranks = np.zeros((100, 2), dtype=float)
        top_ranks.fill(-9999999999999)
        # top_ranks = {}
        for i in range(0, self.vocab.size):

            total = 0
            total = np.sum(MAP_matrix[i,:])
            # print("total = ", total)
            for j in range(0, self.newsgroups.size):
                # rank[i][j] = MAP_matrix[i][j] * (float(MAP_matrix[i][j]) / total)
                rank[i][j] = MAP_matrix[i][j] / total
                # rank[i][j] = -np.log2(float(MAP_matrix[i][j]))
                # rank[i][j] = -(float(MAP_matrix[i][j]) / total) * np.log2(float(MAP_matrix[i][j]) / total)
                # rank[i][j] = MAP_matrix[i][j] * -np.log2(float(MAP_matrix[i][j]) / total)
                # rank[i][j] = -np.log2(MLE_matrix[j] * (float(MAP_matrix[i][j]) / total))
                # print("rank[",i,"][",j,"] = ", rank[i][j], "MAP[",i,"][",j,"] = ",MAP_matrix[i][j])
                # if sum(rank[i,:]) > np.min(top_ranks[:,0]):
                # total += MAP_matrix[i][j] * MLE_matrix[j]
                # print(MAP_matrix[i][j] * MLE_matrix[j], MAP_matrix[i][j], MLE_matrix[j])
            # for j in range(0, self.newsgroups.size):
                # rank[i][j] = MAP_matrix[i][j] * MLE_matrix[j] / total
            if np.max(rank[i,:]) > np.min(top_ranks[:,0]):
            # if sum(rank[i,:]) > np.min(top_ranks[:,0]):

                # if rank[i][j] > np.min(top_ranks[:,0]):
                k = np.argmin(top_ranks[:,0])
                    # print("Old TR[",k,"][0] = ", top_ranks[k][0]," TR[",k,"][1] = ", top_ranks[k][1] )
                top_ranks[k][1] = i
                # top_ranks[k][0] = sum(rank[i,:])
                top_ranks[k][0] = np.max(rank[i,:])
                    # top_ranks[k][0] = rank[i][j]

                    # print("New TR[",k,"][0] = ", top_ranks[k][0]," TR[",k,"][1] = ", top_ranks[k][1] )
                # top_ranks[i] = np.max(rank[i,:])

        # top_ranks = np.sort(top_ranks,0)
        print(top_ranks)
        words = []
        output_file = open("./model_results/top_words.txt", "w")

        for h in range(0, 100):
            word = self.vocab.get_word(top_ranks[99-h][1])
            print(word, file=output_file)
            words.append(word)





        print(words)

        return words

