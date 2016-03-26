#!/usr/env python3
# filename: ranking.py
# authors: Jon David and Jarrett Decker
# description:
#   Uses model to rank words.
#

from collections import *
import vocabulary as vb
import newsgroups as ng
import numpy as np

class WordRanker:
    DEFAULT_MLE_FILENAME = "./models/mle.model"
    DEFAULT_MAP_FILENAME = "./models/map.model"
    DEFAULT_EVIDENCE_FILENAME = "./models/evidence.model"

    def __init__(self,
                 mlefilename=DEFAULT_MLE_FILENAME,
                 mapfilename=DEFAULT_MAP_FILENAME,
                 evidencefilename=DEFAULT_EVIDENCE_FILENAME):
        """Initializes this WordRanker."""
        self.vocab = vb.Vocabulary()
        self.mlefilename = mlefilename
        self.mapfilename = mapfilename
        self.evidencefilename = evidencefilename
        self.MLE_vec = np.loadtxt(self.mlefilename)
        self.MAP_matrix = np.loadtxt(self.mapfilename)
        self.EVIDENCE_vec = np.loadtxt(self.evidencefilename)

    def _calc_score(self, MLE_vec, MAP_matrix, EVIDENCE_vec):
        """Calculates score for each word."""
        MLE_vec_T = MLE_vec.T
        MAP_matrix_T = MAP_matrix.T
        score_vec = MLE_vec.dot(MAP_matrix_T)  #/EVIDENCE_vec
        return score_vec

    def _sort_by_score(self, x):
        """Used to sort a list of tuples by the 3rd element."""
        return x[2]

    def rank(self):
        """Ranks all words, returns a score list."""
        score_vec = self._calc_score(self.MLE_vec, self.MAP_matrix,
                                     self.EVIDENCE_vec)
        score_list = self._rank(score_vec)
        return score_list
            
    def _rank(self, score_vec):
        """Ranks all words, returns a score list."""
        score_list = [] #each item is a (word_id, score) pair.
        for i in range(0,score_vec.size):
            w_id = i+1
            score_list.append((w_id, self.vocab.get_word(w_id), score_vec[i]))
        score_list.sort(key=self._sort_by_score, reverse=True)
        return score_list

    def top_100(self, score_list):
        score_list.sort(key=self._sort_by_score, reverse=True)
        return score_list[:100]
    
