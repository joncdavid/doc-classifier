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

        vocabsize = self.vocab.size
        newsgroupssize = self.newsgroups.size

        self.dataarray = np.zeros((newsgroupssize, vocabsize), dtype=int)
        self.labeldict = defaultdict()

    def getlabels(self):
        with open(self.labelfile, 'r') as labelf:
            linenum = 1
            # (linenum)
            for line in labelf:
                label = line.strip().lower()
                # print(label)
                self.labeldict[linenum] = label
                # print(linenum,  " = " + self.labeldict[linenum])
                linenum += 1

    def getdata(self):
        self.getlabels()

        # for key in self.labeldict:
        #     print(key, " key = " + self.labeldict[key])


        with open(self.datafile, 'r') as dataf:
            for line in dataf:
                doc_id = 0
                doc_label = 0
                word_id = 0
                word_count = 0

                data = line.split()

                doc_id = int(data[0])


                word_id = int(data[1])
                word_count = int(data[2])
                # print("doc_id = " + doc_id, " word_id = " + word_id, " word_count = " + word_count )

                doc_label = int(self.labeldict[doc_id])
                self.dataarray[doc_label - 1, word_id - 1] = word_count

    def train(self):

        self.getdata()