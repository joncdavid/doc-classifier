#!/usr/env python3
# filename: test_train.py
# authors: Jon David and Jarrett Decker
# description:
#   Trains a model using training data files.
#

from collections import *

from document import *
from vocabulary import *
from newsgroups import *
from filereaders import *


class ModelTrainer:
    DEFAULT_DATA_FILE = "./data/train.data"
    DEFAULT_LABEL_FILE = "./data/train.label"
    DEFAULT_VOCAB_FILE = "./data/vocabulary.txt"
    DEFAULT_NEWSGROUPS_FILE = "./data/newsgrouplabels.txt"
    
    def __init__(self, datafile=DEFAULT_DATA_FILE,
                 labelfile=DEFAULT_LABEL_FILE,
                 vocabfile=DEFAULT_VOCAB_FILE,
                 newsgroupsfile=DEFAULT_NEWSGROUPS_FILE):
        """Initializes an instance of ModelTrainer."""
        self.datafile = datafile
        self.labelfile = labelfile
        self.vocabfile = vocabfile
        self.newsgroupsfile = newsgroupsfile

        self.vocabulary = None
        self.newsgroups = None
        self.doc_dict = self._loadlabels(self.labelfile)
        self.raw_data = self._loaddata(self.datafile)
        
    def _loadlabels(self, labelfilename):
        """Reads label file and returns dictionary of
        (document_id, document) pairs."""
        reader = LabelsReader(labelfilename)
        return reader.read(reader.filename)

    def _loaddata(self, datafilename):
        """Reads from data file and loads data into
        appropriate document."""
        f = open(filename, 'r')
        nparray = loadtxt(f, dtype=int)
        return nparray

    def _load_data_into_docs(self, raw_data, doc_dict):
        
