#!/usr/env python3
# filename: filereaders.py
# authors: Jon David and Jarrett Decker
# description:
#   Loads vocabulary from text file.
#

from collections import *

from vocabulary import *
from document import *

class VocabularyReader:
    def __init__(self, filename):
        self.filename = filename
        self.vocabulary = Vocabulary(filename)

class NewsGroups:
    def __init__(self, filename):
        self.filename = filename
        self.newsgroups = NewsGroups(filename)
        
class DataReader:
    DEFAULT_DATA_FILE = "./data/train.data"

    def __init__(self, filename=None):
        self.filename = filename
        if not self.filename:
            self.filename = self.DEFAULT_DATA_FILE
        read(filename)

    def read(self, filename):
        f = open(filename, 'r')
        nparray = loadtxt(f, dtype=int)
        return nparray
            

class LabelsReader:
    DEFAULT_LABELS_FILE = "./data/train.label"

    def __init__(self, filename=DEFAULT_LABELS_FILE):
        """Initializes LabelsReader"""
        self.filename = filename
        read(filename)

    def read(self, filename):
        """Reads labels file and returns a dictionary of
        (docment_id, document) pairs."""
        f = open(filename, 'r')
        doc_dict = defaultdict(Document())
        document_id = 1
        for line in f:
            label = line.strip().lower()
            doc = Document(document_id, label)
            doc_dict.update({document_id : label})
            document_id += 1
        return doc_dict
