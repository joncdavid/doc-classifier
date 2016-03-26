#!/usr/env python3
# filename: stopwords.py
# authors: Jon David and Jarrett Decker
# description:
#   Loads list of stop words from file.
#

from collections import *

class StopWords:
    DEFAULT_STOPWORDS_FILE = "./data/stopwords.txt"

    def __init__(self, filename=DEFAULT_STOPWORDS_FILE):
        """Initializes StopWords"""
        self.filename = filename
        self.stop_dict = self.load(self.filename)
        self.size = len(self.stop_dict)

    def load(self, filename=DEFAULT_STOPWORDS_FILE):
        """Loads vocabulary from file."""
        stop_dict = defaultdict(int)
        f = open(filename, 'r')
        for line in f:
            word = line.strip().lower()
            stop_dict[word] = 0 #dummy assignment
            
