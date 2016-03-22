#!/usr/env python3
# filename: document.py
# authors: Jon David and Jarrett Decker
# description:
#   Loads document from text file.
#

from collections import *


class Document:
    def __init__(self, ID, label):
        self.ID = ID
        self.label = label

        #<key> is word_id, <value> is number of times
        # those words appear in this document.
        self.word_histogram = defaultdict(int)
        self.total_words = self._count_words()

    def _count_words():
        count = 0
        for k,v in self.word_histogram.items():
            count += v
        return count
    
    def add_word(word_id, count):
        self.word_histogram[word_id] = count    
