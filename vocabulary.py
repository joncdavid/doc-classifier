#!/usr/env python3
# filename: vocabulary.py
# authors: Jon David and Jarrett Decker
# description:
#   Loads vocabulary from text file.
#

class Vocabulary:
    DEFAULT_VOCAB_FILE = "./data/vocabulary.txt"

    def __init__(self, filename=None, stop_words=False):
        """Initializes DocVocabulary."""
        self.filename = filename
        self.stop_words = stop_words
        if not self.filename:
            self.filename = self.DEFAULT_VOCAB_FILE
        self.word_to_id_dict =  {}
        self.id_to_word_dict = {}
        self.size = 0
        self.stop_words_list = []
        self.stop_words_id_list = []
        self.load()

    def load(self, filename=None):
        """Loads vocabulary from file."""
        if not filename:
            filename = self.DEFAULT_VOCAB_FILE
        f = open(filename, 'r')
        if self.stop_words:
            swf = open("./data/stopwords.txt", "r")
            word = swf.readline().strip()
            while(word != ""):
                self.stop_words_list.append(word)
                word = swf.readline().strip()
        word_id = 1
        for line in f:
            word = line.strip().lower()
            if self.stop_words:
                if word in self.stop_words_list:
                    self.stop_words_id_list.append(word_id)
            self.id_to_word_dict[word_id] = word
            self.word_to_id_dict[word] = word_id
            word_id += 1
        self.size = len(self.id_to_word_dict)

    def get_id(self, word):
        """Returns the word_id associated with word."""
        return self.word_to_id_dict[word]

    def get_word(self, word_id):
        """Returns the word associated with word_id."""
        return self.id_to_word_dict[word_id]
