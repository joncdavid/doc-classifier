#!/usr/env python3
# filename: test_classifier.py
# authors: Jon David and Jarrett Decker
from classifier import *

DEFAULT_TEST_PATH = "./test_output/"

def test_populate_word_histogram():
    print("\tTesting populate_word_histogram()...")
    c = NaiveBayesClassifier()
    word_hist = c.populate_word_histogram()

    fname = DEFAULT_TEST_PATH + "test_pop_word_hist.txt"
    test_f = open(fname, 'w')
    print_str = "{}, \nlen:{}"
    print(print_str.format(word_hist,
                           len(word_hist)),
          file=test_f)
    
def test_const_word_vector():
    print("\tTesting const_word_vector()...")
    c = NaiveBayesClassifier()
    word_hist = c.populate_word_histogram()
    word_vec = c.const_word_vector(word_hist)
    
    fname = DEFAULT_TEST_PATH + "test_const_word_vec.txt"
    test_f = open(fname, 'w')
    print_str = "{}, \nshape:{}, min:{}, max:{}, len:{}"
    print(print_str.format(word_vec,
                           word_vec.shape,
                           word_vec.min(),
                           word_vec.max(),
                           len(word_vec)),
          file=test_f)

def test_load_models():
    print("\tTesting load_models()...")
    c = NaiveBayesClassifier()
    #NaiveBayesClassifier.load_models is called in init.
    
    fname = DEFAULT_TEST_PATH + "test_load_models.txt"
    test_f = open(fname, 'w')
    print_str = "MLE:{}, \nMAP:{}"
    print(print_str.format(c.MLE_vec, c.MAP_matrix),
          file=test_f)
    
def test_classify():
    print("\tTesting classify()...")
    c = NaiveBayesClassifier()
    c.classify()
            
##==-- Main --==##
test_populate_word_histogram()
test_const_word_vector()
test_load_models()
test_classify()
