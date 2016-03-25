#!/usr/env python3
# filename: test_classifier.py
# authors: Jon David and Jarrett Decker
from classifier import *

DEFAULT_TEST_PATH = "./test_output/"

def test_const_word_histograms():
    print("\tTesting const_word_histograms()...")
    c = NaiveBayesClassifier()
    doc_word_dict = c.const_word_histograms()

    fname = DEFAULT_TEST_PATH + "test_const_word_hist.txt"
    test_f = open(fname, 'w')
    pstr = "{}, \nlen:{}"
    print(pstr.format(doc_word_dict,
                      map(len,doc_word_dict.values())),
          file=test_f)
    
def test_const_word_vectors():
    print("\tTesting const_word_vector()...")
    c = NaiveBayesClassifier()
    doc_word_dict = c.const_word_histograms()
    doc_word_vec_dict = c.const_word_vectors(doc_word_dict)
    
    fname = DEFAULT_TEST_PATH + "test_const_word_vec.txt"
    test_f = open(fname, 'w')
    print_str = "{}, \nlen:{}"
    print(print_str.format(doc_word_vec_dict,
                           len(doc_word_vec_dict)),
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
    doc_word_dict = c.const_word_histograms()
    doc_word_vec_dict = c.const_word_vectors(doc_word_dict)
    predictions = c.classify(doc_word_vec_dict, c.MLE_vec, c.MAP_matrix)

    fname = DEFAULT_TEST_PATH + "test_classify.txt"
    test_f = open(fname, 'w')
    for prediction in predictions.tolist():
        print(int(prediction), file=test_f)

def test_confusion():
    print("\tTesting confusionmatrix()...")
    c = NaiveBayesClassifier()
    fname = DEFAULT_TEST_PATH + "test_classify.txt"
    print(c.accuracy(c.confusionmatrix()))


##==-- Main --==##
test_const_word_histograms()
test_const_word_vectors()
test_load_models()
test_classify()
test_confusion()
