#!/usr/env python3
# filename: test_training.py
# authors: Jon David and Jarrett Decker
from training import Trainer

DEFAULT_TEST_PATH = "./test_output/"

def test_calc_vector_MLE():
    """Tests the Trainer.calc_vector_MLE function."""
    print("\tTesting Trainer.calc_vector_MLE()...")
    t = Trainer()
    MLE_vec = t.calc_vector_MLE()

    fname = DEFAULT_TEST_PATH + "test_calc_vector_MLE.txt"
    test_file = open(fname, "w")
    print(MLE_vec, file=test_file)
    test_file.close()

def test_calc_matrix_MAP():
    """Tests the Trainer.calc_matrix_MAP function."""
    print("\tTesting Trainer.calc_matrix_MAP()...")
    t = Trainer()
    MLE_vec = t.calc_vector_MLE() #required because it sets labeldict.
    MAP_matrix = t.calc_matrix_MAP()

    fname = DEFAULT_TEST_PATH + "test_calc_matrix_MAP.txt"
    test_f = open(fname, "w")
    print(MAP_matrix, file=test_f)
    print("shape:{}".format(MAP_matrix.shape), file=test_f)
    print("type:{}".format(MAP_matrix.dtype), file=test_f)
    print("min:{}".format(MAP_matrix.min()), file=test_f)
    print("max:{}".format(MAP_matrix.max()), file=test_f)
    test_f.close()

def test_train():
    print("\tTesting Trainer.train()...")
    t = Trainer()
    t.train()
    MLE_vec, MAP_matrix = t.generate_model()

    fname = DEFAULT_TEST_PATH + "test_generate_model.txt"
    test_f = open(fname, "w")
    mle_f = open(t.DEFAULT_MLE_FILENAME, 'r')
    map_f = open(t.DEFAULT_MAP_FILENAME, 'r')
    print("MLE:{} \n\nMAP:{}".format(mle_f, map_f),
          file=test_f)
    map_f.close()
    mle_f.close()
    test_f.close()

##==-- Main --==##
test_calc_vector_MLE()
test_calc_matrix_MAP()
test_train()
