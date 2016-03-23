#!/usr/env python3
# filename: test_training.py
# authors: Jon David and Jarrett Decker
from training import Trainer

DEFAULT_TEST_PATH = "./test_output/test_trainer/"

def test_getdata():
    """Tests the Trainer.getdata function."""
    filename = DEFAULT_TEST_PATH + "test_getdata.txt"
    test_file = open(filename, "w")
    test_trainer = Trainer()
    test_trainer.getlabels()

    for x in range (1, 100, 10):
        print(test_trainer.labeldict[x],
              file=test_file)
    
    test_trainer.getdata()

    for x in range (0, 100, 20):
        for y in range(0, 9):
            print(test_trainer.dataarray[y, x],
                  file=test_file)
    test_file.close()

def test_calc_vector_MLE():
    """Tests the Trainer.calc_vector_MLE function."""
    t = Trainer()
    t.getdata()
    MLE_vec = t.calc_vector_MLE()

    fname = DEFAULT_TEST_PATH + "test_calc_vector_MLE.txt"
    test_file = open(fname, "w")
    print(MLE_vec, file=test_file)
    test_file.close()

def test_calc_matrix_MAP():
    """Tests the Trainer.calc_matrix_MAP function."""
    t = Trainer()
    t.getdata()
    MAP_matrix = t.calc_matrix_MAP()

    fname = DEFAULT_TEST_PATH + "test_calc_matrix_MAP.txt"
    test_f = open(fname, "w")
    print(MAP_matrix, file=test_f)
    print("shape:{}".format(MAP_matrix.shape), file=test_f)
    print("type:{}".format(MAP_matrix.dtype), file=test_f)
    print("min:{}".format(MAP_matrix.min()), file=test_f)
    print("max:{}".format(MAP_matrix.max()), file=test_f)

    test_f.close()

##==-- Main --==##
#test_getdata()
#test_calc_vector_MLE()
test_calc_matrix_MAP()
