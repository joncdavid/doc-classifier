#!/usr/env python3
# filename: test_training.py
# authors: Jon David and Jarrett Decker
from training import Trainer

test_trainer = Trainer()
test_trainer.getlabels()

for x in range (1, 100, 10):
    print(test_trainer.labeldict[x])

test_trainer.train()

for x in range (0, 100, 20):
    for y in range(0, 9):
        print(test_trainer.dataarray[y, x])

