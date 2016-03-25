# filename: Makefile
# authors: Jon David and Jarrett Decker


CC=python3

all:
	#$(CC) test_trainer.py

test: test_trainer test_classifier test_confusionmatrix

test_trainer:
	time $(CC) test_trainer.py

test_classifier: test_trainer
	time $(CC) test_classifier.py

test_confusionmatrix: test_classifier test_trainer
	time $(CC) test_confusionmatrix.py

clean:
	rm -f *~
	rm -f ./models/*.model
	rm -f ./test_output/*.txt
	rm -f ./__pycache__/*.pyc
	rm -f *.pyc
