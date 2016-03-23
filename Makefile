# filename: Makefile
# authors: Jon David and Jarrett Decker


CC=python3

all:
	#$(CC) test_trainer.py

test:
	#time $(CC) test_trainer.py
	time $(CC) test_classifier.py

clean:
	rm -f *~
	rm -f ./models/*.model
	rm -f ./test_output/*.txt
	rm -f ./__pycache__/*.pyc
	rm -f *.pyc
