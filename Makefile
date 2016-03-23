CC=python3

all:

test:
	$(CC) test_trainer.py
	#$(CC) test_classifier.py

clean:
	*~
	rm ./test_output/*
