# filename: Makefile
# authors: Jon David and Jarrett Decker


CC=python3


DF_INPUT_PREFIX="./data/"
INPUT_TRAIN_DATAFILE="./data/train.data"
INPUT_TRAIN_LABELFILE="./data/train.label"
INPUT_TEST_DATAFILE="./data/test.data"
INPUT_TEST_LABELFILE="./data/test.label"

DF_MODEL_PREFIX="./models/"
DF_MLE_MODELFILE="./models/mle.model"
DF_MAP_MODELFILE="./models/map.model"

DF_PRED_PREFIX="./model_predictions/"
DF_PREDICTIONFILE="./model_predictions/model.prediction"

DF_RESULTS_PREFIX="./model_results/"
DF_ACCURACYFILE="./model_results/model.accuracy"
DF_CONFUSIONMATRIXFILE="./model_results/model.confusion_matrix"


.PHONY: clean test test_all test_trainer test_classifier \
	test_confusionmatrix build_model build_beta_models \
	build_results build

all:
	#$(CC) test_trainer.py

####====---- build section ----================================================
build_model:
	time $(CC) build_models.py --data=$(INPUT_TRAIN_DATAFILE) \
	--label=$(INPUT_TRAIN_LABELFILE) --mle=$(DF_MLE_MODELFILE) \
	--map=$(DF_MAP_MODELFILE)

build_beta_models:
	make -f Makefile.betamodels

#build_predictions: 

build_results:
	time $(CC) build_results.py $(INPUT_TEST_LABELFILE) $(DF_PREDICTIONFILE) $(DF_ACCURACYFILE) $(DF_CONFUSIONMATRIXFILE)

####====---- test section ----=================================================
test: test_all

test_all: test_trainer test_classifier test_confusionmatrix

test_trainer: test_trainer.py
	time $(CC) test_trainer.py

test_classifier: test_classifier.py
	time $(CC) test_classifier.py

test_confusionmatrix: test_confusionmatrix.py
	time $(CC) test_confusionmatrix.py

clean:
	rm -f *.done
	rm -f *~
	rm -f $(DF_MODEL_PREFIX)/*.model
	rm -f $(DF_PRED_PREFIX)/*.prediction
	rm -f $(DF_RESULTS_PREFIX)/*.accuracy
	rm -f $(DF_RESULTS_PREFIX)/*.confusion_matrix
	rm -f ./test_output/*.txt
	rm -f ./__pycache__/*.pyc
	rm -f *.pyc
