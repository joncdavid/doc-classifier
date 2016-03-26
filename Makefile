# filename: Makefile
# authors: Jon David and Jarrett Decker


CC=python3


DF_INPUT_PREFIX=./data/
DF_TRAIN_DATAFILE=./data/train.data
DF_TRAIN_LABELFILE=./data/train.label
DF_TEST_DATAFILE=./data/test.data
DF_TEST_LABELFILE=./data/test.label

DF_MODEL_PREFIX=./models/
DF_MLE_MODELFILE=./models/mle.model
DF_MAP_MODELFILE=./models/map.model
DF_EVIDENCE_MODELFILE=./models/evidence.model

DF_PRED_PREFIX=./model_predictions/
DF_PREDICTIONFILE=./model_predictions/model.prediction

DF_RESULTS_PREFIX=./model_results/
DF_WORD_RANKFILE=./model_results/word.rankings
DF_ACCURACYFILE=./model_results/model.accuracy
DF_CONFUSIONMATRIXFILE=./model_results/model.confusion_matrix


.PHONY: clean test test_all test_trainer test_classifier \
	test_confusionmatrix build_model build_beta_models \
	build_prediction build_beta_predictions build_result \
	build_beta_results

all:
	#$(CC) test_trainer.py

####====---- questions section ----============================================
#build_q1:
#build_q2:
#build_q3:
#build_q4:
#build_q5:
#build_q6:
#build_q7:


####====---- build section ----================================================
build: build_model build_prediction build_result

build_betas: build_beta_models build_beta_ranks build_beta_predictions \
	build_beta_results

build_model:
	time $(CC) build_models.py --data=$(DF_TRAIN_DATAFILE) \
	--label=$(DF_TRAIN_LABELFILE) --mle=$(DF_MLE_MODELFILE) \
	--map=$(DF_MAP_MODELFILE) --evidence=$(DF_EVIDENCE_MODELFILE)

build_beta_models:
	time make -f Makefile.betas build_beta_models

build_rank:
	time $(CC) build_ranks.py $(DF_MLE_MODELFILE) $(DF_MAP_MODELFILE) \
	$(DF_EVIDENCE_MODELFILE) $(DF_WORD_RANKFILE)

build_beta_ranks:
	time make -f Makefile.betas build_beta_ranks

build_prediction:
	time $(CC) build_predictions.py $(DF_MLE_MODELFILE) \
	$(DF_MAP_MODELFILE) $(DF_PREDICTIONFILE)

build_beta_predictions:
	time make -f Makefile.betas build_beta_predictions

build_result:
	time $(CC) build_results.py $(DF_TEST_LABELFILE) $(DF_PREDICTIONFILE) \
	$(DF_ACCURACYFILE) $(DF_CONFUSIONMATRIXFILE)

build_beta_results:
	time make -f Makefile.betas build_beta_results


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
	rm -f $(DF_MODEL_PREFIX)/*.model*
	rm -f $(DF_PRED_PREFIX)/*.prediction*
	rm -f $(DF_RESULTS_PREFIX)/*.accuracy*
	rm -f $(DF_RESULTS_PREFIX)/*.confusion_matrix*
	rm -f $(DF_RESULTS_PREFIX)/*.data*
	rm -f $(DF_RESULTS_PREFIX)/*.rankings*
	rm -f $(DF_RESULTS_PREFIX)/*top_words*.txt
	rm -f ./test_output/*.txt
	rm -f ./__pycache__/*.pyc
	rm -f *.pyc
