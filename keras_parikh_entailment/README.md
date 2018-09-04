# Entailment 

This directory contains a modified implementation of the entailment prediction model described by 
[Parikh et al. (2016)](https://arxiv.org/pdf/1606.01933.pdf). Most of the code were written by [Matthew Honnibal](https://github.com/explosion/spaCy/tree/master/examples/keras_parikh_entailment). 
It is used the entailment module in the FEVER system pipeline.

To preprocess the data and save the preprocessed vector to `DATA_DIR`:

```
$ python3 parikh preprocess -train-loc TRAIN_LOC -dev-loc DEV_LOC -test-loc TEST_LOC -preprocessed-data-dir DATA_DIR
```

To train the model:
```
python3 parikh train -model-dir MODEL_DIR -preprocess-data-dir  DATA_DIR
```

To use the trained model to make predictions:
```
python3 parikh evaluate -model-dir MODEL_DIR -preprocess-data-dir  DATA_DIR
```
