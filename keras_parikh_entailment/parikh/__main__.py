from __future__ import division, unicode_literals, print_function
import spacy
import logging, sys
from matplotlib import pyplot as plt
import plac
from pathlib import Path, PosixPath
import ujson as json
import numpy
import keras
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json
from keras.optimizers import *
import random, string
import os
from os import listdir
from os.path import isfile, join
from tqdm import tqdm

from util import *
from spacy_hook import *
from keras_decomposable_attention import build_model, test_build_model, test_fit_model


def train(train_loc, dev_loc, shape, settings, test_train=False):
    preprocessed_data_path = Path(settings['preprocessed_data_dir'])

    [train_X1, train_X2,
     train_ents_labels1, train_ents_labels2,
     train_ents_ids1,    train_ents_ids2,
     train_labels]  = load_preprocessed_vectors('train', preprocessed_data_path)

    [dev_X1, dev_X2,
     dev_ents_labels1, dev_ents_labels2,
     dev_ents_ids1,    dev_ents_ids2,
     dev_labels]  = load_preprocessed_vectors('dev', preprocessed_data_path)

    trains = [train_X1, train_X2]
    devs = [dev_X1, dev_X2]
    if settings['use_ent']:
        trains = compute_inputs_with_NE(train_X1, train_X2,
                                        train_ents_ids1,    train_ents_ids2,
                                        train_ents_labels1, train_ents_labels2)
        devs = compute_inputs_with_NE(dev_X1, dev_X2,
                                      dev_ents_ids1,    dev_ents_ids2,
                                      dev_ents_labels1, dev_ents_labels2)

    if test_train:
        trains = [x[:10] for x in trains]
        devs = [x[:10] for x in devs]
        train_labels = train_labels[:10]
        dev_labels = dev_labels[:10]

    print("train\t", [x.shape for x in trains])
    print("dev\t", [x.shape for x in devs])

    print('Loading spaCy')
    nlp = spacy.load('en_core_web_lg')

    print('Compiling network')
    embed_vectors = get_embeddings(nlp.vocab)

    model_dir = Path(settings['model_dir'])
    model = build_model(embed_vectors, shape, settings)
    if model_dir.is_dir():
        print("Initializing model from", model_dir)
        weights = pickle.load(open(model_dir / 'model', "rb"))
        model.set_weights(weights)

        model.compile(
            optimizer=Adam(),
            loss='categorical_crossentropy',
            metrics=['accuracy'])
    else:
        model_dir.mkdir()
        model = build_model(embed_vectors, shape, settings)


    filepath= str(model_dir / "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5")
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    model.fit(
        trains,
        train_labels,
        validation_data=(devs, dev_labels),
        epochs=settings['nr_epoch'],
        callbacks=[checkpoint],
        batch_size=settings['batch_size'],
        shuffle=True
    )

    print("Saving to", model_dir)
    model.save_weights(model_dir / 'model.hdf5')
    with (model_dir / 'config.json').open('w') as file_:
        file_.write(model.to_json())


def evaluate_model(test_loc, shape, settings):
    test_data = load_preprocessed_vectors('test', Path(settings['preprocessed_data_dir']))
    print([x.shape for x in test_data])
    [test_X1, test_X2,
     ents_test_labels1, ents_test_labels2,
     ents_test_ids1, ents_test_ids2,
     test_claim_ids, test_ev_ids] = test_data

    model_dir = Path(settings['model_dir'])
    prediction_file = model_dir / 'prediction_scores.npy'
    print("Initializing model from", model_dir)


    print('Compiling network')
    try:
        model = model_from_json(open(model_dir / 'config.json').read())
    except:
        print('Loading spaCy')
        nlp = spacy.load('en_core_web_lg')
        embed_vectors = get_embeddings(nlp.vocab)

        model = build_model(embed_vectors, shape, settings)
    weights = pickle.load(open(model_dir / 'model', "rb"))
    model.set_weights(weights)

    model.compile(
        optimizer=Adam(lr=settings['lr']),
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    tests = [test_X1, test_X2]
    if settings['use_ent']:
        tests = compute_inputs_with_NE(test_X1, test_X2,
                                       ents_test_ids1, ents_test_ids2,
                                       ents_test_labels1, ents_test_labels2)

    predictions = model.predict(tests,
                                batch_size=settings['batch_size'],
                                verbose=1)
    print("Save prediction vectors to ", prediction_file)
    numpy.save(prediction_file, predictions)

    write_prediction_files(test_claim_ids, test_ev_ids, predictions, test_loc, model_dir)



def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


@plac.annotations(
    mode=("Mode to execute", "positional", None, str, ["preprocess", "train", "evaluate", "demo", "test"]),
    train_loc=("Path to training data", "option", None, Path),
    dev_loc=("Path to development data", "option", None, Path),
    test_loc=("Path to test data", "option", None, Path),
    preprocessed_data_dir=("Path to save preprocessed data", "option", None, str),
    text_max_length=("Length to truncate sentences", "option", "L", int),
    use_ent=("Use named entity as feature", "flag", "N", bool),
    ent_max_length=("Max number of entities", "option", 'm', int),
    nr_hidden=("Number of hidden units", "option", "H", int),
    dropout=("Dropout level", "option", "d", float),
    learn_rate=("Learning rate", "option", "e", float),
    batch_size=("Batch size for neural network training", "option", "b", int),
    nr_epoch=("Number of training epochs", "option", "i", int),
    tree_truncate=("Truncate sentences by tree distance", "flag", "T", bool),
    gru_encode=("Encode sentences with bidirectional GRU", "flag", "E", bool),
    embedding=("Path to embedding to use", "option", 'M', str),
    model_dir=("Name of the directory to save model to/load model from", "option", None, str)
)

def main(mode, train_loc, dev_loc, test_loc,
         preprocessed_data_dir=None,
        tree_truncate=False,
        gru_encode=False,
        text_max_length=250,
        use_ent=False,
        nr_hidden=200,
        dropout=0.25,
        learn_rate=0.002,
        batch_size=140,
        nr_epoch=5,
        embedding="default",
        ent_max_length=25,
        model_dir=None,
        ):
    if use_ent:
        shape = (text_max_length+(2*ent_max_length), nr_hidden, 3)
    else:
        shape = (text_max_length, nr_hidden, 3)

    if model_dir is None:
        model_dir = id_generator()

    settings = {
        'lr': learn_rate,
        'dropout': dropout,
        'batch_size': batch_size,
        'nr_epoch': nr_epoch,
        'tree_truncate': tree_truncate,
        'gru_encode': gru_encode,
        'model_dir': model_dir,
        'text_max_length': text_max_length,
        'ent_max_length': ent_max_length,
        'use_ent': use_ent,
        'preprocessed_data_dir':preprocessed_data_dir,
    }

    if mode == 'train':
        train(train_loc, dev_loc, shape, settings)
    elif mode == 'evaluate':
        assert test_loc is not None, 'evaluate model requires test data (use `-test-loc`)'
        evaluate_model(test_loc, shape, settings)
    elif mode == 'preprocess':
        assert preprocessed_data_dir
        preprocess(train_loc, dev_loc, test_loc, settings)
    elif mode =='test':
        test_build_model(settings,shape)
        train(train_loc, dev_loc, shape, settings, test_train=True)
    else:
        demo()

if __name__ == '__main__':
    plac.call(main)
