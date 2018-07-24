from __future__ import division, unicode_literals, print_function
import spacy
import logging, sys

import plac
from pathlib import Path, PosixPath
import ujson as json
import numpy
import keras
from keras.callbacks import ModelCheckpoint
from keras.utils.np_utils import to_categorical
from keras.models import model_from_json
from keras.optimizers import Adam
import random, string
import os
from os import listdir
from os.path import isfile, join
from tqdm import tqdm


from spacy_hook import *

from keras_decomposable_attention import build_model, test_build_model, test_fit_model
from keras_lr_finder import LRFinder

try:
    import cPickle as pickle
except ImportError:
    import pickle


def preprocess(train_loc, dev_loc, test_loc, settings):
    train_texts1, train_texts2, train_labels, _, _ = read_fever(train_loc)
    dev_texts1, dev_texts2, dev_labels, _, _ = read_fever(dev_loc)
    test_texts1, test_texts2, _ , test_ev_ids, test_claim_ids = read_fever(test_loc, blind=True)

    print('Loading spaCy')
    nlp = spacy.load('en_core_web_lg')

    if (nlp.vocab.vectors_length == 0):
        logging.error("spaCy model has vocab vector length 0")
        exit(1)

    preprocessed_data_path = settings['preprocess_data_dir']
    if not os.path.isdir(preprocessed_data_path):
        os.mkdir(preprocessed_data_path)

    Xs = [] # each array is data_count * max_length
    all_entity_labels = [] # each array is data_count * entity_length
    all_entity_ids = []

    print('Processing text')
    all_texts = [train_texts1, train_texts2, dev_texts1, dev_texts2, test_texts1, test_texts2]
    for texts in all_texts:
        piped = list(nlp.pipe(texts, n_threads=20, batch_size=20000))
        text_ids = get_word_ids(piped,
                                max_length=settings['text_max_length'],
                                rnn_encode=settings['gru_encode'],
                                tree_truncate=settings['tree_truncate'])
        Xs.append(text_ids)

        print('Getting entity labels')
        ents_doc, ents_label = get_ents_labels(piped,
                                            max_length=settings['ent_max_length'])
        all_entity_labels.append(ents_label)

        ents_ids = get_word_ids(ents_doc,
                                max_length=settings['ent_max_length'],
                                rnn_encode=settings['gru_encode'],
                                tree_truncate=False)
        all_entity_ids.append(ents_ids)


    print("Xs\t", [x.shape for x in Xs])
    print("entity texts\t", [x.shape for x in all_entity_ids])
    print("entity labels\t", [x.shape for x in all_entity_labels])


    train_X1, train_X2,\
        dev_X1, dev_X2, \
        test_X1, test_X2 = Xs

    ents_train_labels, ents_train_labels,\
        ents_dev_labels, ents_dev_labels,\
        ents_test_labels, ents_test_labels = all_entity_labels
    numpy.save(preprocessed_data_path + 'ents_train_labels',[ents_train_labels, ents_train_labels] )
    numpy.save(preprocessed_data_path + 'ents_dev_labels',[ents_dev_labels, ents_dev_labels])
    numpy.save(preprocessed_data_path + 'ents_test_labels',[ents_test_labels, ents_test_labels])

    ents_train_ids, ents_train_ids,\
        ents_dev_ids, ents_dev_ids,\
        ents_test_ids, ents_test_ids = all_entity_labels

    numpy.save(preprocessed_data_path + 'ents_train_ids',[ents_train_ids, ents_train_ids] )
    numpy.save(preprocessed_data_path + 'ents_dev_ids',[ents_dev_ids, ents_dev_ids])
    numpy.save(preprocessed_data_path + 'ents_test_ids',[ents_test_ids, ents_test_ids])


    print('Save processed arrays')
    # save claim/evidence ids of only for the test set for evaluation
    numpy.save(preprocessed_data_path + 'train',[train_X1, train_X2] )
    numpy.save(preprocessed_data_path + 'train_label', train_labels)


    numpy.save(preprocessed_data_path + 'dev',[dev_X1, dev_X2])
    numpy.save(preprocessed_data_path + 'dev_label', dev_labels)

    numpy.save(preprocessed_data_path + 'test',[test_X1, test_X2])
    # numpy.save(preprocessed_data_path + 'test_label', test_labels)
    numpy.save(preprocessed_data_path + 'test_claim_ids', test_claim_ids)
    numpy.save(preprocessed_data_path + 'test_ev_ids', test_ev_ids)



def load_train_dev_data(path):
    train_X1, train_X2 = numpy.load(path + 'train.npy')
    dev_X1, dev_X2 = numpy.load(path + 'dev.npy')
    train_labels = numpy.load(path + 'train_label.npy')
    dev_labels = numpy.load(path + 'dev_label.npy')

    ents_train_labels1, ents_train_labels2 = numpy.load(path + 'ents_train_labels.npy')
    ents_train_ids1,    ents_train_ids2  = numpy.load(path + 'ents_train_ids.npy')
    ents_dev_labels1,   ents_dev_labels2 = numpy.load(path + 'ents_dev_labels.npy')
    ents_dev_ids1,      ents_dev_ids2    = numpy.load(path + 'ents_dev_ids.npy')

    return train_X1,           train_X2,           \
            ents_train_ids1,    ents_train_ids2,    \
            ents_train_labels1, ents_train_labels2, \
            train_labels, \
            dev_X1,             dev_X2,             \
            ents_dev_ids1,      ents_dev_ids2,      \
            ents_dev_labels1,   ents_dev_labels2,   \
            dev_labels



def load_test_data(path):
    test_X1, test_X2 = numpy.load(path + 'test.npy')
    ents_test_labels1, ents_test_labels2 = numpy.load(path + 'ents_test_labels.npy')
    ents_test_ids1, ents_test_ids2 = numpy.load(path + 'ents_test_ids.npy')
    # test_labels = numpy.load(path + 'test_label.npy')
    test_claim_ids = numpy.load(path + 'test_claim_ids.npy')
    test_ev_ids = numpy.load(path + 'test_ev_ids.npy')


    return test_X1, test_X2, \
            ents_test_labels1, ents_test_labels2, \
            ents_test_ids1, ents_test_ids2, \
            test_claim_ids, test_ev_ids

def compute_inputs_with_NE(input_X1, input_X2,
                            ents_input_ids1, ents_input_ids2,
                            ents_input_labels1, ents_input_labels2):

    input_X1 = numpy.column_stack((input_X1, ents_input_ids1))
    input_X2 = numpy.column_stack((input_X2, ents_input_ids2))

    inputs = [input_X1,           input_X2, \
              ents_input_labels1, ents_input_labels2]

    return inputs


def train(train_loc, dev_loc, shape, settings, test_train=False):
    preprocessed_data_path = settings['preprocess_data_dir']

    train_X1,           train_X2,           \
        ents_train_ids1,    ents_train_ids2,    \
        ents_train_labels1, ents_train_labels2, \
        train_labels, \
        dev_X1,             dev_X2,             \
        ents_dev_ids1,      ents_dev_ids2,      \
        ents_dev_labels1,   ents_dev_labels2,   \
        dev_labels = load_train_dev_data(preprocessed_data_path)



    trains = [train_X1, train_X2]
    devs = [dev_X1, dev_X2]
    if settings['use_ent']:
        train_X1 = numpy.column_stack((train_X1, ents_train_ids1))
        train_X2 = numpy.column_stack((train_X2, ents_train_ids2))
        dev_X1 = numpy.column_stack((dev_X1, ents_dev_ids1))
        dev_X2 = numpy.column_stack((dev_X2, ents_dev_ids2))
        trains = [train_X1,           train_X2, \
                  ents_train_labels1, ents_train_labels2]
        devs   = [dev_X1,           dev_X2, \
                  ents_dev_labels1, ents_dev_labels2]

    if test_train:
        trains = [x[:10] for x in trains]
        devs = [x[:10] for x in devs]
        train_labels = train_labels[:10]
        dev_labels = dev_labels[:10]

    print("train\t", [x.shape for x in trains])
    print("dev\t", [x.shape for x in devs])

    print('Loading spaCy')
    spacy_model = '/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/en_core_web_lg/en_core_web_lg-2.0.0'
    nlp = spacy.load(spacy_model)

    if (nlp.vocab.vectors_length == 0):
        logging.error("spaCy model has vocab vector length 0")
        exit(1)


    print('Compiling network')
    embed_vectors = get_embeddings(nlp.vocab)

    model_dir = Path(settings['model_dir'])
    if model_dir.is_dir():
        print("Initializing model from", model_dir)
        model = model_from_json(open(model_dir / 'config.json').read())
        weights = pickle.load(open(model_dir / 'model', "rb"))
        model.set_weights(weights)

        model.compile(
            optimizer=Adam(lr=settings['lr']),
            loss='categorical_crossentropy',
            metrics=['accuracy'])
    else:
        model = build_model(embed_vectors, shape, settings)
        model_dir.mkdir()

    filepath= str(model_dir / "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5")
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    model.fit(
        trains,
        train_labels,
        validation_data=(devs, dev_labels),
        epochs=settings['nr_epoch'],
        callbacks=[checkpoint],
        batch_size=settings['batch_size'])



    model.fit(
        trains,
        train_labels,
        validation_data=(devs, dev_labels),
        epochs=settings['nr_epoch'],
        batch_size=settings['batch_size'])

    model_dir = Path(settings['model_dir'])

    if not model_dir.exists():
        model_dir.mkdir()

    print("Saving to", model_dir)
    weights = model.get_weights()
    with (model_dir / 'model').open('wb') as file_:
        pickle.dump(weights, file_)
    with (model_dir / 'config.json').open('w') as file_:
        file_.write(model.to_json())

def evaluate_model(test_loc, shape, settings):
    test_X1, test_X2, \
        ents_test_labels1, ents_test_labels2, \
        ents_test_ids1, ents_test_ids2, \
        test_claim_ids, test_ev_ids = load_test_data(settings['preprocess_data_dir'])

    model_dir = Path(settings['model_dir'])

    prediction_file = model_dir / 'prediction_scores.npy'

    if prediction_file.is_file():
        predictions = numpy.load(prediction_file)
    else:
        # apply this PR if receiving errors
        # https://github.com/keras-team/keras/pull/8031
        print("Initializing model from", model_dir)

        model = model_from_json(open(model_dir / 'config.json').read())
        weights = pickle.load(open(model_dir / 'model', "rb"))
        model.set_weights(weights)

        model.compile(
            optimizer=Adam(lr=settings['lr']),
            loss='categorical_crossentropy',
            metrics=['accuracy'])

        tests = [test_X1, test_X2]
        if settings['use_ent']:
            tests = compute_inputs_with_NE(test_X1, test_X2,
                                           ents_test_labels1, ents_test_labels2,
                                           ents_test_ids1, ents_test_ids2)

        predictions = model.predict(tests, verbose=1)
        print("Save prediction vectors to ", prediction_file)
        numpy.save(prediction_file, predictions)

    write_prediction_files(test_claim_ids, test_ev_ids, predictions, test_loc, model_dir)


def sample_random_sentences(total_sent_count, sentences, ids):
    rand_sents = []
    rand_ids = []
    for i in random.sample(range(0,total_sent_count), 5):
        rand_sents.append(sentences[i])
        rand_ids.append(ids[i])
    return rand_sents, rand_ids


def read_fever(path, blind=False):
    LABELS = {'SUPPORTS': 0, 'REFUTES': 1, 'NOT ENOUGH INFO': 2}
    texts1 = []
    texts2 = []
    labels = []
    evidence_ids = []
    claim_ids = []
    start_idx = 2
    if blind:
        start_idx = 0

    evidence_count = 0

    # list of ids with no evidence found
    no_evidence_found = []

    with path.open() as file_:
        for i, line in tqdm(enumerate(file_)):
            try:
                content = json.loads(line)
            except:
                print(line, type(line))
            claim = content["claim"]
            if len(content["evidence"]) == 0: # sentencefinder output no evidence
                if blind:
                    no_evidence_found.append(content["id"])
                    continue
                else:
                    rand_sents, rand_ids = sample_random_sentences(evidence_count, texts1, evidence_ids)
                    texts1 += rand_sents
                    texts2 += [claim for x in range(5)]
                    claim_ids += [content["id"] for x in range(5)]
                    evidence_ids += rand_ids
                    evidence_count += 5
                continue
            for evidence_set in content["evidence"]:
                coref_sents = []
                for ev in evidence_set:
                    coref_sent = expand_pageid(ev[start_idx], ev[start_idx+2])
                    coref_sents.append(coref_sent)

                ev_id = ".".join(["~".join([ev[start_idx], str(ev[start_idx+1])]) for ev in evidence_set])
                ev_sent = " ".join(coref_sents)
                texts1.append(ev_sent)
                texts2.append(claim)
                evidence_ids.append(ev_id)
                claim_ids.append(content["id"])
                evidence_count += 1
                if not blind:
                    label = content["label"]
                    labels.append(LABELS[label])

    if not blind:
        labels = to_categorical(numpy.asarray(labels, dtype='int32'))
    return texts1, texts2, labels, evidence_ids, claim_ids


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


@plac.annotations(
    mode=("Mode to execute", "positional", None, str, ["preprocess", "train", "evaluate", "demo", "test"]),
    train_loc=("Path to training data", "option", None, Path),
    dev_loc=("Path to development data", "option", None, Path),
    test_loc=("Path to test data", "option", None, Path),
    preprocess_data_dir=("Path to save preprocessed data", "option", None, str),
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
         preprocess_data_dir='preprocessed_data_coref/',
        tree_truncate=False,
        gru_encode=False,
        text_max_length=250,
        use_ent=False,
        nr_hidden=200,
        dropout=0.2,
        learn_rate=0.001,
        batch_size=100,
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
        'preprocess_data_dir':preprocess_data_dir
    }

    if mode == 'train':
        train(train_loc, dev_loc, shape, settings)
    elif mode == 'evaluate':
        assert test_loc is not None, 'evaluate model requires test data (use `-test-loc`)'
        evaluate_model(test_loc, shape, settings)
    elif mode == 'preprocess':
        preprocess(train_loc, dev_loc, test_loc, settings)
    elif mode =='test':
        test_build_model(settings,shape)
        train(train_loc, dev_loc, shape, settings, test_train=True)
    else:
        demo()

if __name__ == '__main__':
    plac.call(main)
