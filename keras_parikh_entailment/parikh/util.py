import json
import numpy
import os
from pathlib import Path, PosixPath
import spacy
from tqdm import tqdm
from spacy_hook import *

def preprocess_(loc, name, nlp, settings):
    def save_numpy(dir, name, subname, vector):
        newfile = dir / (name + subname)
        print('Save to', newfile)
        numpy.save(newfile, vector)
        return

    if name not in {'train', 'dev', 'test'}:
        exit(2)

    blind = name == 'test'
    texts1, texts2, labels , ev_ids, claim_ids = read_fever(loc, blind=blind)

    Xs = [] # each array is data_count * max_length
    all_entity_labels = [] # each array is data_count * entity_length
    all_entity_ids = []

    print('Processing text')
    all_texts = [texts1, texts2]
    for texts in all_texts:
        piped = list(nlp.pipe(texts, n_threads=20, batch_size=50000))
        text_ids = get_word_ids(piped,
                                max_length=settings['text_max_length'],
                                rnn_encode=settings['gru_encode'],
                                tree_truncate=settings['tree_truncate'])
        Xs.append(text_ids)

        print('Getting entity labels', name)
        ents_doc, ents_label = get_ents_labels(piped,
                                            max_length=settings['ent_max_length'])
        all_entity_labels.append(ents_label)

        ents_ids = get_word_ids(ents_doc,
                                max_length=settings['ent_max_length'],
                                rnn_encode=settings['gru_encode'],
                                tree_truncate=False)
        all_entity_ids.append(ents_ids)


    preprocessed_data_dir = Path(settings['preprocess_data_dir'])

    # for all data, save texts1/2, ents_labels1/2, ents_ids1/2
    save_numpy(preprocessed_data_dir, name, '_texts', Xs)
    save_numpy(preprocessed_data_dir, name, '_ents_labels', all_entity_labels)
    save_numpy(preprocessed_data_dir, name, '_ents_ids', all_entity_ids)

    if name == 'test':
        # save evidence_ids and claim_ids only for test data
        save_numpy(preprocessed_data_dir, name, '_claim_ids', claim_ids)
        save_numpy(preprocessed_data_dir, name, '_ev_ids', ev_ids)
        # numpy.save(preprocessed_data_dir / (name + '_ev_ids'), ev_ids)
    else:
        # save labels only for train dev data
        save_numpy(preprocessed_data_dir, name, '_labels', labels)


def load_preprocessed_vectors(name, path):
    X1, X2 = numpy.load(path / (name + '_texts.npy'))
    ents_labels1, ents_labels2 = numpy.load(path / (name + '_ents_labels.npy'))
    ents_ids1,    ents_ids2  = numpy.load(path / (name + '_ents_ids.npy'))

    if name == 'test':
        claim_ids = numpy.load(path / 'test_claim_ids.npy')
        ev_ids = numpy.load(path / 'test_ev_ids.npy')
        return [X1, X2,
                ents_labels1, ents_labels2,
                ents_ids1, ents_ids2,
                claim_ids, ev_ids]
    else:
        labels = numpy.load(path / (name + '_labels.npy'))
        return [X1, X2,
                ents_labels1, ents_labels2,
                ents_ids1, ents_ids2,
                labels]


def preprocess(train_loc, dev_loc, test_loc, settings):
    print('Loading spaCy')
    nlp = spacy.load('en_core_web_lg')

    if (nlp.vocab.vectors_length == 0):
        logging.error("spaCy model has vocab vector length 0")
        exit(1)

    preprocessed_data_dir = Path(settings['preprocessed_data_dir'])
    if not preprocessed_data_dir.is_dir():
        os.mkdir(preprocessed_data_dir)

    if train_loc is not None:
        preprocess_(train_loc, 'train', nlp, settings)
    if dev_loc is not None:
        preprocess_(dev_loc, 'dev', nlp, settings)
    if test_loc is not None:
        preprocess_(test_loc, 'test', nlp, settings)


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
            content = json.loads(line)
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
    print ([len(x) for x in [texts1, texts2, labels, evidence_ids, claim_ids]])
    return texts1, texts2, labels, evidence_ids, claim_ids


def write_prediction_files(ids, evids, predictions, infile, model_dir):
    # result_by_claim = write_classifier_prediction(sents1, sents2, ids, predictions, infile, model_dir)
    result_by_claim = aggregate_classifier_result(ids, evids, predictions)
    write_fever_prediction(result_by_claim, infile, model_dir)


def aggregate_classifier_result(ids, evids, predictions):
    LABELS = ['SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO']
    result_by_claim = dict()
    for i, (num, evid, pred) in enumerate(zip(ids, evids, predictions)):
        try:
            data = result_by_claim[num]
            assert data['id'] == num
        except KeyError:
            result_by_claim[num] = dict()
            data = result_by_claim[num]
            data['id'] = num
            data['predicted_evidence'] = []
        data['predicted_evidence'].append({'prediction': LABELS[int(pred.argmax())],
                                           'score': list(map(float, list(pred))),
                                           'evidence_id' : evid
                                           })
    return result_by_claim

def write_fever_prediction(result_by_claim, infile, model_dir, max_evidence=5):
    def format_evidence(dicts, debug=False):
        predicted_evidence = []
        for x in dicts:
            evid = x['evidence_id']
            page, idx = evid.split("~")
            output = [page, int(idx)]
            if debug:
                output += [x['prediction'], x['score']]
            predicted_evidence.append(output)
        return predicted_evidence

    LABELS = ['SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO']
    stamp = str(datetime.datetime.now())
    predict_file = model_dir / ('test_predictions.jsonl-' + stamp)
    debug_file = model_dir / ("test_log.jsonl-" + stamp)
    with open(predict_file, 'w') as f, \
         open(debug_file,  'w') as f_debug, \
         open(infile, 'r') as infile_:
        for i, line in enumerate(infile_):
            content = json.loads(line)
            num = content['id']
            pred_evidence = content['evidence']

            debug_content = {'id': content['id'],
                             'claim': content['claim'],
                             'retrieved_evidence': content['evidence']}

            discarded_evidences = []
            top_scored_evidences = []
            if len(pred_evidence) > 0:
                result = result_by_claim[num]
                non_nei_evidences = []
                # Remove the evidences with label NEI
                # non_nei_evidences = [x for x in result['predicted_evidence'] if x['prediction'] != 'NOT ENOUGH INFO']
                for ev in result['predicted_evidence']:
                    if ev['prediction'] != 'NOT ENOUGH INFO':
                        non_nei_evidences.append(ev)
                    else:
                        discarded_evidences.append(ev)

                # Calculate the end label for the claim
                # by summing all evidences scores
                if len(non_nei_evidences) > 0:
                    all_scores = [ev['score'] for ev in non_nei_evidences]
                    score_sum = numpy.array([sum(x) for x in zip(*all_scores)])
                    end_label = LABELS[int(score_sum.argmax())]
                    debug_content['predicted_label'] = end_label

                    # Remove evidences with label different from end_label
                    # consistent_evidences = [x for x in non_nei_evidences if x['prediction'] == end_label]
                    # inconsistent_evidences = [x for x in non_nei_evidences if x['prediction'] != end_label]
                    # discarded_evidences += inconsistent_evidences

                    # Choose top 5 evidences with highest score
                    top_scored_evidences = sorted(consistent_evidences, key=lambda k: max(k['score']))[:max_evidence]
                    debug_content["predicted_evidence"] = format_evidence(top_scored_evidences, debug=True)
                    debug_content['discarded_evidence'] = format_evidence(discarded_evidences, debug=True)

                else:
                    debug_content['predicted_evidence'] = []
                    debug_content['predicted_label'] = 'NOT ENOUGH INFO'
            else:
                debug_content['predicted_evidence'] = []
                debug_content['predicted_label'] = 'NOT ENOUGH INFO'


            f_debug.write(json.dumps(debug_content) + "\n")
            output_content = {'id': num,
                              'predicted_label': debug_content['predicted_label'],
                              'predicted_evidence': format_evidence(top_scored_evidences)
                              }
            f.write(json.dumps(output_content) + "\n")

    f.close()
    f_debug.close()
    print("Write files:", predict_file, debug_file)
