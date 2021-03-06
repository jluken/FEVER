import numpy
import numpy.random
import json
from io import StringIO
import random
from tqdm import tqdm
import datetime

from spacy.tokens.span import Span
import sys

try:
    import cPickle as pickle
except ImportError:
    import pickle


def get_embeddings(vocab, nr_unk=100):
    """
    returns a vector with indexed by lex.rank
    """
    nr_vector = max(lex.rank for lex in vocab) + 1
    vectors = numpy.zeros((nr_vector+nr_unk+2, vocab.vectors_length), dtype='float32')
    for lex in vocab:
        if lex.has_vector:
            vectors[lex.rank+1] = lex.vector / lex.vector_norm
    return vectors


def post_pad(arr, maxlen):
    if len(arr) >= maxlen:
        return numpy.array(arr)
    return numpy.pad(arr, (0, maxlen - len(arr)%maxlen), 'constant', constant_values=(18))


def get_ents_labels(docs, max_length=50):
    docs_ents_doc = []
    docs_ents_labels =[]

    LABEL_IDX = {'CARDINAL': 0,
                 'DATE': 1,
                 'EVENT': 2,
                 'FAC': 11,
                 'GPE': 5,
                 'LANGUAGE': 16,
                 'LAW': 13,
                 'LOC': 8,
                 'MONEY': 17,
                 'NORP': 15,
                 'ORDINAL': 14,
                 'ORG': 10,
                 'PERCENT': 12,
                 'PERSON': 4,
                 'PRODUCT': 7,
                 'QUANTITY': 3,
                 'TIME': 9,
                 'WORK_OF_ART': 6}

    for doc in tqdm(docs):
        i = 0
        ents_doc = []
        ents_labels = []
        els = [(ent, ent.label_) for ent in doc.ents]
        for (ent, label) in els:
            doc_len = len(ent.as_doc())
            i += doc_len
            ents_doc += ent.as_doc()
            ents_labels += [LABEL_IDX[label] for j in range(doc_len)]
            if i >= max_length:
                break

        docs_ents_doc.append(ents_doc[:max_length])

        label_arr = post_pad(ents_labels[:max_length], max_length)
        docs_ents_labels.append(label_arr)

    return docs_ents_doc, numpy.array(docs_ents_labels)


def get_word_ids(docs, rnn_encode=False, tree_truncate=False, max_length=250, nr_unk=100):
    Xs = numpy.zeros((len(docs), max_length), dtype='int32')
    for i, doc in tqdm(enumerate(docs)):
        if tree_truncate:
            if isinstance(doc, Span):
                queue = [doc.root]
            else:
                queue = [sent.root for sent in doc.sents]
        else:
            queue = list(doc)
        words = []
        while len(words) <= max_length and queue:
            word = queue.pop(0)
            if rnn_encode or (not word.is_punct and not word.is_space):
                words.append(word)
            if tree_truncate:
                queue.extend(list(word.lefts))
                queue.extend(list(word.rights))
        words.sort()
        for j, token in enumerate(words):
            if token.has_vector:
                Xs[i, j] = token.rank+1
            else:
                Xs[i, j] = (token.shape % (nr_unk-1))+2
            j += 1
            if j >= max_length:
                break
        else:
            Xs[i, len(words)] = 1
    return Xs



# def read_embedding_file(path):
#     """
#     read embedding files path_index_lookup and path_vectors_only
#     returns a index-vocab lookup and a list of vectors
#     Example:
#     j = index_lookup['cat']
#     vector[j] = numpy.array(...) -- the corresponding vector for the word 'cat'
#     """
#     vectors = pickle.load(open(path+'_vectors_only', 'rb'))
#     index_lookup: dict() = pickle.load(open(path+'_index_lookup', 'rb'))

#     """
#     with open(path, 'r') as vf:
#         for i, line in enumerate(vf):
#             linesplit = line.split(' ', 1)
#             k, v = linesplit[0], linesplit[1]
#             index_lookup[k] = i
#             vec = numpy.loadtxt(StringIO(v))
#             vectors.append(vec)
#     """

#     return index_lookup, vectors


# def get_word_ids_from_lookup(docs, index_lookup, max_length=100):
#     """
#     """
#     total_words_count = len(index_lookup.values())
#     Xs = numpy.zeros((len(docs), max_length), dtype='int32')
#     unk = index_lookup['<unk>']
#     for i, doc in enumerate(docs):
#         words = [token.lemma_.lower() for token in doc]
#         if (i % 1000 == 0):
#             print (i, "/", len(docs))
#         for j, token in enumerate(words):
#             if j >= max_length:
#                 break
#             try:
#                 Xs[i, j] = index_lookup[token]
#             except:
#                 print(token)
#                 Xs[i, j] = unk
#     return Xs



def strip_brackets(sent):
    index = sent.find('_-LRB')
    if index == -1:
        return sent
    return sent[:index]


def replace_word_with_pageid(word, pageid, sent):
    index = sent.find(word)
    assert (index != -1)
    prev = sent[:index]
    after = sent[index+len(word):]
    return prev+pageid+after


def expand_pageid(pageid, sent):
    pageid = strip_brackets(pageid.replace('_', ' '))
    if len(sent) == 0:
        return sent
    pronouns = ['he', 'she', 'it', 'they']
    words = list(sent.split(' '))
    # if the first half of the sentence contains a pronoun, replace it with page id
    for word in words[:int(len(words)/2)]:
        if word.lower() in pronouns:
            return replace_word_with_pageid(word, pageid, sent)

    # if the page id already appears in the sentence, return as it is
    if pageid in sent:
        return sent
    # if parts of the page id (e.g.: `Hill` is in the sentence where as `Dave_Hill` is the id),
    # replace `Hill` with `Dave Hill`
    for word in pageid.split(' '):
        index = sent.find(word)
        if index != -1:
            return replace_word_with_pageid(word, pageid, sent)

    # simply concatenate pageid in front of the sentence
    return pageid + ' ' + sent
