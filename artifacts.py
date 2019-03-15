import json
from pprint import pprint
import random

from os import listdir
from os.path import isfile, join
import io
import logging
import re
import string
import spacy
nlp = spacy.load('en')


# Reading and processing data

from collections import Counter
import sys

categories = Counter()
claims = {"SUPPORTS": [], "REFUTES": [], "NOT ENOUGH INFO": []}

with open('data/train.jsonl') as f:
    for i, line in enumerate(f):
        if i % 10000 == 0:
            sys.stdout.write('.')
        content = json.loads(line)
        claim = content['claim']
        label = content['label']
        categories[label] += 1 
        claims[label].append(claim)
		
claim_docs = dict()
for k in claims.keys():
    processed = list(nlp.pipe(claims[k], n_threads=40, batch_size=20000))
    claim_docs[k] = processed
	
claim_vocab = {"SUPPORTS": Counter(), "REFUTES": Counter(), "NOT ENOUGH INFO": Counter()}
claim_len = {"SUPPORTS": Counter(), "REFUTES": Counter(), "NOT ENOUGH INFO": Counter()}

# vocab count of all claims regardless of class
total_vocab = Counter()

total_word_count = 0
for cat in claim_docs.keys():
    for doc in claim_docs[cat]:
        claim_len[cat][len(doc)] += 1
        total_word_count += len(doc)
        for token in doc:
            claim_vocab[cat][token.lemma_] += 1
            total_vocab[token.lemma_] += 1

total_word_count == len(list(total_vocab.elements())) == 1377880


# Claim sentence length (by tokens)


# mean sent len
%matplotlib inline
import matplotlib.pyplot as plt
import numpy
plot_args = []

colors = ['r', 'b', 'g']

stats = []
for i, k in enumerate(claim_len.keys()):
    count = claim_len[k]
    elem = list(count.elements())
    stats.append([k, numpy.mean(elem), numpy.std(elem), min(elem), max(elem)])
    #print(" | ".join(list(map(str, [k, numpy.mean(elem), numpy.std(elem), min(elem), max(elem)]))))
    pmf, bins = numpy.histogram(elem, bins=range(0, 40), density=True)
    plt.plot(bins[:-1], pmf, colors[i], label=k)

plt.xlabel("sentence length (# tokens)")
plt.ylabel("probability of sentence length")
plt.legend(loc='upper right')
plt.show()

# PMI between each word and class in the training set


# calculate PMI 
def normalize_counter(oldcount, total):
    count = dict()
    for key in oldcount.keys():
        count[key] = oldcount[key]/total
    return count


# a dictionary mapping a class to probability of a word in that class
norm_vocab = dict()
# probability of a class

#for k in ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]:
#    vocab = claim_vocab[k]
#    claim_vocab[k] = {k:v for (k,v) in vocab.items()}
    
#total_word_count = sum([sum(vocab.values()) for vocab in claim_vocab.values()])
for k in ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]:
    norm_vocab[k] = normalize_counter(claim_vocab[k], total_word_count)

# probability of a word across all classes
norm_total_vocab = normalize_counter(total_vocab, total_word_count)

# should all sum to 1
sum(norm_total_vocab.values())
sum(norm_cat.values())
# sum([sum(v.values()) for v in norm_vocab.values()])

import math
pmi = dict()
for k in ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]:
    pmi[k] = dict()
    for word in claim_vocab[k].keys():
        pword_class = claim_vocab[k][word] / total_word_count
        pword = total_vocab[word]/total_word_count
        pmi[k][word] = pword_class/(pword * pclass[k])
		
print("Words with most PMI in each class")
from pprint import pprint
top_pmi = []
import operator
import csv
with open("pmi.tsv", "w") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerow(["Class", "Word", "P(class)", "P(word)", "P(word, class)", "pmi", "Count", "Class count"])
    for k in ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]:     
        sort = sorted(pmi[k].items(), key=operator.itemgetter(1), reverse=True)
        tops_class = list(map(list, sort[:10000]))
        top_pmi.append(tops_class)
        for word in tops_class:
            word = word[0]
            if (k == "NOT ENOUGH INFO"):
                key = "NEI"
            else:
                key = k
            writer.writerow([key, word, round(pclass[k],4), round(norm_total_vocab[word], 10), 
                             round(norm_vocab[k][word],10), 
                             round(pmi[k][word], 2),
                             round(total_vocab[word], 2), 
                             round(claim_vocab[k][word], 2)] )
							 
set([value for (word, value) in pmi["REFUTES"].items() if value > pmi["REFUTES"]["only"]])


# PMI of negation words


for k in ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]:
    for word in ["not", "never", "nor", "none", "null"]:
        try:
            print(k, word, pmi[k][word])
        except KeyError:
            print(k, word, "never occured")
    print("")

	
# PMI of temporal words


for k in ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]:
    for word in ["often", "always", "rarely", "frequently", "often", "regularly", "sometimes", "usually"]:
        try:
            print(k, word, pmi[k][word])
        except KeyError:
            print(k, word, "never occured")
    print("")







