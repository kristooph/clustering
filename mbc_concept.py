# -*- coding: utf-8 -*-
# source:
# https://nlp.stanford.edu/IR-book/html/htmledition/model-based-clustering-1.html

import os
from fnmatch import fnmatch
import re
import copy

debug = False
# debug = True

root = 'data'
pattern = "*.txt"

# add every word from every document to bag-of-words
bag_of_words = {}
# all files from all possible subdirectories
# https://stackoverflow.com/a/13214966/9153983
for path, subdirs, files in os.walk(root):
    for name in files:
        if fnmatch(name, pattern):
            file = os.path.join(path, name)

            with open(file, 'r', encoding='utf-8') as input:
                text = input.read()

            words = re.sub('[^a-zA-ZÀ-ž \']+', ' ', text)   # leave chars, spaces and '
            words = re.sub('\s+', ' ', words)   # remove multiple whitespaces

            for word in words.split():
                bag_of_words[word] = '0'
if debug:
    print('bag_of_words', bag_of_words)
# >>> bag_of_words
# {'hot': '0', 'chocolate': '0', ...}

class Document():
    def __init__(self, file):
        self.vector = copy.deepcopy(bag_of_words)
        with open(file, 'r', encoding='utf-8') as input:
            text = input.read()

        words = re.sub('[^a-zA-ZÀ-ž \']+', ' ', text)   # save chars, spaces and ' only
        words = re.sub('\s+', ' ', words)   # remove multiple whitespaces

        for word in words.split():
            if word in self.vector.keys():
                self.vector[word] = 1


documents = {}
for path, subdirs, files in os.walk(root):
    for name in files:
        if fnmatch(name, pattern):
            file = os.path.join(path, name)
            documents[name] = Document(file)
if debug:
    print('documents', documents)
# >>> documents[list(documents.keys())[0]].vector
# {'hot': 1, 'chocolate': 1, ..., 'sweet': '0', 'cake': '0', ...}

K = 2                   # clusters
eps = 0.0001
N = len(documents)      # documents (total)
M = len(bag_of_words)   # terms (total uniques)

clusters = list(range(1, K+1))
if debug:
    print('clusters', clusters)
# >>> clusters
# [1, 2]

p_new_doc_in_cluster = {}
# The probability $\alpha_k$ is the prior of cluster $\omega_k$:
# the probability that a document $d$ is in $\omega_k$ if we have no information about $d$
for cluster in clusters:
    p_new_doc_in_cluster[cluster] = 0.0
if debug:
    print('p_new_doc_in_cluster', p_new_doc_in_cluster)
# >>> p_new_doc_in_cluster
# {1: 0.0, 2: 0.0}

p_term_in_cluster = copy.deepcopy(bag_of_words)
# p_term_in_cluster - probability that a document from
# cluster $\omega_k$ contains term $\tcword_m$
for term in p_term_in_cluster:
    p_term_in_cluster[term] = {cluster:0.0 for cluster in clusters}
if debug:
    print('p_term_in_cluster', p_term_in_cluster)
# >>> p_term_in_cluster
# {'hot': {1: 0.0, 2: 0.0}, ...}

soft_assignment = {}
# soft_assignment - $r_{nk}$ is the soft assignment of
# document $d_n$ to cluster $k$
# as computed in the preceding iteration
for doc in documents:
    soft_assignment[doc] = {cluster:0.0 for cluster in clusters}
# if debug:
#     print('soft_assignment', soft_assignment)
# >>> soft_assignment
# {'1.txt': {1: 0.0, 2: 0.0}, ...}



# --- expectation-maximization init ---
# todo: initialization for given set of documents
# 0-th iteration
soft_assignment['6.txt'][1] = 1.0
soft_assignment['6.txt'][2] = 0.0
soft_assignment['7.txt'][1] = 0.0
soft_assignment['7.txt'][2] = 1.0
if debug:
    print('soft_assignment', soft_assignment)



# --- expectation-maximization algorithm ---
# todo: change from fixed #iters to convergence of objective function
iterations = 1
for iteration in range(iterations):
    # --- maximization step ---

    for term in p_term_in_cluster:
        for cluster in p_term_in_cluster[term]:
            # this has to be done other way than in the paper
            qmk_num = eps
            qmk_denom = 0.0
            for doc in documents:
                # description under table lies, nicely done cambridge
                qmk_num += ((soft_assignment[doc][cluster]) if documents[doc].vector[term] == 1 else 0.0)
                qmk_denom += (soft_assignment[doc][cluster])
            p_term_in_cluster[term][cluster] = qmk_num / qmk_denom

    for cluster in p_new_doc_in_cluster:
        alpha_num = eps
        for doc in documents:
            alpha_num += (soft_assignment[doc][cluster])
        # this has to be done other way than in the paper
        p_new_doc_in_cluster[cluster] = alpha_num
    # normalize
    norm = 0.0
    for cluster in p_new_doc_in_cluster:
        norm += p_new_doc_in_cluster[cluster]
    for cluster in p_new_doc_in_cluster:
        p_new_doc_in_cluster[cluster] /= norm

    # --- expectation step ---

    sa_denoms = {}
    for doc in soft_assignment:
        sa_denoms[doc] = 0.0
        for cluster in soft_assignment[doc]:
            rnk_num = p_new_doc_in_cluster[cluster]
            for term in bag_of_words:
                rnk_num *= (p_term_in_cluster[term][cluster] if documents[doc].vector[term] == 1 else (1.0 - p_term_in_cluster[term][cluster]))
            soft_assignment[doc][cluster] = rnk_num
            sa_denoms[doc] += rnk_num
    for doc in soft_assignment:
        for cluster in soft_assignment[doc]:
            soft_assignment[doc][cluster] /= sa_denoms[doc]

print('a1', p_new_doc_in_cluster[1])
print('r1', soft_assignment['1.txt'][1])
print('r2', soft_assignment['2.txt'][1])
print('r3', soft_assignment['3.txt'][1])
print('r4', soft_assignment['4.txt'][1])
print('r5', soft_assignment['5.txt'][1])
print('r6', soft_assignment['6.txt'][1])
print('r7', soft_assignment['7.txt'][1])
print('r8', soft_assignment['8.txt'][1])
print('r9', soft_assignment['9.txt'][1])
print('r10', soft_assignment['10.txt'][1])
print('r11', soft_assignment['11.txt'][1])
print('qafrica1', p_term_in_cluster['africa'][1])
print('qafrica2', p_term_in_cluster['africa'][2])
print('qbrazil1', p_term_in_cluster['brazil'][1])
print('qbrazil2', p_term_in_cluster['brazil'][2])
print('qcocoa1', p_term_in_cluster['cocoa'][1])
print('qcocoa2', p_term_in_cluster['cocoa'][2])
print('qsugar1', p_term_in_cluster['sugar'][1])
print('qsugar2', p_term_in_cluster['sugar'][2])
print('qsweet1', p_term_in_cluster['sweet'][1])
print('qsweet2', p_term_in_cluster['sweet'][2])