# coding=utf-8

from __future__ import print_function
import _pickle as pickle #cPickle
#import cPickle as pickle
import time
import re
import numpy as np
from gensim.models import word2vec, KeyedVectors

WORD_VECTOR_SIZE = 300

#raw_movie_conversations = open('data/movie_conversations.txt', 'r').read().split('\n')[:-1]
raw_ubuntu_conversations = open('data/ubuntu/s0-s1.txt', 'r').read().split('\n')#[:-1]

#utterance_dict = pickle.load(open('data/utterance_dict', 'rb'))

ts = time.time()
corpus = word2vec.Text8Corpus("data/tokenized_all_words.txt")
word_vector = word2vec.Word2Vec(corpus, size=WORD_VECTOR_SIZE)
word_vector.wv.save_word2vec_format(u"model/word_vector.bin", binary=True)
word_vector = KeyedVectors.load_word2vec_format('model/word_vector.bin', binary=True)
print("Time Elapsed: {} secs\n".format(time.time() - ts))

""" Extract only the vocabulary part of the data """
def refine(data):
    words = re.findall("[a-zA-Z'-]+", data)
    words = ["".join(word.split("'")) for word in words]
    # words = ["".join(word.split("-")) for word in words]
    data = ' '.join(words)
    return data

ts = time.time()
conversations = []
print('len conversation', len(raw_ubuntu_conversations))
con_count = 0
traindata_count = 0

MAX_SIZE = 20
con_a_1 = ''
for i in range(len(raw_ubuntu_conversations)-1):
    con_a_2 = raw_ubuntu_conversations[i]
    con_b = raw_ubuntu_conversations[i]
    if len(con_a_1.split()) <= MAX_SIZE and len(con_a_2.split()) <= MAX_SIZE and len(con_b.split()) <= MAX_SIZE:
        con_a = "{} {}".format(con_a_1, con_a_2)
        con_a = [refine(w) for w in con_a.lower().split()]
        # con_a = [word_vector[w] if w in word_vector else np.zeros(WORD_VECTOR_SIZE) for w in con_a]
        conversations.append((con_a, con_b, con_a_2))
        # former_sents.append(con_a_2)
        traindata_count += 1
    con_a_1 = con_a_2

pickle.dump(conversations, open('data/ubuntu/conversations_lenmax22_formersents2_with_former', 'wb'), True)
# pickle.dump(former_sents, open('data/conversations_lenmax22_former_sents', 'wb'), True)
print("Time Elapsed: {} secs\n".format(time.time() - ts))
