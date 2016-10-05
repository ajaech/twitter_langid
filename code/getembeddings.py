"""
Utility for printing nearest neighbors to a given word in embedding space.

Uses the char2vec portion of a trained model and the tweetlid vocabulary.
Nearest neighbors are given based on cosine similarity.
"""
import sys
reload(sys)
sys.setdefaultencoding('utf8')

import argparse
import json
import numpy as np
import os
import tensorflow as tf

from batcher import Dataset
from char2vec import CharCNN as Char2Vec
from vocab import Vocab

parser = argparse.ArgumentParser()
parser.add_argument('expdir')
args = parser.parse_args()

config = tf.ConfigProto(inter_op_parallelism_threads=10,
                intra_op_parallelism_threads=10)

dataset = Dataset(10, preshuffle=False)
dataset.ReadData('../data/tweetlid/training.tsv.gz', 'all', 'tweet')

input_vocab = Vocab.MakeFromData(dataset.GetSentences(), min_count=1)
char_vocab = Vocab.Load(os.path.join(args.expdir, 'char_vocab.pickle'))

max_word_len = max([len(x) for x in input_vocab.GetWords()]) + 2
print 'max word len {0}'.format(max_word_len)

with open(os.path.join(args.expdir, 'model_params.json'), 'r') as f:
  model_params = json.load(f)

c2v = Char2Vec(char_vocab, model_params,
               max_sequence_len=max_word_len)
the_words, word_lengths = c2v.MakeMat(input_vocab, pad_len=max_word_len)

saver = tf.train.Saver(tf.all_variables())
session = tf.Session(config=config)

saver.restore(session, os.path.join(args.expdir, 'model.bin'))

embeds = tf.nn.l2_normalize(c2v.word_embeddings, 1)

out = session.run([embeds], {c2v.words_as_chars: the_words})[0]


while True:
  print 'please input a word:'
  user_word = raw_input()

  user_chars, _ = c2v.MakeMat([user_word, 'DUMMY_WORD'],
                              pad_len=max_word_len)
  user_embed = session.run([embeds], {c2v.words_as_chars: user_chars})[0][0, :]
  sims = np.matmul(out, user_embed)
  idx = np.argsort(sims)

  for i in range(1, 21):
    index = idx[-i]
    score = sims[index]

    print '{0} {1:.3f} {2}'.format(i, score, input_vocab[index])
