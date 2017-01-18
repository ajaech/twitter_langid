"""This file holds the Dataset class, which helps with the loading
and organizing of the training data."""
import argparse
import collections
import gzip
import itertools
import numpy as np
import re
import sys


np.random.seed(666)


def LoadData(filename, mode='train', model='tweet'):
  """Load data stored in tweetlid format.
  (i.e. tab-separated tweetid, language, tweet)

  Partitioning between train/dev/eval is done by the last digit of the
  id number for each training example. Digits 2 through 9 are used for
  training and 1 is used as a dev set. Currently, the eval set is
  never loaded and confusingly to load the dev set you have to use
  'eval' for the mode argument.

  This function splits the tweet into a list of units. The level of
  splitting is controlled by the model argument.

  Args:
    filename: where to get the data
    mode: train or eval or all
    unit: word, tweets or chars

  Returns:
    tuple of sentences, labels and ids
  """
  ids, labels, sentences = [], [], []
  with gzip.open(filename, 'r') as f:
    for line in f:
      tweetid, lang, tweet = line.split('\t')

      idx = int(tweetid) % 10  # use this to partition data
      if mode == 'train' and idx < 2:
        continue
      if mode == 'eval+final' and idx > 2:
        continue
      if mode == 'eval' and idx != 1:
        continue
      if mode == 'final' and idx != 0:
        continue

      ids.append(tweetid)
      # split used to handle code switching tweets
      labels.append(re.split(r'\+|/', lang))

      # The { and } brackets are used for start/end symbols
      if model in ['word', 'tweet']:
        #split on whitespace to get words
        sentences.append(['{'] + [unicode(x_.decode('utf8'))
                                  for x_ in tweet.split()] + ['}'])
      elif model=='char':
        #include full tweet as single unicode string (list of length 3)
        sentences.append([u'{'] + [unicode(tweet.decode('utf8'))] + [u'}'])

      else:
        msg = 'Invalid unit type <{0}> for tokenizing tweet'.format(model)
        raise ValueError(msg)

  print '{0} examples loaded'.format(len(sentences))
  return sentences, labels, ids


class Dataset(object):

  def __init__(self, batch_size, preshuffle=True, name='unnamed'):
    """Init the dataset object.

    Args:
      batch_size: size of mini-batch
      preshuffle: should the order be scrambled before the first epoch
      name: optional name for the dataset
    """
    self._sentences = []
    self._labels = []
    self._ids = []
    self.dataset_weights = []
    self._lines = []
    self.name = name

    self.batch_size = batch_size
    self.preshuffle = preshuffle

  def ReadData(self, filename, mode, modeltype, weight=1.0):
    d = LoadData(filename, mode, modeltype)
    self.AddDataSource(d, weight=weight)

  def AddDataSource(self, data, weight=1.0):
    sentences, labels, ids = data
    self._sentences.append(sentences)
    self._labels.append(labels)
    self.dataset_weights.append(weight)
    self._ids.append(ids)

  def GetSentences(self):
    return itertools.chain(*self._sentences)

  def GetIds(self):
    return [x for x in itertools.chain(*self._ids)]

  def GetLabelSet(self):
    label_set = set()
    for d in self._labels:
      label_set.update([x for x in itertools.chain(*d)])
    return label_set

  def Prepare(self,in_vocab,out_vocab,und_label='und',ignore_categories=[]):
    # Add a dummy dataset to make the batch size evenly divide the number
    # of sentences.
    batch_size = self.batch_size
    total_sentences = sum([len(x) for x in self._sentences])
    r = total_sentences % batch_size
    if r > 0:
      n = batch_size - r
      self.AddDataSource(([list('{dummy}')] * n, [[und_label]] * n, [0] * n),
                         weight=0.0)

    sentences = list(itertools.chain(*self._sentences))
    labels = list(itertools.chain(*self._labels))

    self.example_weights = []
    for i in xrange(len(self.dataset_weights)):
      w = self.dataset_weights[i]
      for _ in xrange(len(self._sentences[i])):
        self.example_weights.append(w)
    self.example_weights = np.array(self.example_weights)

    self.seq_lens = np.array([len(x) for x in sentences])
    self.max_sequence_len = self.seq_lens.max()

    self.batch_size = batch_size
    self.current_idx = 0

    self.sentences = self.GetNumberLines(sentences, in_vocab,
                                         self.max_sequence_len)
    self.labels = np.zeros((len(labels), len(out_vocab)))
    for i, w in enumerate(labels):
      for w_ in w:
        self.labels[i, out_vocab[w_]] = 1.0
      self.labels[i, :] /= self.labels[i, :].sum()

    # class weights
    # There is a hack to only use the examples with weight 1 as a way
    # to prevent wikipedia from dominating the weights.
    counts = self.labels[self.example_weights == 1, :].sum(axis=0)
    self.w = 1.0/(1.0 + counts)
    self.w /= self.w.mean()  # scale the class weights to reasonable values

    self.N = len(sentences)
    if self.preshuffle:
      self._Permute()

  @staticmethod
  def GetNumberLines(lines, vocab, pad_length):
    """Convert list of words to matrix of word ids."""
    out = []
    for line in lines:
      if len(line) < pad_length:
        line += ['}'] * (pad_length - len(line))
      out.append([vocab[w] for w in line])
    return np.array(out)

  def GetNumBatches(self):
    """Returns num batches per epoch."""
    return self.N / self.batch_size

  def _Permute(self):
    """Shuffle the training data."""
    s = np.arange(self.N)
    np.random.shuffle(s)

    self.sentences = self.sentences[s, :]
    self.seq_lens = self.seq_lens[s]
    self.labels = self.labels[s, :]
    self.example_weights = self.example_weights[s]

  def GetNextBatch(self):
    if self.current_idx + self.batch_size > self.N:
      self.current_idx = 0
      self._Permute()

    idx = range(self.current_idx, self.current_idx + self.batch_size)
    self.current_idx += self.batch_size

    return (self.sentences[idx, :], self.seq_lens[idx],
            self.labels[idx, :], self.example_weights[idx])


# print some dataset statistics
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('data')
  args = parser.parse_args()

  _, labels, _ = LoadData(args.data, 'all')

  total = len(labels)
  print 'total sentences: {0}'.format(total)
  unique_labels = len(set([tuple(x) for x in labels]))
  print 'unique labels: {0}'.format(unique_labels)

  counts = collections.Counter([tuple(s) for s in labels])

  for lang in sorted(counts.keys()):
    print '{0}\t{1}\t{2:.1f}'.format(' '.join(lang), counts[lang],
                                     100 * counts[lang] / float(total))
