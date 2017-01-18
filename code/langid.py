import sys
reload(sys)  # need to reload to set default encoding
sys.setdefaultencoding('utf8')

import argparse
import collections
import json
import logging
import numpy as np
import os
import shutil
import tensorflow as tf
import util

from char2vec import CharCNN as Char2Vec
from batcher import Dataset
from vocab import Vocab
from models import WordSeqModel, CharSeqModel, TweetSeqModel, WordLevelModel


parser = argparse.ArgumentParser()
parser.add_argument('expdir')
parser.add_argument('--start', help='init parameters from saved model')
parser.add_argument('--mode', choices=['train', 'debug', 'eval', 'final',
                                       'apply'], default='train')
parser.add_argument('--data', default='../data/smallwiki.tsv.gz')
parser.add_argument('--params', default='default_params.json', 
                    help='load hyperparams from json file')
parser.add_argument('--model', choices=['word', 'char', 'tweet'],
                    help='pass "word", "char", or "tweet" to use '\
                    'WordSeqModel, CharSeqModel or TweetSeqModel',
                    default='tweet') #default is hierarchical model
args = parser.parse_args()

if not os.path.exists(args.expdir):
  os.mkdir(args.expdir)
mode = args.mode

if mode == 'train':
  logging.basicConfig(filename=os.path.join(args.expdir, 'logfile.txt'),
                      level=logging.INFO)
config = tf.ConfigProto(inter_op_parallelism_threads=10,
                        intra_op_parallelism_threads=10)
tf.set_random_seed(666)

baseline = False

batch_size = 25
dataset = Dataset(batch_size, preshuffle=mode=='train')
und_symbol='und'

dataset.ReadData(args.data, mode, args.model)

# Make the input vocabulary (words that appear in data)
if baseline:
  # The baseline is to use fixed word embeddings.
  if mode == 'train':
    # The input vocab is fixed during training.
    input_vocab = Vocab.MakeFromData(dataset.GetSentences(), min_count=2)
    input_vocab.Save(os.path.join(args.expdir, 'input_vocab.pickle'))
  else:
    # During testing we need to load the saved input vocab.
    input_vocab = Vocab.Load(os.path.join(args.expdir, 'input_vocab.pickle'))
else:
  # The open vocabulary can be regenerated with each run.
  min_count = 1
  if mode == 'debug':
    min_count = 10  # When visualizing word embeddings hide rare words
  maxlens = {'word':40, 'char':150, 'tweet':40}
  input_vocab = Vocab.MakeFromData(dataset.GetSentences(),
                                   min_count=min_count,
                                   max_length=maxlens[args.model])

if mode == 'train':
  # Make the character vocabulary
  if args.start:
    shutil.copyfile(os.path.join(args.start, 'char_vocab.pickle'),
                    os.path.join(args.expdir, 'char_vocab.pickle'))
    char_vocab = Vocab.Load(os.path.join(args.expdir, 'char_vocab.pickle'))
    with open(os.path.join(args.start, 'model_params.json'), 'r') as f:
      model_params = json.load(f)
  else:
    x = [util.Graphemes(w) for w in input_vocab.GetWords()]
    char_vocab = Vocab.MakeFromData(x, min_count=2)
    char_vocab.Save(os.path.join(args.expdir, 'char_vocab.pickle'))

    with open(args.params, 'r') as f:
      model_params = json.load(f)

  with open(os.path.join(args.expdir, 'model_params.json'), 'w') as f:
    json.dump(model_params, f)
else:  # eval or debug mode
  char_vocab = Vocab.Load(os.path.join(args.expdir, 'char_vocab.pickle'))
  with open(os.path.join(args.expdir, 'model_params.json'), 'r') as f:
    model_params = json.load(f)

# Make the output vocab (the set of possible languages to predict)
output_vocab_filename = os.path.join(args.expdir, 'out_vocab.pickle')
if mode == 'train':
  labels = [[x] for x in dataset.GetLabelSet()]
  if [und_symbol] not in labels: labels += [[und_symbol]]
  output_vocab = Vocab.MakeFromData(labels, min_count=1,
                                    no_special_syms=True)
  output_vocab.Save(output_vocab_filename)
else:
  output_vocab = Vocab.Load(output_vocab_filename)

ignore_categories = ['mixed', 'ambiguous', 'fw', 'unk', '{', '}']
dataset.Prepare(input_vocab, output_vocab, und_symbol, ignore_categories)

#Set other hyperparameters and create model
max_word_len = max([len(x) for x in input_vocab.GetWords()]) + 2
print 'max word len {0}'.format(max_word_len)

dropout_keep_prob = tf.placeholder_with_default(1.0, (), name='keep_prob')
if baseline:
  c2v = BasicEmbedding(model_params, vocab_size=len(input_vocab))
else:                    
  c2v = Char2Vec(char_vocab, model_params,
                 max_sequence_len=max_word_len,
                 dropout_keep_prob=dropout_keep_prob)
  the_words, word_lengths = c2v.MakeMat(input_vocab, pad_len=max_word_len)

models = {'word': WordSeqModel, 'char': CharSeqModel, 'tweet': TweetSeqModel}
if args.data == 'codeswitch':
  models['tweet'] = WordLevelModel

model = models[args.model](batch_size=batch_size, model_params=model_params,
                           max_sequence_len=dataset.max_sequence_len,
                           dropout_keep_prob=dropout_keep_prob,
                           out_vocab_size=len(output_vocab),
                           weights=dataset.w, c2v=c2v)
saver = tf.train.Saver(tf.all_variables())
session = tf.Session(config=config)


def Apply(expdir):
  saver.restore(session, os.path.join(expdir, 'model.bin'))

  results = collections.defaultdict(list)

  for _ in xrange(dataset.GetNumBatches()):
    words, seqlen, labs, _ = dataset.GetNextBatch()
    batch_data = MakeFeedDict(words, seqlen, labs)

    cvocab = char_vocab
    batch_vocab, words_remapped = Char2Vec.GetBatchVocab(words)
    charseqs = the_words[batch_vocab]
    hh, hidx = session.run([c2v.hh, c2v.hidx], batch_data)
    
    for word_i in range(len(batch_vocab)):
      for filter_i in range(hh.shape[-1]):
        activation = hh[word_i, filter_i]
        if activation == 0.0:
          continue
        loc = hidx[word_i, 0, filter_i]
        char_seq = charseqs[word_i, loc:loc+4]
        unit = ''.join([cvocab[i] for i in char_seq])
        results[filter_i].append(('{0:.1f}'.format(activation), unit))
      
  for filtnum in results:
    dedup = collections.Counter(results[filtnum])
    z = sorted(dedup.keys(), key=lambda x: -float(x[0]))

    with open('filters/{0}.txt'.format(filtnum), 'w') as f:
      for i in range(len(z)):
        first = '{0} {1} {2}\n'.format(z[i][0], z[i][1], dedup[z[i]])
        f.write(first)

def Eval(expdir):
  """Evaluates on dev data.

  Writes results to a results.tsv file in the expdir for use in the 
  scoring script.

  Args:
    expdir: path to experiment directory
  """ 
  if args.data == 'codeswitch':
    return EvalPerWord(expdir)

  saver.restore(session, os.path.join(expdir, 'model.bin'))

  all_preds, all_labs = [], []
  for _ in xrange(dataset.GetNumBatches()):
    words, seqlen, labs, weights  = dataset.GetNextBatch()
    batch_data = MakeFeedDict(words, seqlen, labs)

    if args.model in ["word", "tweet"]:
      model_vars = [model.probs, model.preds_by_word]
      probs, pp = session.run(model_vars, batch_data)

    elif args.model in ["char"]:
      probs, pp = session.run([model.probs, model.preds], 
                              batch_data)
      
    idx = weights != 0
    all_preds += [output_vocab[p] for p in np.argmax(probs[idx, :], axis=1)]
    all_labs += [output_vocab[p] for p in np.argmax(labs[idx, :], axis=1)]

  util.Metrics(all_preds, all_labs)

  # This output file is in the format needed to score for TweetLID
  ids = dataset.GetIds()
  with open(os.path.join(expdir, 'results.tsv'), 'w') as f:
    for idnum, p in zip(ids, all_preds):
      f.write('{0}\t{1}\n'.format(idnum, p))

  
def Train(expdir):
  logging.info('Input Vocab Size: {0}'.format(len(input_vocab)))
  logging.info('Char Vocab Size: {0}'.format(len(char_vocab)))

  tvars = tf.trainable_variables()
  grads, _ = tf.clip_by_global_norm(tf.gradients(model.cost, tvars), 5.0)
  optimizer = tf.train.AdamOptimizer(0.001)
  train_op = optimizer.apply_gradients(zip(grads, tvars))

  session.run(tf.initialize_all_variables())
  if args.start:
    saver.restore(session, os.path.join(args.start, 'model.bin'))
  util.PrintParams(tf.trainable_variables(), handle=logging.info)

  maxitr = model_params.get('num_training_iters', 80001)
  print "Training for {} iterations...".format(maxitr)
  for idx in xrange(maxitr): 
    if args.data == 'codeswitch':
      words, seqlen, labs, ws, lines = dataset.GetNextBatch()
    else:
      words, seqlen, labs, ws = dataset.GetNextBatch()
    batch_data = MakeFeedDict(words, seqlen, labs, ws)

    if idx % 25 == 0:
      probs = session.run([model.probs], batch_data)[0]
      s = [input_vocab[i] for i in words[0, :seqlen[0]]]
      print ' '.join(s)
      if args.data != 'codeswitch':
        print "Predicted:", GetTopPreds(probs[0, :])
        print "Actual:", GetTopPreds(labs[0,:])
      else:
        print "Predict:", GetTopWordLevelPreds(probs[0,:,:], seqlen[0])
        print "Actual: ", GetTopWordLevelPreds(labs[0,:,:], seqlen[0])

    cost, _ = session.run([model.cost, train_op], batch_data)
    logging.info({'iter': idx, 'cost': float(cost)})

    if idx == maxitr-1 or (idx % 2000 == 0 and idx > 0):
      saver.save(session, os.path.join(expdir, 'model.bin'))


def Debug(expdir):
  """Plots language and word embeddings from saved model."""
  saver.restore(session, os.path.join(expdir, 'model.bin'))

  # Plot the language embeddings
  z = [x for x in tf.trainable_variables() if 'pred_mat' in x.name][0]
  zz = z.eval(session)
  c = util.GetProj(zz.T)
  lang_names = [util.GetLangName(output_vocab[i]) 
                for i in xrange(len(output_vocab))]
  util.PlotText(c, lang_names) 

  # plot some word embeddings 
  batch_data = {c2v.words_as_chars: the_words}
  word_embeddings = session.run([c2v.word_embeddings], batch_data)[0]
  c = util.GetProj(word_embeddings)
  util.PlotText(c, input_vocab)
  

def GetTopPreds(probs):
  top_preds = []
  for ii in reversed(np.argsort(probs)):
    if probs[ii] > 0.05:
      top_preds.append('{0}: {1:.1f}'.format(output_vocab[ii],
                                             100.0 * probs[ii]))
  return top_preds


def MakeFeedDict(words, seqlen, labs, ws=None):
  """Create the feed dict to process each batch.

  All the inputs should be from the GetNextBatch command.

  Args:
    words: matrix of word ids
    seqlen: vector of sequence lengths
    labs: target matrix
    ws: per-example weights

  Returns:
    dictionary to be used as feed dict.
  """
  batch_data = {
    model.seq_lens: seqlen,
    model.x: words,
  }

  if mode == 'train':
    batch_data[model.y] = labs
    batch_data[model.example_weights] = ws
    batch_data[dropout_keep_prob] = model_params['dropout_keep_prob']

  if not baseline:
    batch_vocab, words_remapped = Char2Vec.GetBatchVocab(words)

    batch_data.update({
      c2v.words_as_chars: the_words[batch_vocab, :],
      model.x: words_remapped
    })

    if hasattr(c2v, 'seq_lens'):
      batch_data.update({
        c2v.seq_lens: word_lengths[batch_vocab],
        c2v.batch_dim: len(batch_vocab)
        })

    return batch_data


funcs = {'train': Train,  # Train the model.
         'debug': Debug,  # Plot some graphs.
         'apply': Apply, # Apply model to new data.
         'eval': Eval,    # Evaluate on dev set.
         'final': Eval}  # Evaluate on eval set.
funcs[mode](args.expdir)
