# Misc. helper functions go in this file.
import collections
import numpy as np
import os
import random
import sys

# Need to use these three lines to import pyplot with a good font
import matplotlib
matplotlib.rc('font', family='DejaVu Sans')
from matplotlib import pyplot


def Metrics(preds, labs, show=True):
  """Print precision, recall and F1 for each language.

  Assumes a single language per example, i.e. no code switching.

  Args:
    preds: list of predictions
    labs: list of labels
    show: flag to toggle printing
  """
  all_langs = set(preds + labs)
  preds = np.array(preds)
  labs = np.array(labs)

  label_totals = collections.Counter(labs)
  pred_totals = collections.Counter(preds)
  confusion_matrix = collections.Counter(zip(preds, labs))

  num_correct = 0
  for lang in all_langs:
    num_correct += confusion_matrix[(lang, lang)]
  acc = num_correct / float(len(preds))
  print 'accuracy = {0:.3f}'.format(acc)

  if show:
    print ' Lang     Prec.   Rec.   F1'
    print '------------------------------'

  scores = []
  fmt_str = '  {0:6}  {1:6.2f} {2:6.2f} {3:6.2f}'
  for lang in sorted(all_langs):
    idx = preds == lang
    total = max(1.0, pred_totals[lang])
    precision = 100.0 * confusion_matrix[(lang, lang)] / total

    idx = labs == lang
    total = max(1.0, label_totals[lang])
    recall = 100.0 * confusion_matrix[(lang, lang)] / total

    if precision + recall == 0.0:
      f1 = 0.0
    else:
      f1 = 2.0 * precision * recall / (precision + recall)

    scores.append([precision, recall, f1])
    if show:
      print fmt_str.format(lang, precision, recall, f1)

  totals = np.array(scores).mean(axis=0)
  if show:
    print '------------------------------'
    print fmt_str.format('Total:', totals[0], totals[1], totals[2])
  return totals[2]


def ConfusionMat(preds, labs):
  """Plot and show a confusion matrix.

  Args:
    preds: list of predicted labels
    labs: list of true labels
  """
  all_langs = set(preds + labs)  # this is the set of all possible labels
  num_langs = len(all_langs)

  # create a mapping from labels to id numbers
  lookup = dict(zip(sorted(all_langs), range(num_langs)))

  # make the counts for the confusion matrix
  counts = np.zeros((num_langs, num_langs))
  for p, l in zip(preds, labs):
    counts[lookup[p], lookup[l]] += 1

  # plot a colormap using log scale
  pyplot.imshow(np.log(counts+1.0), interpolation='none')

  # plot the text labels
  for i in xrange(num_langs):
    for j in xrange(num_langs):
      pyplot.text(i, j, str(int(counts[i, j])), color='white',
                  horizontalalignment='center')
  # take care of the axes
  pyplot.xticks(range(num_langs), sorted(all_langs))
  pyplot.yticks(range(num_langs), sorted(all_langs))
  pyplot.xlabel('Prediction')
  pyplot.ylabel('True Label')
  pyplot.show()


def GetColor(percent):
  """Returns an RGB color scale to represent a given percentile."""
  z = int(percent * 512)
  z = 255 - min(255, z)
  hexz = hex(z)[2:]
  if len(hexz) == 1:
    hexz = '0' + hexz
  elif len(hexz) == 0:
    hexz = '00'
  color = '#ff{0}{0}'.format(hexz)
  return color


def PrintParams(param_list, handle=sys.stdout.write):
  """Print the names of the parameters and their sizes.

  Args:
    param_list: list of tensors
    handle: where to write the param sizes to
  """
  handle('NETWORK SIZE REPORT\n')
  param_count = 0
  fmt_str = '{0: <25}\t{1: >12}\t{2: >12,}\n'
  for p in param_list:
    shape = p.get_shape()
    shape_str = 'x'.join([str(x.value) for x in shape])
    handle(fmt_str.format(p.name, shape_str, np.prod(shape).value))
    param_count += np.prod(shape).value
  handle(''.join(['-'] * 60))
  handle('\n')
  handle(fmt_str.format('total', '', param_count))
  if handle==sys.stdout.write:
    sys.stdout.flush()


def GetProj(feat_mat):
  """Projects a feature matrix into 2 dimensions using PCA."""
  m = feat_mat.mean(axis=0)
  feat_mat -= m

  covmat = feat_mat.T.dot(feat_mat)
  m, v = np.linalg.eig(covmat)
  m /= m.sum()

  v = v[:, 0:2]
  proj = feat_mat.dot(v)
  return proj


def PlotText(pts, labels):
  pyplot.plot(pts[:, 0], pts[:, 1], 'x')
  for i in xrange(pts.shape[0]):
    pyplot.text(pts[i, 0], pts[i, 1], labels[i])
  pyplot.show()


def Graphemes(s):
  """ Given a string return a list of graphemes.

  Args:
    s the input string

  Returns:
    A list of graphemes.
  """
  graphemes = []
  current = []

  if type(s) == unicode:
    s = s.encode('utf8')

  for c in s:
    val = ord(c) & 0xC0
    if val == 128:
      # this is a continuation
      current.append(c)
    else:
      # this is a new grapheme
      if len(current) > 0:
        graphemes.append(''.join(current))
        current = []

      if val < 128:
        graphemes.append(c)  # single byte grapheme
      else:
        current.append(c)  # multi-byte grapheme

  if len(current) > 0:
    graphemes.append(''.join(current))

  return graphemes


def Bytes(s):
  if type(s) == unicode:
    s = s.encode('utf8')
  z = s.encode('hex')
  return [z[2*i:2*i+2] for i in range(len(z)/2)]


def GetLangName(code):
  """Convert ISO lang codes to full language names.

  Args:
    code: an ISO language code as a string

  Returns:
    name of language or ISO code if language name not available
  """
  names = {
    'am': 'amharic',
    'ar': 'arabic',
    'bg': 'bulgarian',
    'bn': 'bengali',
    'bo': 'tibetan',
    'bs': 'bosnian',
    'ca': 'catalan',
    'ckb': 'kurdish',
    'cs': 'czech',
    'cy': 'welsh',
    'da': 'danish',
    'de': 'german',
    'dv': 'divehi',
    'el': 'greek',
    'en': 'english',
    'es': 'spanish',
    'et': 'estonian',
    'eu': 'basque',
    'fa': 'persian',
    'fi': 'finnish',
    'fr': 'french',
    'gl': 'galician',
    'gu': 'gujarati',
    'he': 'hebrew',
    'hi': 'hindi',
    'hi-Latn': 'hindi-latin',
    'hr': 'croatian',
    'ht': 'haitian',
    'hu': 'hungarian',
    'hy': 'armenian',
    'id': 'indonesian',
    'is': 'icelandic',
    'it': 'italian',
    'ja': 'japanese',
    'ka': 'georgian',
    'km': 'cambodian',
    'kn': 'kannada',
    'ko': 'korean',
    'lo': 'lao',
    'lt': 'lithuanian',
    'lv': 'latvian',
    'ml': 'malayalam',
    'mr': 'marathi',
    'my': 'burmese',
    'ne': 'nepali',
    'nl': 'dutch',
    'no': 'norwegian',
    'or': 'oriya',
    'pa': 'punjabi',
    'pl': 'polish',
    'ps': 'pashto',
    'pt': 'portuguese',
    'ro': 'romanian',
    'ru': 'russian',
    'sd': 'sindhi',
    'si': 'sinhala',
    'sk': 'slovak',
    'sl': 'slovene',
    'sr': 'serbian',
    'sv': 'swedish',
    'ta': 'tamil',
    'te': 'telugu',
    'th': 'thai',
    'tl': 'tagalog',
    'tr': 'turkish',
    'ug': 'uighur',
    'uk': 'ukranian',
    'ur': 'urdu',
    'vi': 'vietnamese',
    'zh-CN': 'chinese',
    'zh-TW': 'taiwanese'
    }

  if code in names:
    return names[code]
  return code
