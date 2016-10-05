import code
import tensorflow as tf
from tensorflow.models.rnn import rnn
from tensorflow.models.rnn.rnn_cell import LSTMCell
import numpy as np


class BaseModel(object):
  """Holds code shared between all the different model variants."""

  def __init__(self, batch_size, max_sequence_len, out_vocab_size, c2v,
               dropout_keep_prob=0.0):
    self._batch_size = batch_size
    self._dropout_keep_prob = dropout_keep_prob
    self._out_vocab_size = out_vocab_size

    self.x = tf.placeholder(tf.int32, [batch_size, max_sequence_len],
                            name='x')
    self.y = tf.placeholder(tf.float32, [batch_size, out_vocab_size],
                            name='y')
    # The bidirectional rnn code requires seq_lens as int64
    self.seq_lens = tf.placeholder(tf.int64, [batch_size], name='seq_lens')
    self.example_weights = tf.placeholder(tf.float32, [batch_size],
                                          name='example_weights')

    embeddings = c2v.GetEmbeddings(self.x)
    self._inputs = [tf.squeeze(input_, [1]) for input_ in
                    tf.split(1, max_sequence_len, embeddings)]

    # Need to prepare a mask to zero out the padding symbols.

    # Make a batch_size x max_sequence_len matrix where each
    # row contains the length repeated max_sequence_len times.
    lengths_transposed = tf.expand_dims(tf.to_int32(self.seq_lens), 1)
    lengths_tiled = tf.tile(lengths_transposed, [1, max_sequence_len])

    # Make a matrix where each row contains [0, 1, ..., max_sequence_len]
    r = tf.range(0, max_sequence_len, 1)
    range_row = tf.expand_dims(r, 0)
    range_tiled = tf.tile(range_row, [batch_size, 1])

    # Use the logical operations to create a mask
    indicator = tf.less(range_tiled, lengths_tiled)
    sz = [batch_size, max_sequence_len]
    self._mask = tf.select(indicator, tf.ones(sz), tf.zeros(sz))

  def _DoPredictions(self, in_size, mats, class_weights=None):
    """Takes in an array of states and calculates predictions.

    Get the cross-entropy for each example in the vector self._xent.

    Args:
      in_size: size of the hidden state vectors
      mats: list of hidden state vectors
    """
    pred_mat = tf.get_variable('pred_mat',
                               [in_size, self._out_vocab_size])
    pred_bias = tf.get_variable('pred_bias', [self._out_vocab_size])

    # Make a prediction on every word.
    def GetWordPred(o_):
      logits = tf.nn.xw_plus_b(o_, pred_mat, pred_bias)
      return tf.nn.softmax(logits)

    self.preds_by_word = tf.pack([GetWordPred(o_) for o_ in mats])
    self.cs = self._mask / tf.reduce_sum(self._mask, 1, keep_dims=True)

    # The final prediction is the average of the predictions for each word
    # weighted by the individual confidence/utility scores.
    preds_weighted = tf.mul(tf.reshape(tf.transpose(self.cs), [-1, 1]),
                            tf.reshape(self.preds_by_word,
                                       [-1, self._out_vocab_size]))
    preds_weighted_reshaped = tf.reshape(preds_weighted,
                                         self.preds_by_word.get_shape())
    self.probs = tf.reduce_sum(preds_weighted_reshaped, 0)
    self._xent = _SafeXEnt(self.y, self.probs, class_weights=class_weights)


class WordAvgModel(BaseModel): #formerly SimpleModel
  """A bag of word /predictions/."""

  def __init__(self, out_vocab_size=None,
               batch_size=10,
               model_params=None,
               c2v=None,
               max_sequence_len=None,
               dropout_keep_prob=None,
               weights=None):

    super(WordAvgModel, self).__init__(batch_size, max_sequence_len,
                                       out_vocab_size, c2v)

    super(WordAvgModel, self)._DoPredictions(c2v.embedding_dims,
                                            self._inputs)
    self.cost = tf.reduce_mean(self.example_weights * self._xent)


class WordSeqModel(BaseModel):
  """A bag of word embeddings."""
  def __init__(self, out_vocab_size=None,
               batch_size=10,
               model_params=None,
               c2v=None,
               max_sequence_len=None,
               dropout_keep_prob=None,
               weights=None):

    super(WordSeqModel, self).__init__(batch_size, max_sequence_len,
                                      out_vocab_size, c2v)
    in_size = self._inputs[0].get_shape()[1].value

    # Also, output confidence scores at every word.
    confidence_mat = tf.get_variable('confidence_mat', [in_size, 1])
    confidence_scores = tf.concat(1, [tf.matmul(o_, confidence_mat)
                                      for o_ in self._inputs])

    # dropout on confidence_scores
    random_tensor = (1.0 - self._dropout_keep_prob +
                     tf.random_uniform(tf.shape(confidence_scores)))
    binary_tensor = -50.0 * tf.floor(random_tensor)

    csshape = confidence_scores.get_shape()
    self.cs = tf.nn.softmax(tf.constant(1.0, shape=csshape))

    # The final prediction is the average of the predictions for each word
    # weighted by the individual confidence/utility scores.

    wvs = tf.pack(self._inputs)
    wvs_weighted = tf.mul(tf.reshape(tf.transpose(self.cs), [-1, 1]),
                          tf.reshape(wvs, [-1, in_size]))
    wvs_weighted_reshaped = tf.reshape(wvs_weighted, wvs.get_shape())
    wvsum = tf.reduce_sum(wvs_weighted_reshaped,0)

    pred_mat = tf.get_variable('pred_mat', [in_size, self._out_vocab_size])
    pred_bias = tf.get_variable('pred_bias', [self._out_vocab_size])

    # Make a prediction for each tweet.
    def GetWordPred(o_):
      logits = tf.nn.xw_plus_b(o_, pred_mat, pred_bias)
      return tf.nn.softmax(logits)

    preds = GetWordPred(wvsum)
    z = tf.tile(tf.reshape(tf.reduce_sum(preds,1),[-1,1]), [1, out_vocab_size])
    self.preds, self.z = preds, z
    self.probs = tf.div(preds, z) #normalize
    self.unweighted_xent = _SafeXEnt(self.y, self.probs)

    self._xent = _SafeXEnt(self.y, self.probs, class_weights=weights)

    self.cost = tf.reduce_mean(self.example_weights * self._xent)


class TweetSeqModel(BaseModel): #formerly SeqModel
  """Single layer LSTM on top of the word embeddings.

  Lang id predictions are done on each word and then combined via
  a weighted average.
  """

  def __init__(self, out_vocab_size=None,
               batch_size=10, model_params=None,
               c2v=None,
               max_sequence_len=None,
               dropout_keep_prob=None,
               weights=None):
    """Initialize the TweetSeqModel

    Args:
      out_vocab_size: how many languages we are predicting
      batch_size: minibatch size
      model_params: dictionary of other model parameters
      c2v: char2vec class instance
      max_sequence_len: length of all the input sequences
      dropout_keep_prob: dropout probability indicator
      weights: class weights
    """
    hidden_size = model_params['model_hidden_size']
    proj_size = model_params['model_proj_size']  # optional, can be None

    super(TweetSeqModel, self).__init__(batch_size, max_sequence_len,
                                        out_vocab_size, c2v,
                                        dropout_keep_prob)

    weights = tf.constant(weights, dtype=tf.float32, name='class_weights')

    def GetCell():
      """Creates an LSTM cell with dropout."""
      c = LSTMCell(hidden_size, c2v.embedding_dims,
                   use_peepholes=model_params['peepholes'],
                   num_proj=proj_size)
      if dropout_keep_prob is not None:
        c = rnn.rnn_cell.DropoutWrapper(c, input_keep_prob=dropout_keep_prob)
      return c

    # Create the bi-directional LSTM
    with tf.variable_scope('wordrnn'):
      with tf.variable_scope('fw'):
        cell_fw = GetCell()
      with tf.variable_scope('bw'):
        cell_bw = GetCell()

      rnnout, _, _ = rnn.bidirectional_rnn(cell_fw, cell_bw, self._inputs,
                                           dtype=tf.float32,
                                           sequence_length=self.seq_lens)
      if proj_size:
        out_size = 2 * proj_size
      else:
        out_size = 2 * hidden_size
      super(TweetSeqModel, self)._DoPredictions(out_size, rnnout, class_weights=weights)

      self.cost = tf.reduce_mean(self.example_weights * self._xent)


class CharSeqModel(object): #formerly TweetSeqModel
  """
  Treats each document (tweet) as a single "word," which is fed through c2v,
  and the output "embedding" sized to be a vector of language predictions.
  """
  def __init__(self, out_vocab_size=None,
               batch_size=10, model_params=None, c2v=None,
               max_sequence_len=None,
               dropout_keep_prob=None,
               weights=None):
    self.params = model_params
    self._out_vocab_size = out_vocab_size # num. of languages
    self.weights = tf.constant(weights, dtype=tf.float32, name='class_weights')

    with tf.variable_scope("tweetff"):
      hidden = tf.get_variable("ff_hidden",
                               [c2v.embedding_dims, out_vocab_size])
      bias = tf.get_variable('ff_bias', [out_vocab_size])

    #probably useless. at least I don't want to use it
    self.seq_lens = tf.placeholder(tf.int64, [batch_size], name='seq_lens')

    self.x = tf.placeholder(tf.int32, [batch_size, max_sequence_len],
                            name='x')
    self.y = tf.placeholder(tf.float32, [batch_size, out_vocab_size],
                            name='y')
    self.example_weights = tf.placeholder(tf.float32, [batch_size],
                                          name='example_weights')

    # get one 'word' embedding for the full tweet
    tweet_embedding = c2v.GetEmbeddings(self.x)[:,1,:]

    logits = tf.nn.xw_plus_b(tweet_embedding, hidden, bias)
    self.probs = tf.nn.softmax(logits)

    self._xent = tf.nn.softmax_cross_entropy_with_logits(logits, self.y)
    self.cost = tf.reduce_mean(self.example_weights * self._xent)


class WordLevelModel(object):
  """
    Model to evaluate on word-level predictions

    Args:
      batch_size: minibatch size
      model_params: dictionary of other model parameters
      c2v: char2vec class instance
      max_sequence_len: length of all the input/output sequences
      out_vocab_size: how many languages we are predicting
      dropout_keep_prob: dropout probability indicator
      weights: class weights
  """

  def __init__(self, batch_size, model_params, c2v, max_sequence_len,
               out_vocab_size, dropout_keep_prob=0.0, weights=None):
    self._batch_size = batch_size
    self._dropout_keep_prob = dropout_keep_prob
    self._out_vocab_size = out_vocab_size

    self.x = tf.placeholder(tf.int32, [batch_size, max_sequence_len],
                            name='x')
    self.y = tf.placeholder(tf.float32,
                            [batch_size, max_sequence_len, out_vocab_size],
                            name='y')
    # The bidirectional rnn code requires seq_lens as int64
    self.seq_lens = tf.placeholder(tf.int64, [batch_size], name='seq_lens')
    self.example_weights = tf.placeholder(tf.float32, [batch_size],
                                          name='example_weights')

    embeddings = c2v.GetEmbeddings(self.x)
    self._inputs = [tf.squeeze(input_, [1]) for input_ in
                    tf.split(1, max_sequence_len, embeddings)]

    # Need to prepare a mask to zero out the padding symbols.

    # Make a batch_size x max_sequence_len matrix where each
    # row contains the length repeated max_sequence_len times.
    lengths_transposed = tf.expand_dims(tf.to_int32(self.seq_lens), 1)
    lengths_tiled = tf.tile(lengths_transposed, [1, max_sequence_len])

    # Make a matrix where each row contains [0, 1, ..., max_sequence_len]
    r = tf.range(0, max_sequence_len, 1)
    range_row = tf.expand_dims(r, 0)
    range_tiled = tf.tile(range_row, [batch_size, 1])

    self.lengths_transposed = lengths_transposed
    self.lengths_tiled = lengths_tiled
    self.range_row = range_row
    self.range_tiled = range_tiled

    # Use the logical operations to create a mask
    indicator = tf.less(range_tiled, lengths_tiled+1) #i.e. where seq len is less than index
    trim = np.ones(indicator.get_shape())
    trim[:,0] = 0 #ignore start symbol
    indicator = tf.logical_and(indicator, trim.astype(bool))
    self.indicator = indicator

    sz = [batch_size, max_sequence_len]
    self._mask = tf.select(indicator, tf.ones(sz), tf.zeros(sz))

    #-------------------------------#

    self.weights = tf.constant(weights, dtype=tf.float32, name='class_weights')

    hidden_size = model_params['model_hidden_size']
    proj_size = model_params['model_proj_size']  # optional, can be None

    def GetCell():
      """Creates an LSTM cell with dropout."""
      c = LSTMCell(hidden_size, c2v.embedding_dims,
                   use_peepholes=model_params['peepholes'],
                   num_proj=proj_size)
      if dropout_keep_prob is not None:
        c = rnn.rnn_cell.DropoutWrapper(c, input_keep_prob=dropout_keep_prob)
      return c

    # Create the bi-directional LSTM
    with tf.variable_scope('wordrnn'):
      with tf.variable_scope('fw'):
        cell_fw = GetCell()
      with tf.variable_scope('bw'):
        cell_bw = GetCell()

      rnnout, _, _ = rnn.bidirectional_rnn(cell_fw, cell_bw, self._inputs,
                                           dtype=tf.float32,
                                           sequence_length=self.seq_lens)
      if proj_size:
        out_size = 2 * proj_size
      else:
        out_size = 2 * hidden_size
      self._DoPredictions(out_size, rnnout, self.weights)

      self.cost = tf.reduce_mean(self.example_weights * self._xent)

  def _DoPredictions(self, in_size, mats, class_weights=None):
    """Takes in an array of states and calculates predictions.

    Get the cross-entropy for each example in the vector self._xent.

    Args:
      in_size: size of the hidden state vectors
      mats: list of hidden state vectors
    """
    pred_mat = tf.get_variable('pred_mat',
                               [in_size, self._out_vocab_size])
    pred_bias = tf.get_variable('pred_bias', [self._out_vocab_size])

    # Make a prediction on every word.
    def GetWordPred(o_):
      logits = tf.nn.xw_plus_b(o_, pred_mat, pred_bias)
      return tf.nn.softmax(logits)

    #self.preds_by_word1 = tf.pack([GetWordPred(o_) for o_ in mats])
    #self.preds_by_word = tf.reshape(self.preds_by_word1, self.y.get_shape())
    #self.probs = tf.mul(tf.expand_dims(self._mask,2), self.preds_by_word)

    self.preds_by_word = tf.pack([GetWordPred(o_) for o_ in mats])
    self.preds_by_instance = tf.pack([self.preds_by_word[:,i,:] for i in range(self.preds_by_word.get_shape()[1])])
    self.probs = tf.mul(tf.expand_dims(self._mask,2), self.preds_by_instance)

    self._xent = _SafeXEnt(self.y, self.probs, class_weights=class_weights, sumd=[1,2])


def _SafeXEnt(y, probs, eps=0.0001, class_weights=None, sumd=[1]):
  """Version of cross entropy loss that should not produce NaNs.

  If the predicted proability for the true class is near zero then when
  taking the log it can produce a NaN, which ruins everything. This
  function ensures each probability is at least eps and no more than one
  before taking the log.

  Args:
    y: matrix of true probabilities same size as probs
    probs: matrix of probabilities for the minibatch
    eps: value to clip the probabilities at
    class_weights: vector of relative weights to be assigned to each class
    sumd: dimensions along which to sum the x-ent matrix

  Returns:
    cross entropy loss for each example in the minibatch
  """
  adjusted_probs = tf.clip_by_value(probs, eps, 1.0 - eps)
  xent_mat = -y * tf.log(adjusted_probs)
  if class_weights is not None:
    xent_mat *= class_weights

  return tf.reduce_sum(xent_mat, sumd)


def _SafeNegEntropy(probs, batch_size, eps=0.0001):
  """Computes negative entropy in a way that will not overflow."""
  adjusted_probs = tf.clip_by_value(probs, eps, 1.0 - eps)
  entropy = tf.mul(probs, tf.log(adjusted_probs))
  return tf.reduce_sum(entropy) / batch_size
