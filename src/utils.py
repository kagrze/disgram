import os
import re
import logging
import threading
from itertools import chain
import numpy as np
import tensorflow as tf
from nltk.corpus import stopwords

stops = set(stopwords.words("english"))
stops.add("''")
stops.add(u'``')
stops.add("'s")
letters_only = re.compile(r'^[a-z]+$')
number_pattern = re.compile(r'^-?([0-9]*[.,])?[0-9]+$')
NUMBER_TOKEN = 'NUMBER'
UNKNOWN_WORD = 'UNKNOWN'
MIN_WORD_CHARS = 2


def make_dir(path):
    if not os.path.exists(path):
        logging.info('Creating dir {}'.format(path))
        os.makedirs(path)
    return path


def parse_bool(bool_as_string):
    bool_as_string = bool_as_string.lower().strip()
    if bool_as_string == 'true':
        return True
    elif bool_as_string == 'false':
        return False
    else:
        raise ValueError('Unknown boolean value: {}.'.format(bool_as_string))


def get_loss_function(loss_function_arg):
    if loss_function_arg == 'NCE':
        return tf.nn.nce_loss
    elif loss_function_arg == 'SS':
        return tf.nn.sampled_softmax_loss
    else:
        raise ValueError('Unknown loss function: ' + loss_function_arg)


def get_optimizer(optimizer_arg):
    if optimizer_arg == 'GD':
        return tf.train.GradientDescentOptimizer
    elif optimizer_arg == 'AG':
        return tf.train.AdagradOptimizer
    elif optimizer_arg == 'AD':
        return tf.train.AdadeltaOptimizer
    elif optimizer_arg == 'AM':
        return tf.train.AdamOptimizer
    elif optimizer_arg == 'FTRL':
        return tf.train.FtrlOptimizer
    elif optimizer_arg == 'MO':
        return tf.train.MomentumOptimizer
    elif optimizer_arg == 'RMS':
        return tf.train.RMSPropOptimizer
    elif optimizer_arg == 'PGD':
        return tf.train.ProximalGradientDescentOptimizer
    elif optimizer_arg == 'PAG':
        return tf.train.ProximalAdagradOptimizer
    else:
        raise ValueError('Unknown optimizer: ' + optimizer_arg)


def tokenize(line, tokenization_type):
    if tokenization_type == 'basic':
        return line.split()
    elif tokenization_type == 'wsi':
        return [(NUMBER_TOKEN if number_pattern.match(token) else token) for token in line.split() if len(token) >= MIN_WORD_CHARS and token not in stops]
    else:
        raise ValueError('Wrong tokenization type: ' + tokenization_type)


def parse_line(line, dictionary, tokenization_type):
    unknown_word_id = dictionary[UNKNOWN_WORD]
    return [dictionary[token] if token in dictionary else unknown_word_id for token in tokenize(line=line, tokenization_type=tokenization_type)]


def generate_context(doc, target_word_id, window_size, left_window_size, right_window_size):
    """
    returns global word identifiers of context
    target_word_id - position of the target word in a document
    """
    context = list(chain(
        xrange(target_word_id - left_window_size, target_word_id),
        xrange(target_word_id + 1, target_word_id + right_window_size + 1)))
    assert len(context) == window_size
    return [doc[context_word_id] for context_word_id in context]


def generate_cbow_batches(requested_batch_size, dictionary, trainset_file, window_size, left_window_size, right_window_size, tokenization_type):
    """
    Generate CBOW word embedding mini-batches.
    Data mini-batch is a list of tuples (batch_id, context_label)
    It is assumed that trainset_file is a multi-line plain text file
    """
    logging.info('Generating CBOW mini-batches')

    batch_id = 0
    context = []
    current_position_in_batch = 0
    assert window_size == left_window_size + right_window_size
    context_label = np.ndarray(shape=(requested_batch_size, window_size + 1), dtype=np.int32)

    with open(trainset_file, 'r') as f:
        for l in f:
            doc = parse_line(l, dictionary, tokenization_type)
            if len(doc) < window_size + 1:
                continue

            for target_word_id in xrange(left_window_size, len(doc) - right_window_size):
                context_label[current_position_in_batch, :-1] = generate_context(doc, target_word_id, window_size, left_window_size, right_window_size)
                context_label[current_position_in_batch, -1] = doc[target_word_id]
                current_position_in_batch += 1
                if current_position_in_batch == requested_batch_size:
                    yield (batch_id, context_label.copy())  # we return a COPY of context_label ndarray, therefore we do not need to reinitialize it
                    current_position_in_batch = 0
                    batch_id += 1


class ThreadSafeIterator(object):
    def __init__(self, unsafe_iterator):
        self._unsafe_iterator = unsafe_iterator
        self._lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self._lock:
            return self._unsafe_iterator.next()


def build_and_start_thread_pool(job, kwargs, worker_number):
    """
    this is ugly; for each epoch we create new thread pool
    """
    workers = []
    for i in xrange(worker_number):
        workers.append(threading.Thread(target=job, kwargs=kwargs))

    for t in workers:
        t.start()

    for t in workers:
        t.join()


def file_len(fname):
    if not os.path.isfile(fname):
        return 0
    with open(fname) as f:
        for i, l in enumerate(f, 1):
            pass
    return i
