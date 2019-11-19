"""
Prepare training and validation set based on the Wesbury Lab Wikipedia corpus. Data is lowercased, stopwords removed,
integers converted to 'number' and each sentence is placed in a separate line.
The validation set is created by randomly selected specific percentage of documents from training set.

Usage:
  prepare.py [options]

Options:
  --original_corpus=VAL       Original corpus
  --training_set=VAL          Training set
  --validation_set=VAL        Validation set
  --tiny_set=VAL              Tiny set
  --valid_set_size=VAL        Validation set size [default: 0.2]
  --tiny_set_size=VAL         Validation set size [default: 0.001]
  --min_setn_chars=VAL        Minimal number of sentence characters (including spaces) [default: 10]
  --min_setn_words=VAL        Minimal number of sentence words [default: 2]
  --min_word_chars=VAL        Minimal number of word characters [default: 2]
  --encoding=VAL              Encoding [default: utf-8]
  --split_compound_words=VAL  Split compound words [default: True]
  --az_char_only=VAL          Replace words not matching patter [a-z]+ with UNKNOWN char only [default: True]
"""

import logging
import sys
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', datefmt='%Y-%m-%d %H:%M:%S', stream=sys.stdout, level=logging.INFO)
from itertools import chain
import numpy as np
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk import sent_tokenize
from nltk import word_tokenize
from utils import UNKNOWN_WORD, NUMBER_TOKEN, letters_only, number_pattern, stops


def is_known(token):
    return token == NUMBER_TOKEN or letters_only.match(token)


def local_word_tokenize(sent, min_word_chars, split_compound_words, az_char_only):
    tokens = word_tokenize(sent)
    if split_compound_words:
        tokens = chain.from_iterable([token.split('-') for token in tokens])
        tokens = chain.from_iterable([token.split('/') for token in tokens])
    tokens = [(NUMBER_TOKEN if number_pattern.match(t) else t) for t in tokens if len(t) >= min_word_chars and t not in stops]
    if az_char_only:
        tokens = [(token if is_known(token) else UNKNOWN_WORD) for token in tokens]
    return tokens


if __name__ == '__main__':
    from docopt import docopt

    np.random.seed(123)

    arg = docopt(__doc__)
    print(sys.argv)
    print(arg)

    original_corpus = arg['--original_corpus']
    training_set = arg['--training_set']
    validation_set = arg['--validation_set']
    tiny_set = arg['--tiny_set']
    valid_set_size = float(arg['--valid_set_size'])
    tiny_set_size = float(arg['--tiny_set_size'])
    min_setn_chars = int(arg['--min_setn_chars'])
    min_setn_words = int(arg['--min_setn_words'])
    min_word_chars = int(arg['--min_word_chars'])
    encoding = arg['--encoding']
    split_compound_words = arg['--split_compound_words'] == str(True)
    az_char_only = arg['--az_char_only'] == str(True)

    documents = []
    document = []

    logging.info('Reading ' + original_corpus)

    with open(original_corpus, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or len(line) < min_setn_chars:
                continue
            if line == '---END.OF.DOCUMENT---':
                if document:
                    documents.append(document)
                    document = []
                    if len(documents) % 10000 == 0:
                        logging.info('{} documents processed'.format(len(documents)))
            else:
                line = line.replace('IT', 'information technology')
                for sent in sent_tokenize(line.lower().decode(encoding)):
                    tokens = local_word_tokenize(sent, min_word_chars, split_compound_words, az_char_only)
                    if len(tokens) < min_setn_words:
                        continue
                    sent = ' '.join(tokens)
                    if len(sent) >= min_setn_chars:
                        document.append(sent)

    logging.info('Loaded {} documents'.format(len(documents)))

    assert len(documents) <= 3035070  # number of document separators in the original corpus file

    logging.info('Writing to ' + training_set)

    with open(training_set, 'w') as f:
        for doc in documents:
            for sent in doc:
                f.write(sent.encode(encoding) + '\n')

    logging.info('Shuffling')

    order = range(len(documents))
    np.random.shuffle(order)

    if validation_set:
        logging.info('Writing to ' + validation_set)

        with open(validation_set, 'w') as f:
            for doc_id in order[: int(valid_set_size * len(documents))]:
                for sent in documents[doc_id]:
                    f.write(sent.encode(encoding) + '\n')

    if tiny_set:
        logging.info('Writing to ' + tiny_set)

        with open(tiny_set, 'w') as f:
            for doc_id in order[: int(tiny_set_size * len(documents))]:
                for sent in documents[doc_id]:
                    f.write(sent.encode(encoding) + '\n')
