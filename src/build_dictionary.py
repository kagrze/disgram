"""
Build dictionary

Usage:
  build_dictionary.py [options]

Options:
  --corpus_dir=VAL                 Westbury corpus directory
  --dataset_part=VAL               Crate dictionary for the training (0), validation (1) or tiny (2) part of the dataset [default: 0]
  --min_count=VAL                  Ignore all words with total frequency lower than this. [default: 100]
  --requested_vocabulary_size=VAL  Specify the vocabulary size (works only when min_count == 0) [default: 0]
"""

import sys
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', datefmt='%Y-%m-%d %H:%M:%S', stream=sys.stdout, level=logging.INFO)
import os
import collections
import cPickle
import gzip
from utils import UNKNOWN_WORD


if __name__ == '__main__':
    from docopt import docopt

    arg = docopt(__doc__)
    print(sys.argv)
    print(arg)

    corpus_dir = arg['--corpus_dir']
    dataset_part_id = int(arg['--dataset_part'])
    min_count = int(arg['--min_count'])
    requested_vocabulary_size = int(arg['--requested_vocabulary_size'])

    if dataset_part_id == 0:
        dataset_part = 'training'
    elif dataset_part_id == 1:
        dataset_part = 'validation'
    elif dataset_part_id == 2:
        dataset_part = 'tiny'
    else:
        raise ValueError('Unknown dataset part: ' + str(dataset_part_id))

    corpus_full_path = os.path.join(corpus_dir, 'WestburyLab.Wikipedia.Corpus_' + dataset_part + '.txt')

    logging.info('Reading ' + corpus_full_path)

    ctr = collections.Counter()

    with open(corpus_full_path, 'r') as f:
        for l in f:
            ctr.update(l.split())

    ctr[UNKNOWN_WORD] = sys.maxint

    if min_count <= 0:
        count = ctr.most_common(requested_vocabulary_size - 1)
    else:
        count = [w_c for w_c in ctr.most_common() if w_c[1] >= min_count]

    dictionary = dict()

    for word, _ in count:
        dictionary[word] = len(dictionary)

    logging.info('Dictionary size: ' + str(len(dictionary)))

    dict_file = os.path.join(corpus_dir, 'dict_' + (('mc_' + str(min_count)) if min_count > 0 else ('size_' + str(len(dictionary)))) + '_' + dataset_part + '.pkl.gz')

    logging.info('Pickling to ' + dict_file)

    with gzip.open(dict_file, 'wb') as f:
        cPickle.dump(dictionary, f)
