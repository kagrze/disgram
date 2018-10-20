import os
import numpy as np
import logging
import csv
from scipy.spatial.distance import cosine as cosine_distance
from scipy.stats import spearmanr
import re
from itertools import chain
from utils import parse_line
from sklearn.metrics.cluster import adjusted_rand_score

wordsim353_dir = '/net/archive/groups/plgg-dlcuda/datasets/wordsimilarity/wordsim353/downloads'
scws_dir = '/net/archive/groups/plgg-dlcuda/datasets/wordsimilarity/scws/downloads/SCWS'
simlex999_dir = '/net/archive/groups/plgg-dlcuda/datasets/wordsimilarity/SimLex999/SimLex-999'
rw_dir = '/net/archive/groups/plgg-dlcuda/datasets/wordsimilarity/rareword/rw'

SCWS_CONTEXT = re.compile(r'^(.*)<b>(.+)</b>(.+)$')

PRIOR_PRUNING_THRESHOLD = 0.05
# PRIOR_PRUNING_THRESHOLD = 0.0


def scaled_cosine_similarity(vector1, vector2):
    """
    Returns number between 0 and 1. Two equal vectors return similarity 1. Two oposite vectors return 0.
    How should we treat vectors with an angle between them bigger than 90 degree?
    Should we ignore the direction and just return abs. value of the cos.?
    """
    assert vector1.shape == vector2.shape
    assert len(vector1.shape) == 1
    return (2.0 - cosine_distance(vector1, vector2)) / 2


def get_nn(embeddings, word_id, nn_number):
    """
    Returns ids of nn_number embeddings having smallest cosine distance
    Embeddings have to be normalized prior to calling this fction
    """
    return np.argsort(-np.dot(embeddings, embeddings[word_id, :]))[1:nn_number + 1]


def get_euclidean_nn(embeddings, word_id, nn_number):
    return np.argsort(np.linalg.norm(x=(embeddings - embeddings[word_id, :]), axis=1))[1:nn_number + 1]


def print_nearest_neighbors(embeddings, sense_number, dictionary, reverse_dictionary, word, priors):
    if priors is not None:
        pruned_senses = priors[dictionary[word], :] < PRIOR_PRUNING_THRESHOLD
    else:
        pruned_senses = [False for s in xrange(sense_number)]

    for s in xrange(sense_number):
        if pruned_senses[s]:
            continue
        logging.info('Nearest neighbors of ' + word + '_' + str(s) + ' are:')
        for i in get_nn(embeddings, dictionary[word] * sense_number + s, 50):
            word_id = i / sense_number
            sense_id = i % sense_number
            if priors is not None:
                pruned_neighbor_senses = priors[word_id, :] < PRIOR_PRUNING_THRESHOLD
                if pruned_neighbor_senses[sense_id]:
                    continue
            logging.info(' ' + reverse_dictionary[word_id] + '_' + str(sense_id))
        logging.info('')


def test_context_independent(embeddings, sense_number, dictionary, testset_file, priors, prior_weighting, delimiter, skip_header, human_similarity_col_id, first_word_col_id=0, second_word_col_id=1, quoting=csv.QUOTE_MINIMAL):
    with open(testset_file, 'r') as csvfile:
        human_similarities = []
        avg_model_similarities = []
        csv_reader = csv.reader(csvfile, delimiter=delimiter, strict=True, quoting=quoting)
        if skip_header:
            next(csv_reader)
        for row in csv_reader:
            word1 = row[first_word_col_id].lower()
            word2 = row[second_word_col_id].lower()
            if word1 not in dictionary or word2 not in dictionary:
                # if word1 not in dictionary:
                #     logging.warning('Word \'{}\' not in the dictionary! Score will have smaller precision.'.format(word1))
                # if word2 not in dictionary:
                #     logging.warning('Word \'{}\' not in the dictionary! Score will have smaller precision.'.format(word2))
                continue
            word1id = dictionary[word1]
            word2id = dictionary[word2]

            if priors is not None:
                senses_after_pruning_1 = priors[word1id, :] >= PRIOR_PRUNING_THRESHOLD
                senses_after_pruning_2 = priors[word2id, :] >= PRIOR_PRUNING_THRESHOLD
                sense_number_1 = senses_after_pruning_1.sum()
                sense_number_2 = senses_after_pruning_2.sum()
            else:
                senses_after_pruning_1 = [True for s in xrange(sense_number)]
                senses_after_pruning_2 = senses_after_pruning_1
                sense_number_1 = sense_number
                sense_number_2 = sense_number

            if priors is not None and prior_weighting:
                word1priors = priors[word1id, :].copy()
                word2priors = priors[word2id, :].copy()
            else:
                word1priors = np.array([1.0 / sense_number_1 for s in xrange(sense_number)])
                word2priors = np.array([1.0 / sense_number_2 for s in xrange(sense_number)])

            word1priors[np.logical_not(senses_after_pruning_1)] = 0.0
            word2priors[np.logical_not(senses_after_pruning_2)] = 0.0

            weighted_sense_similarities = []
            for s1 in xrange(sense_number):
                for s2 in xrange(sense_number):
                    sense1id = word1id * sense_number + s1
                    sense2id = word2id * sense_number + s2
                    sense1emb = embeddings[sense1id]
                    sense2emb = embeddings[sense2id]
                    assert sense1emb.any() and sense2emb.any()
                    weighted_sense_similarities.append(word1priors[s1] * word2priors[s2] * scaled_cosine_similarity(sense1emb, sense2emb))
            avg_model_similarities.append(sum(weighted_sense_similarities))
            human_similarities.append(float(row[human_similarity_col_id]))
        return 100 * spearmanr(human_similarities, avg_model_similarities).correlation


def test_wordsim353(embeddings, sense_number, dictionary, wordsim353_dir, priors, prior_weighting):
    return test_context_independent(embeddings, sense_number, dictionary, os.path.join(wordsim353_dir, 'combined.csv'), priors=priors, prior_weighting=prior_weighting, delimiter=',', skip_header=True, human_similarity_col_id=2)


def test_simlex999(embeddings, sense_number, dictionary, simlex999_dir, priors, prior_weighting):
    return test_context_independent(embeddings, sense_number, dictionary, os.path.join(simlex999_dir, 'SimLex-999.txt'), priors=priors, prior_weighting=prior_weighting, delimiter='\t', skip_header=True, human_similarity_col_id=3)


def test_rw(embeddings, sense_number, dictionary, rw_dir, priors, prior_weighting):
    return test_context_independent(embeddings, sense_number, dictionary, os.path.join(rw_dir, 'rw.txt'), priors=priors, prior_weighting=prior_weighting, delimiter='\t', skip_header=False, human_similarity_col_id=2)


def test_scws_context_independed(embeddings, sense_number, dictionary, scws_dir, priors, prior_weighting):
    return test_context_independent(embeddings, sense_number, dictionary, os.path.join(scws_dir, 'ratings.txt'), priors=priors, prior_weighting=prior_weighting, delimiter='\t', skip_header=False, human_similarity_col_id=7, first_word_col_id=1, second_word_col_id=3, quoting=csv.QUOTE_NONE)


def tokenize_context(context, center_word, dictionary):
    match_obj = SCWS_CONTEXT.match(context.lower())
    left_context = match_obj.group(1).strip()
    assert match_obj.group(2).strip() == center_word
    right_context = match_obj.group(3).strip()

    tokenization_type = 'wsi'

    return parse_line(left_context, dictionary, tokenization_type), parse_line(right_context, dictionary, tokenization_type)


def calculate_context_embedding(embeddings, sense_number, sense_id, dictionary, context, window_size, two_side_window, priors, prior_weighting):
    assert sense_number * len(dictionary) == embeddings.shape[0], (sense_number, len(dictionary), embeddings.shape[0])

    word_embeddings = []

    if not window_size:
        words_to_be_taken = chain.from_iterable(context)
    else:
        words_to_be_taken = context[0][-window_size:]
        if two_side_window:
            words_to_be_taken.extend(context[1][:window_size])

    for word_id in words_to_be_taken:
        if priors is not None and sense_number > 1:
            senses_after_pruning = priors[word_id, :] >= PRIOR_PRUNING_THRESHOLD
            sense_number_after_pruning = senses_after_pruning.sum()
        else:
            senses_after_pruning = [True for s in xrange(sense_number)]
            sense_number_after_pruning = sense_number

        if priors is not None and prior_weighting:
            word_priors = priors[word_id, :].copy()
        else:
            word_priors = np.array([1.0 / sense_number_after_pruning for s in xrange(sense_number)])

        assert len(word_priors) == sense_number, '{}, {}'.format(len(word_priors), sense_number)

        word_priors[np.logical_not(senses_after_pruning)] = 0.0

        if sense_id:
            word_embeddings.append(embeddings[(word_id * sense_number + sense_id)])
        else:
            weighted_sense_embeddings = np.vstack([embeddings[(word_id * sense_number + s), :] * word_priors[s] for s in xrange(sense_number)])

            word_embeddings.append(weighted_sense_embeddings.sum(axis=0))

    stacked_word_embeddings = np.vstack(word_embeddings)

    return stacked_word_embeddings.mean(axis=0)


def estimate_probability_distribution_cosine(embeddings, sense_number, context_embeddings, context_sense_number, dictionary, word_id, context, window_size, two_side_window, priors, prior_weighting):
    """
    Estimate distribution of P(w, c, k), probability that word 'w' takes the k-th sense given context c.
    Estimation based on similarities between all senses of word_id and a given context
    """
    if context_sense_number > 0:
        context_embedding = calculate_context_embedding(embeddings=context_embeddings, sense_number=context_sense_number, sense_id=None, dictionary=dictionary,
                                                        context=context, window_size=window_size, two_side_window=two_side_window, priors=priors, prior_weighting=prior_weighting)

    context_similarities = []

    for s in xrange(sense_number):
        if context_sense_number < 0:  # in that case, we assume that context_sense_number equals sense_number and for each sense we calculate context embeddings separately
            context_embedding = calculate_context_embedding(embeddings=context_embeddings, sense_number=sense_number, sense_id=s, dictionary=dictionary,
                                                            context=context, window_size=window_size, two_side_window=two_side_window, priors=priors, prior_weighting=prior_weighting)
        sense_id = word_id * sense_number + s
        sense_emb = embeddings[sense_id]
        context_similarities.append(scaled_cosine_similarity(sense_emb, context_embedding))

    dist = context_similarities / sum(context_similarities)

    if priors is not None:
        pruned_senses = priors[word_id, :] < PRIOR_PRUNING_THRESHOLD

        dist[pruned_senses] = 0.0

    return dist


def assert_probability_distribution(dist):
    assert sum(dist) - 1 < 1e-6
    for p in dist:
        assert p >= 0 and p <= 1


def test_scws_context_depended(embeddings, sense_number, context_embeddings, context_sense_number, dictionary, estimate_sense_distribution_function, scws_dir, local_sim, window_size, two_side_window, priors, prior_weighting):
    """
    Compute either avgSimC or localSim as described in "Efficient Non-parametric Estimation of Multiple Embeddings per Word in Vector Space"
    """
    with open(os.path.join(scws_dir, 'ratings.txt'), 'r') as csvfile:
        human_similarities = []
        avg_model_similarities = []
        csv_reader = csv.reader(csvfile, delimiter='\t', strict=True, quoting=csv.QUOTE_NONE)
        for row in csv_reader:
            word1 = row[1].lower()
            word2 = row[3].lower()
            if word1 not in dictionary or word2 not in dictionary:
                # if word1 not in dictionary:
                #     logging.warning('Word \'{}\' not in the dictionary! Score will have smaller precision.'.format(word1))
                # if word2 not in dictionary:
                #     logging.warning('Word \'{}\' not in the dictionary! Score will have smaller precision.'.format(word2))
                continue
            word1id = dictionary[word1]
            word2id = dictionary[word2]

            context1 = tokenize_context(row[5], word1, dictionary)
            context2 = tokenize_context(row[6], word2, dictionary)

            if estimate_sense_distribution_function:
                assert not context_embeddings
                assert not context_sense_number
                context_probabilities = estimate_sense_distribution_function(word1id, word2id, context1, context2, full_context=(window_size is None))
                context1probabilities = context_probabilities[0, :]
                context2probabilities = context_probabilities[1, :]
            else:
                assert context_embeddings is not None
                context1probabilities = estimate_probability_distribution_cosine(embeddings, sense_number, context_embeddings, context_sense_number, dictionary, word1id, context1, window_size=window_size, two_side_window=two_side_window, priors=priors, prior_weighting=prior_weighting)
                context2probabilities = estimate_probability_distribution_cosine(embeddings, sense_number, context_embeddings, context_sense_number, dictionary, word2id, context2, window_size=window_size, two_side_window=two_side_window, priors=priors, prior_weighting=prior_weighting)

            assert_probability_distribution(context1probabilities)
            assert_probability_distribution(context2probabilities)

            if local_sim:
                sense1id_nearest_to_context = word1id * sense_number + np.argmax(context1probabilities)
                sense2id_nearest_to_context = word2id * sense_number + np.argmax(context2probabilities)
                sense1emb_nearest_to_context = embeddings[sense1id_nearest_to_context]
                sense2emb_nearest_to_context = embeddings[sense2id_nearest_to_context]
                assert sense1emb_nearest_to_context.any() and sense2emb_nearest_to_context.any()
                avg_model_similarities.append(scaled_cosine_similarity(sense1emb_nearest_to_context, sense2emb_nearest_to_context))
            else:
                sense_similarities = []
                for s1 in xrange(sense_number):
                    for s2 in xrange(sense_number):
                        sense1id = word1id * sense_number + s1
                        sense2id = word2id * sense_number + s2
                        sense1emb = embeddings[sense1id]
                        sense2emb = embeddings[sense2id]
                        sense_similarities.append(context1probabilities[s1] * context2probabilities[s2] * scaled_cosine_similarity(sense1emb, sense2emb))
                avg_model_similarities.append(sum(sense_similarities))

            human_similarities.append(float(row[7]))

        return 100 * spearmanr(human_similarities, avg_model_similarities).correlation


def dump_histogram(probabilities, work_dir, epoch_number):
    import gzip, cPickle
    hist = np.histogram(probabilities, bins=100)
    hist_file = os.path.join(work_dir, 'sense_hist_' + str(epoch_number) + '_epoch.pkl.gz')
    logging.info('Pickling to {}'.format(hist_file))
    with gzip.open(hist_file, 'wb') as f:
        cPickle.dump(hist, f)


def test_correlation_and_print_nn(embeddings, priors, dictionary, reverse_dictionary, sense_number, test_words, estimate_sense_distribution_function, window_size, two_side_window, work_dir=None, epoch_number=None):
    assert embeddings.shape[0] == len(dictionary) * sense_number

    if priors is not None:
        sense_numbers = []
        probabilities_of_all_senses_of_all_the_words = []
        for w in dictionary:
            sense_numbers.append((priors[dictionary[w], :] > PRIOR_PRUNING_THRESHOLD).sum())
            probabilities_of_all_senses_of_all_the_words.extend(priors[dictionary[w], :])
        logging.info('Avg sense number: ' + str(np.mean(sense_numbers)))

        if work_dir:
            dump_histogram(probabilities=probabilities_of_all_senses_of_all_the_words, work_dir=work_dir, epoch_number=epoch_number)

    embeddings = normalize(embeddings)

    metric_prefix = 'Spearman\'s cor. x 100 for '

    logging.info(metric_prefix + 'avgSim on WordSim-353 dataset: ' + str(round(test_wordsim353(embeddings, sense_number, dictionary, wordsim353_dir, priors=priors, prior_weighting=False), 1)))
    logging.info(metric_prefix + 'avgSimP on WordSim-353 dataset: ' + str(round(test_wordsim353(embeddings, sense_number, dictionary, wordsim353_dir, priors=priors, prior_weighting=True), 1)))
    logging.info(metric_prefix + 'avgSim on SCWS dataset: ' + str(round(test_scws_context_independed(embeddings, sense_number, dictionary, scws_dir, priors=priors, prior_weighting=False), 1)))
    logging.info(metric_prefix + 'avgSimP on SCWS dataset: ' + str(round(test_scws_context_independed(embeddings, sense_number, dictionary, scws_dir, priors=priors, prior_weighting=True), 1)))

    if sense_number > 1:
        if estimate_sense_distribution_function:
            logging.info(metric_prefix + 'avgSimC on SCWS dataset using variant 1 of probability estimation: ' + str(round(test_scws_context_depended(embeddings, sense_number, None, None, dictionary, estimate_sense_distribution_function, scws_dir, local_sim=False, window_size=window_size, two_side_window=None, priors=priors, prior_weighting=False), 1)))
            logging.info(metric_prefix + 'avgSimC on SCWS dataset using variant 3 of probability estimation: ' + str(round(test_scws_context_depended(embeddings, sense_number, None, None, dictionary, estimate_sense_distribution_function, scws_dir, local_sim=False, window_size=None, two_side_window=None, priors=priors, prior_weighting=False), 1)))
        logging.info(metric_prefix + 'avgSimC on SCWS dataset using variant 5 of probability estimation: ' + str(round(test_scws_context_depended(embeddings, sense_number, embeddings, sense_number, dictionary, None, scws_dir, local_sim=False, window_size=None, two_side_window=None, priors=priors, prior_weighting=False), 1)))
        logging.info(metric_prefix + 'avgSimC on SCWS dataset using variant 7 of probability estimation: ' + str(round(test_scws_context_depended(embeddings, sense_number, embeddings, sense_number, dictionary, None, scws_dir, local_sim=False, window_size=None, two_side_window=None, priors=priors, prior_weighting=True), 1)))
        logging.info(metric_prefix + 'avgSimC on SCWS dataset using variant 8 of probability estimation: ' + str(round(test_scws_context_depended(embeddings, sense_number, embeddings, -1, dictionary, None, scws_dir, local_sim=False, window_size=None, two_side_window=None, priors=priors, prior_weighting=False), 1)))

        if estimate_sense_distribution_function:
            logging.info(metric_prefix + 'localSim on SCWS dataset using variant 1 of probability estimation: ' + str(round(test_scws_context_depended(embeddings, sense_number, None, None, dictionary, estimate_sense_distribution_function, scws_dir, local_sim=True, window_size=window_size, two_side_window=None, priors=priors, prior_weighting=False), 1)))
            logging.info(metric_prefix + 'localSim on SCWS dataset using variant 3 of probability estimation: ' + str(round(test_scws_context_depended(embeddings, sense_number, None, None, dictionary, estimate_sense_distribution_function, scws_dir, local_sim=True, window_size=None, two_side_window=None, priors=priors, prior_weighting=False), 1)))
        logging.info(metric_prefix + 'localSim on SCWS dataset using variant 5 of probability estimation: ' + str(round(test_scws_context_depended(embeddings, sense_number, embeddings, sense_number, dictionary, None, scws_dir, local_sim=True, window_size=None, two_side_window=None, priors=priors, prior_weighting=False), 1)))
        logging.info(metric_prefix + 'localSim on SCWS dataset using variant 7 of probability estimation: ' + str(round(test_scws_context_depended(embeddings, sense_number, embeddings, sense_number, dictionary, None, scws_dir, local_sim=True, window_size=None, two_side_window=None, priors=priors, prior_weighting=True), 1)))
        logging.info(metric_prefix + 'localSim on SCWS dataset using variant 8 of probability estimation: ' + str(round(test_scws_context_depended(embeddings, sense_number, embeddings, -1, dictionary, None, scws_dir, local_sim=True, window_size=None, two_side_window=None, priors=priors, prior_weighting=False), 1)))

    logging.info(metric_prefix + 'avgSim on SimLex-999 dataset: ' + str(round(test_simlex999(embeddings, sense_number, dictionary, simlex999_dir, priors=priors, prior_weighting=False), 1)))
    logging.info(metric_prefix + 'avgSimP on SimLex-999 dataset: ' + str(round(test_simlex999(embeddings, sense_number, dictionary, simlex999_dir, priors=priors, prior_weighting=True), 1)))
    # logging.info(metric_prefix + 'avgSim on Rare Word dataset: ' + str(round(test_rw(embeddings, sense_number, dictionary, rw_dir, priors=priors), 1)))

    logging.info('Probability estimation variants:')
    logging.info(' 1 - lower network weights for surrounding context words (window size as during training)')
    logging.info(' 3 - lower network weights for all context words')
    logging.info(' 5 - cosine similarity to context representation,')
    logging.info('     where the context representation is an average of all senses embeddings for all context words')
    logging.info(' 7 - as 5 but context representation is an average of all senses embeddings weighted with priors')
    logging.info(' 8 - cosine similarity to context representation,')
    logging.info('     where the context representation is an average of senses embeddings for the same sense ID as the currently processed sense of the central word')

    if priors is not None:
        for w in test_words:
            logging.info('Priors for word \'{:6}\': {}'.format(w, priors[dictionary[w], :]))

    for w in test_words:
        print_nearest_neighbors(embeddings, sense_number, dictionary, reverse_dictionary, w, priors)


def normalize(vectors):
    logging.info('Normalizing')

    norm = np.sqrt(np.sum(np.square(vectors), axis=1, keepdims=True))

    # if not norm.all():
    #     print 'norms:'
    #     print norm == 0.0
    #     print 'embed:'
    #     print embeddings == 0.0
    #     exit()

    assert norm.all()
    # norm[norm == 0.0] = 1

    return vectors / norm


def reduce_dims(embeddings, work_dir, epoch_number):
        import gzip
        import cPickle
        from sklearn import decomposition

        pca = decomposition.PCA(n_components=2)
        logging.info('Fitting PCA')
        pca.fit(embeddings)
        low_dim_embs = pca.transform(embeddings)

        output_file = os.path.join(work_dir, 'reduced_' + str(epoch_number) + '_epoch.gzip')
        logging.info('Dumping to ' + output_file)

        with gzip.open(output_file, 'wb') as f:
            cPickle.dump(low_dim_embs, f)


def read_raw_wwsi(wsi_dir):
    raw_wwsi = {}
    with open(os.path.join(wsi_dir, 'dataset-14'), 'r') as f:
        current_word = None
        some_big_number = 0
        remaining_ctx_number = 0
        for l in f:
            splitted = l.split()
            if len(splitted) == 5 and splitted[1] == '(disambiguation)':
                if current_word:
                    assert not some_big_number
                    raw_wwsi[current_word] = current_contexts
                current_word = splitted[2]
                current_page = splitted[0]
                some_big_number = int(splitted[3])
                ambiguous_page_number = int(splitted[4])
                current_contexts = []
            elif len(splitted) >= 3 and len(splitted) <= 6 and splitted[-2].isdigit() and splitted[-1].isdigit():
                assert not remaining_ctx_number
                some_big_number -= int(splitted[-2])
                remaining_ctx_number = int(splitted[-1])
            elif len(splitted) == 1:
                int(splitted[0])  # WTF is this???
                pass
            else:
                current_contexts.append(l)
                remaining_ctx_number -= 1
        assert not some_big_number
        raw_wwsi[current_word] = current_contexts
    assert len(raw_wwsi) == 188
    return raw_wwsi


def test_ari(wsi_dir, testset_name, dictionary, estimate_sense_distribution_fct):
    wsi_ctx_file = os.path.join(wsi_dir, 'dataset.txt')
    wsi_gt_file = os.path.join(wsi_dir, 'key.txt')

    # logging.info(' Loading ground truth')

    ground_truth = {}

    with open(wsi_gt_file) as f:
        for l in f:
            splitted = l.split()
            word_with_pos = splitted[0]
            if word_with_pos not in ground_truth:
                ground_truth[word_with_pos] = {}
            word_with_pos_with_ctx_id = splitted[1].split('.')
            assert word_with_pos == '.'.join([word_with_pos_with_ctx_id[0], word_with_pos_with_ctx_id[1]])
            ctx_id = int(word_with_pos_with_ctx_id[2])
            word_with_pos_with_sense_id = splitted[2].split('.')
            assert word_with_pos.split('.')[0] == word_with_pos_with_sense_id[0]  # There is no PoS in WWSI
            if '&&' in word_with_pos_with_sense_id[2]:  # TODO: handle it better; this occurs in SemEval 2010
                continue
            sense_id = int(word_with_pos_with_sense_id[2])
            ground_truth[word_with_pos][ctx_id] = sense_id

    if __debug__:
        sense_numbers = []

        # logging.info('Total ctx number: {}. Unique word with POS number: {}'.format(sum([len(ctx_dict) for ctx_dict in ground_truth.itervalues()]), len(ground_truth)))

        for w in ground_truth.itervalues():
            sense_numbers.append(len(set(w.itervalues())))

        # logging.info('Sense numbers. Min: {}, max: {}, median: {} and mean: {}.'.format(min(sense_numbers), max(sense_numbers), np.median(sense_numbers), np.mean(sense_numbers)))

    # logging.info('Predicting')

    ctx_number = 0
    ctx_expected_number = 0
    current_word_with_pos = None
    aris = []

    # unknown_word_id = dictionary[UNKNOWN_WORD]
    # reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

    disambiguation_examples = {}
    disambiguation_examples_words = ()  # ('mouse', 'goal', 'heart', 'wave', 'root', 'coup', 'java', 'ruby', 'time', 'train', 'mass', 'bus', 'foot', 'mini', 'lion', 'mirror', 'burn', 'monopoly', 'apache')

#    if testset_name == 'WWSI':
#        wwsi_raw = read_raw_wwsi(wsi_dir)

    with open(wsi_ctx_file, 'r') as f:
        for l in f:
            splitted = l.split()
            if len(splitted) == 2 and '\t' not in l:
                assert ctx_number == ctx_expected_number

                if current_word_with_pos:
                    aris.append(adjusted_rand_score(predicted_senses, expected_senses))

                predicted_senses = []
                expected_senses = []

                current_word_with_pos = splitted[0]
                current_word = current_word_with_pos.split('.')[0]
                if current_word not in dictionary:
                    current_word_with_pos = None
                    continue
                ctx_expected_number = int(splitted[1])
                ctx_number = 0
            elif current_word_with_pos:
                ctx_number += 1

                sense_dict = ground_truth[current_word_with_pos]
                if ctx_number not in sense_dict:
                    continue  # due to skipping '&&' in SemEval2010

                tab_splitted = l.split('\t')
                assert len(tab_splitted) == 2

                ctx = (parse_line(tab_splitted[0], dictionary, 'wsi'), parse_line(tab_splitted[1], dictionary, 'wsi'))

                if ctx != ([], []):  # this is possible in case of SemEval 2013
                    expected_senses.append(sense_dict[ctx_number])

                    sense_distribution = estimate_sense_distribution_fct(w=dictionary[current_word], c=ctx)

                    # logging.info('word: {}, ctx size: {}, first word: {}, unknown number: {}, expected sense: {}, predicted sense: {}'.format(current_word, len(ctx[0]) + len(ctx[1]), reverse_dictionary[list(chain.from_iterable(ctx))[0]], len([t for t in chain.from_iterable(ctx) if t == unknown_word_id]), expected_senses[-1], predicted_senses[-1]))
                    predicted_senses.append(np.argmax(sense_distribution) + 1)

                    if testset_name == 'WWSI' and current_word in disambiguation_examples_words:
                        if current_word not in disambiguation_examples:
                            disambiguation_examples[current_word] = []
                        disambiguation_examples[current_word].append((
                            wwsi_raw[current_word][ctx_number - 1],
                            expected_senses[-1],
                            sense_distribution))

    logging.info('  Averaged ARI for {:12}: {:.3}'.format(testset_name, np.mean(aris)))

#    logging.info('Disambiguation_examples:')

    for w in disambiguation_examples:
        logging.info('  Word {}:'.format(w))
        for sense_id in xrange(len(disambiguation_examples[w][0][2])):
            logging.info('    Sense {}:'.format(sense_id))
            top_examples = sorted(disambiguation_examples[w], key=lambda ex: ex[2][sense_id], reverse=True)[:4]
            for ex in top_examples:
                logging.info('      {:.3} {} {}'.format(ex[2][sense_id], ex[1], ex[0]))


def test_ari_all_datasets(wsi_dir, dictionary, estimate_sense_distribution_fct):
    test_ari(wsi_dir=os.path.join(wsi_dir, 'semeval-2007'), testset_name='SemEval 2007', dictionary=dictionary, estimate_sense_distribution_fct=estimate_sense_distribution_fct)
    test_ari(wsi_dir=os.path.join(wsi_dir, 'semeval-2010'), testset_name='SemEval 2010', dictionary=dictionary, estimate_sense_distribution_fct=estimate_sense_distribution_fct)
    test_ari(wsi_dir=os.path.join(wsi_dir, 'semeval-2013'), testset_name='SemEval 2013', dictionary=dictionary, estimate_sense_distribution_fct=estimate_sense_distribution_fct)
    test_ari(wsi_dir=os.path.join(wsi_dir, 'wwsi'), testset_name='WWSI', dictionary=dictionary, estimate_sense_distribution_fct=estimate_sense_distribution_fct)


def test_wsi(wsi_dir, dictionary, estimate_sense_distribution_model_fct, estimate_sense_distribution_cosine_fct):
    logging.info('Probability estimated as cosine similarity to context representation,')
    logging.info('where the context representation is an average of all senses embeddings for all context words')

    f = lambda w, c: estimate_sense_distribution_cosine_fct(w=w, c=c, variable_context_size=True, prior_weighting=False, ctx_based_on_out_emb=False)
    test_ari_all_datasets(wsi_dir=wsi_dir, dictionary=dictionary, estimate_sense_distribution_fct=f)
