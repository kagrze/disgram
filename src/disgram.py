"""
In case of multisense embeddings the dictionary is standard (single item per word)
Embedding matrix has two dimensions: (dict_size x sense_number) x embed_size
Vectors for multiple senses of a single word are placed one-by-one in the embeddings matrix.
To access given sense of given word you need to compute: word_id * sense_number + sense_id

Usage:
  disgram.py [options]

Options:
  --work_dir=VAL            Dump codes to work_dir
  --trainset_file=VAL       Train set file [default: None]
  --word_dim_num=VAL        Word embedding size [default: 300]
  --dict_file=VAL           Pickled dictionary file [default: None]
  --ta=VAL                  Training algorithm. CBOW or SG [default: SG]
  --lr_train=VAL            Learning rate [default: 0.5]
  --batch_size=VAL          Mini-batch size [default: 128]
  --train_epoch_num=VAL     Epoch number [default: 1]
  --loss_function=VAL       Loss function NCE or SS [default: SS]
  --num_sampled=VAL         The number of classes to randomly sample per batch when computing NCE or SS loss [default: 64]
  --optimizer=VAL           Optimizer: GD, AG, AD, AM, FTRL, MO or RMS [default: MO]
  --window=VAL              Window - how many words to each side should be taken into account (only applies to CBOW) [default: 5]
  --two_side_window=VAL     Two side window [default: True]
  --center-word-in-win=VAL  Include center word inside window [default: False]
  --context-rep=VAL         Context representation:
                            0 - all input word vectors are summed or averaged
                            1 - all input word vectors are concatenated
                            2 - context word vectors are summed and concatenated with center word [default: 0]
  --senses-concat=VAL       Should weighted embeddings of center word senses be concatenated instead fo summed? (apply only to PMSSG) [default: False]
  --cbow_mean=VAL           If the input vectors should not be concatenated then should they be averaged or just summed?
                            (only applies to DM when concatenation if False) [default: False]
  --momentum=VAL            The momentum hyperparameter [default: 0.5,0.6,0.7,0.8,0.9]
  --dropout_prob=VAL        Dropout probability [default: 0.0]
  --sense_number=VAL        Number of senses per word [default: 1]
  --prob_sense_assign=VAL   Probabilistic sense assignment:
                            0 - disabled,
                            1 - similarity based,
                            2 - softmax based
                            3 - softmax based with separate distribution for each central word [default: 0]
  --lower_hidden_units=VAL  Number of hidden units in the 'lower' network. If zero then there is no hidden layer in the 'lower' network [default: 0]
  --batch_norm=VAL          Batch normalization:
                            0 - disabled
                            1 - in the hidden layer of the lower network
                            2 - before the softmax layer of the lower network
                            3 - before the softmax layer of the lower network and before the softmax layer of the upper network [default: 0]
  --lower_penalty=VAL       L1 or L2 weight penalty for softmax weights in the lower network. No penalty by default. [default: 0]
  --lower_penalty_val=VAL   Value for L1 or L2 weight penalty for softmax weights in the lower network. [default: 0.0]
  --uncertain_penalty=VAL   Penalize lower softmax distributions with high entropy (balanced distributions) [default: 0.0]
  --parallel_penalty=VAL    Penalize parallel sense vectors [default: 0.0]
  --huber_loss=VAL          Use Huber loss instead of squared loss to compute parallel penalty [default: 0]
  --ctx-embedding-src=VAL   Context embedding sources:
                            0 - sense embeddings,
                            1 - global vectors,
                            2 - context words prediction matrix (output embeddings) [default: 2]
  --dual-learning=VAL       In case of training multiple senses without global vectors propagate gradients to senses through context model as well as center word model [default: False]
  --write_summary=VAL       Write TensorBoard summaries [default: False]
  --worker_number=VAL       Worker number [default: 48]
  --token_type=VAL          Tokenization type: standard, imdb or bookcorpus [default: basic]
  --test_words=VAL          Comma-separated list of test words [default: ]
  --save_each_epoch=VAL     Save the model after each epoch separately [default: True]
  --pad_every_sentence=VAL  Pad every single sentence to allow prediction of marginal words. [default: True]
  --mode=VAL                Mode of operation
                            0 - train
                            1 - estimate priors
                            2 - test
                            3 - WSI
                            4 - Parameter dump
                            5 - Reduce dims
                            6 - Prepare representation
                            7 - WSI with sense disambiguation vectors [default: 0]
  --ctx_softmax_temp=VAL    Context softmax temperature [default: 1.0]
  --learn_ctx_smx_temp=VAL  If not zero then learn context softmax temperature. Use this value as initial value. [default: 0.0]
  --wsi_dir=VAL             WSI testsets directory [default: None]
  --init_model_dir=VAL      A directory from where initialization data are taken. If none then use standard initialization. [default: None]
  --relaxed_one_hot=VAL     Use sampling from relaxed one-hot categorical distribution instead of classic softmax for sense prediction [default: 0]
  --subwords_range=VAL      In skip-gram use subwords of the center word in a given range to predict context words [default: 0-0]
  --subword_bucket_num=VAL  Subword bucket number [default: 2000000]
  --separate_subwords=VAL   Learn separate subword embeddings for each sense [default: 0]
  --max_subword_number=VAL  Maximal number of subwords for a single word. If 0 then no limit. [default: 20]
  --check_num_batch=VAL     Starting from this mini-batch check all tensors for NaN and Inf values. If zero then disabled. [default: 0]
  --doc_corpus_dir=VAL      Directory where the document corpus is stored [default: None]
  --doc_represent_dir=VAL   Directory where the document representaiton will be extracted [default: None]
  --doc_rep_disamb=VAL      Should disambiguation be used for the doc representation [default: 0]
  --doc_rep_disamb_out=VAL  Should output embeddings be used for building vector represntation of the context for disambugiation used for the doc representation [default: 0]
"""

import sys
import logging
import collections
import os
import random
import cPickle
import gzip
import threading
import numpy as np
import tensorflow as tf
from fnvhash import fnv1a_32
from utils import UNKNOWN_WORD, ThreadSafeIterator, build_and_start_thread_pool, generate_context, generate_cbow_batches, file_len, parse_bool, get_loss_function, get_optimizer, make_dir
from utils import parse_line as parse_line_base
from test import PRIOR_PRUNING_THRESHOLD, test_correlation_and_print_nn, estimate_probability_distribution_cosine, reduce_dims, test_wsi


assert sys.version_info.major == 2

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', datefmt='%Y-%m-%d %H:%M:%S', stream=sys.stdout, level=logging.INFO)

MODEL_FILE = 'model.ckpt'
EPSILON = 0.001
EPSILON_BIS = 0.00001
EMBEDDINGS = 'embeddings'
MAIN_SOFTMAX_W = 'main_softmax_w'
MAIN_SOFTMAX_B = 'main_softmax_b'
CTX_SOFTMAX_W = 'ctx_softmax_w'
CTX_SOFTMAX_B = 'ctx_softmax_b'
ARCH_EXT = '.pkl'
APPROXIMATELY_WIKI_BATCH_NUMBER = 34e6


class Word2VecBase(object):
    def define_common_variables(self, vocabulary_size, embedding_size, window, two_side_window, sense_number, context_embedding_source, tokenization_type, model_dtype, optimizer_class, init_model_dir):
        self._left_window_size = window
        self._right_window_size = window if two_side_window else 0
        self._window_size = self._left_window_size + self._right_window_size

        self._input_labels = tf.placeholder(dtype=tf.int32, shape=[None, 1], name='input_labels')

        self._sense_number = sense_number

        self._embedding_size = embedding_size

        self._init_model_dir = init_model_dir

        self._model_dtype = model_dtype

        embeddings_shape = [vocabulary_size * self._sense_number, self._embedding_size]

        if os.path.isdir(self._init_model_dir):
            self._word_embeddings_initializer = tf.placeholder(shape=embeddings_shape, dtype=self._model_dtype)
        else:
            init_width = 0.5 / self._embedding_size / self._sense_number
            self._word_embeddings_initializer = tf.random_uniform(shape=embeddings_shape, minval=-init_width, maxval=init_width, dtype=self._model_dtype)

        self._word_embeddings = tf.Variable(initial_value=self._word_embeddings_initializer, name='train_embeddings')

        self._context_embedding_source = context_embedding_source

        self._tokenization_type = tokenization_type

        w_shape = [vocabulary_size, self._hidden_layer_size]
        b_shape = [vocabulary_size]

        if os.path.isdir(self._init_model_dir):
            w_initializer = self.load_init_data(data_name=MAIN_SOFTMAX_W, expected_shape=w_shape)
            b_initializer = self.load_init_data(data_name=MAIN_SOFTMAX_B, expected_shape=b_shape)
        else:
            w_initializer = tf.zeros(shape=w_shape, dtype=self._model_dtype)
            b_initializer = tf.zeros(shape=b_shape, dtype=self._model_dtype)

        self._weights_t = tf.Variable(initial_value=w_initializer, name='weights')
        self._biases = tf.Variable(initial_value=b_initializer, name='biases')

        self._saver_variables = [self._word_embeddings, self._weights_t, self._biases]  # do we need to store biases?

        if self._context_embedding_source == 1:
            assert self._sense_number > 1
            self._global_vectors = tf.Variable(initial_value=tf.random_uniform(shape=[vocabulary_size, self._embedding_size], minval=-init_width, maxval=init_width, dtype=self._model_dtype), name='global_vectors')
            self._saver_variables.append(self._global_vectors)

        self._optimizer_class = optimizer_class

    def dump_parameters(self, parameters, work_dir, data_name):
        dump_dir = os.path.join(work_dir, 'parameters_dump')

        if not os.path.isdir(dump_dir):
            os.mkdir(dump_dir)

        file_path = os.path.join(dump_dir, data_name + ARCH_EXT)
        logging.info('Dumping {}'.format(file_path))

        with open(file_path, 'wb') as f:
            cPickle.dump(parameters, f)

        logging.info('GZIPing ' + file_path)
        os.system('gzip -f ' + file_path)

    def load_init_data(self, data_name, expected_shape):
        file_path = os.path.join(self._init_model_dir, data_name + ARCH_EXT + '.gz')
        logging.info('Loading initialization data from {}'.format(file_path))
        with gzip.open(file_path, 'rb') as f:
            init_data = cPickle.load(f)
        assert init_data.shape == tuple(expected_shape)
        return init_data

    def define_summaries(self):
        self._train_loss_summary = tf.summary.scalar(name="train_loss", tensor=self._train_loss)
        self._train_histograms = tf.summary.merge(inputs=self.define_histograms())

    def define_histograms(self):
        histograms = []
        assert self._train_grads_and_vars[0][1] == self._word_embeddings
        assert self._train_grads_and_vars[1][1] == self._weights_t
        histograms.append(tf.summary.histogram(name="word_embeddings", values=self._train_grads_and_vars[0][1]))
        histograms.append(tf.summary.histogram(name="weights", values=self._train_grads_and_vars[1][1]))
        histograms.append(tf.summary.histogram(name="word_embedding_gradients", values=self._train_grads_and_vars[0][0]))
        histograms.append(tf.summary.histogram(name="weights_gradients", values=self._train_grads_and_vars[1][0]))
        return histograms

    def define_second_layer(self, dropout_prob, num_sampled, vocabulary_size, loss_function, shift_threshold=False, use_locking=False):
        assert self._train_embed.get_shape().as_list()[1] == self._hidden_layer_size

        train_dropped = tf.nn.dropout(x=self._train_embed, keep_prob=1. - dropout_prob)

        self._train_loss = tf.reduce_mean(input_tensor=loss_function(weights=self._weights_t, biases=self._biases, inputs=train_dropped,
                                                                     labels=self._input_labels, num_sampled=num_sampled, num_classes=vocabulary_size))

        regularization_losses = tf.get_collection(key=tf.GraphKeys.REGULARIZATION_LOSSES)

        expected_number_of_regularization_losses = 0

        if self._context_softmax_weights_penalty:
            expected_number_of_regularization_losses += 1

        if self._uncertain_penalty:
            expected_number_of_regularization_losses += 1

        if self._parallel_penalty:
            expected_number_of_regularization_losses += 1

        assert len(regularization_losses) == expected_number_of_regularization_losses

        if regularization_losses:
            self._train_loss += tf.add_n(inputs=regularization_losses)

        self._lr_ph = tf.placeholder(dtype=tf.float32, shape=[], name='LR')
        # below we define two separate optimizers; one is used at the training time and the second is used during the inference phase
        if self._optimizer_class == tf.train.MomentumOptimizer or self._optimizer_class == tf.train.RMSPropOptimizer:
            self._momentum = tf.placeholder(dtype=tf.float32, shape=[], name='momentum')
            train_optimizer = self._optimizer_class(learning_rate=self._lr_ph, momentum=self._momentum, use_locking=use_locking)
        else:
            train_optimizer = self._optimizer_class(learning_rate=self._lr_ph, use_locking=use_locking)

        self._train_grads_and_vars = train_optimizer.compute_gradients(self._train_loss, var_list=self.get_train_var_list())

        self._apply_train_gradients_op = train_optimizer.apply_gradients(self._train_grads_and_vars)

        logging.info('Defining saver for variables: ' + ', '.join([v.name for v in self._saver_variables]))

        # self._saver = tf.train.Saver(var_list=self._saver_variables)
        self._saver = tf.train.Saver(var_list=None)  # TODO: in order to train already trained model we need to save optimizer variables to which we do not have explicit access

    def get_train_var_list(self):
        var_list = [self._word_embeddings, self._weights_t, self._biases]

        if self._context_embedding_source == 1:
            var_list.append(self._global_vectors)

        return var_list

    def init(self, session):
        logging.info("Initializing variables")
        if os.path.isdir(self._init_model_dir):
            embeddings_shape = self._word_embeddings.get_shape().as_list()
            feed_dict = {self._word_embeddings_initializer: self.load_init_data(data_name=EMBEDDINGS, expected_shape=embeddings_shape)}
        else:
            feed_dict = None
        session.run(self._init_op, feed_dict=feed_dict)

    def visualize_embeddings(self, logdir, reverse_dictionary, summary_writer):
        from tensorflow.contrib.tensorboard.plugins import projector
        config = projector.ProjectorConfig()

        embedding = config.embeddings.add()
        embedding.tensor_name = self._word_embeddings.name
        embedding.metadata_path = os.path.join(logdir, 'vocabulary.tsv')
        with open(embedding.metadata_path, 'w') as f:
            for word_id in xrange(len(reverse_dictionary)):
                for s in xrange(self._sense_number):
                    f.write(reverse_dictionary[word_id] + '_' + str(s + 1) + '\n')

        output_embedding = config.embeddings.add()
        output_embedding.tensor_name = self._weights_t.name
        output_embedding.metadata_path = os.path.join(logdir, 'vocabulary_output.tsv')
        with open(output_embedding.metadata_path, 'w') as f:
            for word_id in xrange(len(reverse_dictionary)):
                f.write(reverse_dictionary[word_id] + '\n')

        projector.visualize_embeddings(summary_writer, config)

    def save_epoch(self, session, work_dir, epoch_number):
        save_path = os.path.join(work_dir, 'model_' + str(epoch_number) + '_epoch')
        make_dir(save_path)
        self.save(session=session, save_path=save_path)

    def restore_epoch(self, session, work_dir, epoch_number):
        save_path = os.path.join(work_dir, 'model_' + str(epoch_number) + '_epoch')
        self.restore(session=session, save_path=save_path)

    def train(self, session, dictionary, reverse_dictionary, trainset_file, train_epoch_num, batch_size, lr_train, momentum,
              write_summary, work_dir, worker_number, save_each_epoch, pad_every_sentence, ctx_softmax_temp, word_subwords):
        if write_summary:
            logdir = os.path.join(work_dir, 'logs')
            make_dir(logdir)
            summary_writer = tf.summary.FileWriter(logdir=logdir, graph=session.graph)
            self.visualize_embeddings(logdir, reverse_dictionary, summary_writer)

        logging.info('Begin training')

        from datetime import datetime
        start_time = datetime.now()

        self._max_batch_id = 0

        def _train_thread(epoch, batch_iterator):
            for batch in batch_iterator:
                batch_id = batch[0]
                if len(lr_train) == 1:
                    if train_epoch_num == 1:
                        lr_decay_factor = float(batch_id) / APPROXIMATELY_WIKI_BATCH_NUMBER
                    else:
                        lr_decay_factor = 1.0 / train_epoch_num * epoch
                    lr_to_pass = lr_train[0] - float(lr_train[0]) * lr_decay_factor
                else:
                    lr_to_pass = lr_train[epoch]

                if len(ctx_softmax_temp) == 1:
                    ctx_softmax_temp_to_pass = ctx_softmax_temp[0]
                else:
                    ctx_softmax_temp_to_pass = ctx_softmax_temp[epoch]

                feed_dict = self.get_feed_dict(batch=batch, lr_to_pass=lr_to_pass, ctx_softmax_temp=ctx_softmax_temp_to_pass, is_training=True, word_subwords=word_subwords)

                ops_to_run = [self._apply_train_gradients_op]

                if not self._probabilistic_sense_assignment:
                    ops_to_run.append(self._updated_context_cluster_centers)

                if self._batch_norm:
                    expected_number_of_moving_stats = 4 if self._batch_norm == 3 else 2
                    assert len(self._updated_moving_mean_and_moving_variance) == expected_number_of_moving_stats
                    ops_to_run.extend(self._updated_moving_mean_and_moving_variance)

                if self._optimizer_class == tf.train.MomentumOptimizer or self._optimizer_class == tf.train.RMSPropOptimizer:
                    feed_dict[self._momentum] = momentum[epoch]

                if self._check_num_batch and batch_id > self._check_num_batch:
                    feed_dict[self._word_embeddings_initializer] = np.zeros(self._word_embeddings_initializer.get_shape().as_list())  # because self._check_numerics_op checks ALL tensors
                    ops_to_run.append(self._check_numerics_op)

                session.run(ops_to_run, feed_dict=feed_dict)

                if write_summary:
                    summary_writer.add_summary(self._train_loss_summary.eval(feed_dict=feed_dict, session=session), (self._max_batch_id + 1) * epoch + batch_id)

                if batch_id % 1000 == 0:
                    info_str = 'Thread: ' + threading.current_thread().name
                    info_str += ', epoch: ' + str(epoch)
                    info_str += ', batch: ' + str(batch_id)
                    info_str += ', time: {:.1f}'.format((datetime.now() - start_time).total_seconds() * APPROXIMATELY_WIKI_BATCH_NUMBER / (batch_id + 1) / 3600)
                    info_str += ', lr: {:.4f}'.format(lr_to_pass)
                    if self._learn_ctx_softmax_temp:
                        temp = self._ctx_softmax_temp.eval(session=session)[0]
                    else:
                        temp = feed_dict[self._ctx_softmax_temp]
                    info_str += ', temp: {:.4f}'.format(temp)
                    # if self._optimizer_class == tf.train.MomentumOptimizer or self._optimizer_class == tf.train.RMSPropOptimizer:
                    #     info_str += ', momentum: ' + str(feed_dict[self._momentum])
                    info_str += ', loss: {:.4f}'.format(self._train_loss.eval(feed_dict=feed_dict, session=session))  # can we log here loss summary instead of loss?
                    logging.info(info_str)
                    if write_summary:
                        summary_writer.add_summary(self._train_histograms.eval(feed_dict=feed_dict, session=session), (self._max_batch_id + 1) * epoch + batch_id)
                if batch_id > self._max_batch_id:
                    self._max_batch_id = batch_id

        epoch_to_train = [train_epoch_num - 1] if save_each_epoch else xrange(train_epoch_num)

        for epoch in epoch_to_train:
            if self._sense_number == 1:
                batch_iterator = self.generate_batches(batch_size, dictionary, trainset_file, pad_every_sentence)
            else:
                batch_iterator = self.generate_batches_multiple_senses(batch_size, dictionary, trainset_file, pad_every_sentence)
            kwargs = {'epoch': epoch, 'batch_iterator': ThreadSafeIterator(batch_iterator)}
            build_and_start_thread_pool(_train_thread, kwargs, worker_number)

        self.save_epoch(session, work_dir, train_epoch_num)

        logging.info('End training. Total time: {}.'.format(datetime.now() - start_time))

    def save(self, session, save_path):
        pkl_file = os.path.join(save_path, MODEL_FILE)
        logging.info('Saving model to ' + pkl_file)
        self._saver.save(session, pkl_file)

    def restore(self, session, save_path):
        pkl_file = os.path.join(save_path, MODEL_FILE)
        logging.info('Loading model from ' + pkl_file)
        self._saver.restore(session, pkl_file)

    def get_word_embeddings(self, session):
        return self._word_embeddings.eval(session=session)

    def get_output_embeddings(self, session):
        return self._weights_t.eval(session=session)


Word2VecSGConfig = collections.namedtuple('Word2VecSGConfig', ['vocabulary_size', 'word_embedding_size', 'num_sampled', 'loss_function', 'optimizer_class',
                                          'dropout_prob', 'write_summary', 'window', 'two_side_window', 'center_word_in_window',
                                          'sense_number', 'probabilistic_sense_assignment', 'context_embedding_source', 'dual_learning',
                                          'context_rep', 'senses_concat', 'tokenization_type', 'model_dtype', 'lower_hidden_units', 'batch_norm',
                                          'context_softmax_weights_penalty', 'context_softmax_weights_penalty_val', 'uncertain_penalty', 'parallel_penalty',
                                          'huber_loss', 'init_model_dir', 'relaxed_one_hot', 'subword_bucket_number', 'max_subword_number', 'separate_subwords',
                                          'learn_ctx_softmax_temp', 'check_num_batch'])


def entropy(t, reduction_index):
    # add epsilon to prevent log(0)
    return tf.negative(x=tf.reduce_sum(input_tensor=tf.multiply(x=t, y=tf.log(x=t + EPSILON_BIS)), axis=[reduction_index]))


class Word2VecSG(Word2VecBase):
    def __init__(self, config):
        self._probabilistic_sense_assignment = config.probabilistic_sense_assignment
        assert config.probabilistic_sense_assignment == 0 or config.sense_number > 1
        self._hidden_layer_size = config.word_embedding_size if (self._probabilistic_sense_assignment == 0 or not config.senses_concat) else config.word_embedding_size * config.sense_number
        self.define_common_variables(vocabulary_size=config.vocabulary_size, embedding_size=config.word_embedding_size, window=config.window, two_side_window=config.two_side_window,
                                     sense_number=config.sense_number, context_embedding_source=config.context_embedding_source, tokenization_type=config.tokenization_type,
                                     model_dtype=config.model_dtype, optimizer_class=config.optimizer_class, init_model_dir=config.init_model_dir)
        self._context_rep = config.context_rep
        if not self._probabilistic_sense_assignment:
            self._context_cluster_centers = tf.Variable(initial_value=tf.zeros(shape=[config.vocabulary_size * config.sense_number, config.word_embedding_size], dtype=self._model_dtype), name='context_cluster_centers')
            # self._saver_variables.append(self._context_cluster_centers)  # do we need to store context_cluster_centers ? I do not think so.
        self._lower_hidden_units = config.lower_hidden_units
        self._batch_norm = config.batch_norm
        self._context_softmax_weights_penalty = config.context_softmax_weights_penalty
        self._context_softmax_weights_penalty_val = config.context_softmax_weights_penalty_val
        self._uncertain_penalty = config.uncertain_penalty
        self._parallel_penalty = config.parallel_penalty
        self._huber_loss = config.huber_loss
        self._relaxed_one_hot = config.relaxed_one_hot
        self._max_subword_number = config.max_subword_number
        self._subword_bucket_number = config.subword_bucket_number
        self._check_num_batch = config.check_num_batch
        if self._check_num_batch:
            self._check_numerics_op = tf.add_check_numerics_ops()
        self._separate_subwords = config.separate_subwords
        if self._max_subword_number:
            init_width = 0.5 / self._embedding_size
            if self._separate_subwords:
                # do we need those 'zero' embeddings replicated as many times as number of senses? or just one would be enough? for now we assume just one
                embeddings_shape = [self._subword_bucket_number * self._sense_number, self._embedding_size]
                init_width /= self._sense_number
            else:
                embeddings_shape = [self._subword_bucket_number, self._embedding_size]
            zero_init_val = tf.zeros(shape=[1, self._embedding_size], dtype=self._model_dtype)  # since we use zeroing mask maybe we do not to initialize first row with zeros?
            rand_init_val = tf.random_uniform(shape=embeddings_shape, minval=-init_width, maxval=init_width, dtype=self._model_dtype)
            self._subword_embeddings = tf.Variable(initial_value=tf.concat(values=[zero_init_val, rand_init_val], axis=0), name='subword_embeddings')
            self._saver_variables.append(self._subword_embeddings)
        self._learn_ctx_softmax_temp = config.learn_ctx_softmax_temp
        self.define_first_layer_activations(dual_learning=config.dual_learning, senses_concat=config.senses_concat,
                                            center_word_in_window=config.center_word_in_window, vocabulary_size=config.vocabulary_size)
        self.define_second_layer(dropout_prob=config.dropout_prob, num_sampled=config.num_sampled, vocabulary_size=config.vocabulary_size, loss_function=config.loss_function)
        self._init_op = tf.global_variables_initializer()
        if config.write_summary:
            self.define_summaries()
        if not self._context_rep == 1 and (self._probabilistic_sense_assignment == 2 or self._probabilistic_sense_assignment == 3):
            self.build_graph_for_sense_distribution_estimation_with_variable_context_size()

    def dump_all_parameters(self, session, work_dir):
        self.dump_parameters(self._word_embeddings.eval(session=session), work_dir, EMBEDDINGS)

        self.dump_parameters(self._weights_t.eval(session=session), work_dir, MAIN_SOFTMAX_W)

        self.dump_parameters(self._biases.eval(session=session), work_dir, MAIN_SOFTMAX_B)

        if self._probabilistic_sense_assignment == 2:
            self.dump_parameters(self._context_softmax_weights.eval(session=session), work_dir, CTX_SOFTMAX_W)

            self.dump_parameters(self._context_softmax_biases.eval(session=session), work_dir, CTX_SOFTMAX_B)

    def define_histograms(self):
        histograms = super(Word2VecSG, self).define_histograms()
        if self._probabilistic_sense_assignment == 2 or self._probabilistic_sense_assignment == 3:
            assert self._train_grads_and_vars[3][1] == self._context_softmax_weights
            histograms.append(tf.summary.histogram(name="context_softmax_weights", values=self._train_grads_and_vars[3][1]))
            histograms.append(tf.summary.histogram(name="context_softmax_weights_gradients", values=self._train_grads_and_vars[3][0]))
        return histograms

    def l1_loss(self, w):
        l1_loss = tf.reduce_sum(input_tensor=tf.abs(x=w))
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES,
                             value=self._context_softmax_weights_penalty_val * l1_loss)

    def l2_loss(self, w):
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES,
                             value=self._context_softmax_weights_penalty_val * tf.nn.l2_loss(t=w))

    def huber_loss(self, tensor, delta=1.0):
        '''
        https://en.wikipedia.org/wiki/Huber_loss
        '''
        abs_error = tf.abs(x=tensor)
        quadratic = tf.minimum(x=abs_error, y=delta)
        linear = (abs_error - quadratic)
        return 0.5 * quadratic ** 2 + delta * linear

    def pp_loss(self, w):
        ww = tf.matmul(a=w, b=w, transpose_a=True)
        ww_order = ww.get_shape().as_list()[-1]
        neg_ones = [-1.0] * ww_order
        inverted_diag = tf.diag(diagonal=neg_ones) + 1
        ww_with_zero_diag = tf.multiply(x=ww, y=inverted_diag)

        if self._huber_loss:
            losses = self.huber_loss(tensor=ww_with_zero_diag)
        else:
            losses = tf.square(x=ww_with_zero_diag)

        pp_loss = tf.reduce_sum(input_tensor=losses) / ww_order

        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES,
                             value=self._parallel_penalty * pp_loss)

    def get_word_with_subwords_representation_for_each_sense(self, word_emb):
        if self._separate_subwords:
            raise ValueError('Unsupported execution flow')
        else:
            subword_emb = tf.nn.embedding_lookup(params=self._subword_embeddings, ids=self._input_subwords)  # 3D embedding lookup
            mask = tf.cast(x=tf.expand_dims(input=(self._input_subwords > 0), axis=-1), dtype=self._model_dtype)  # 3D mask
            masked_subword_emb = subword_emb * mask
            masked_subword_emb_sum = tf.reduce_sum(input_tensor=masked_subword_emb, axis=1)
            subword_counts_plus_one = tf.reduce_sum(input_tensor=mask, axis=1) + 1  # plus one because we need average of senter word and subwords
            return [(masked_subword_emb_sum + word_emb[s]) / subword_counts_plus_one for s in xrange(self._sense_number)]

    def apply_uncertain_penalty(self, probs):
        '''
        Balanced output distributions penalty
        '''
        assert probs.get_shape()[1] == self._sense_number
        ent = entropy(t=probs, reduction_index=1)
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES,
                             value=self._uncertain_penalty * tf.reduce_mean(input_tensor=ent))

    def define_first_layer_activations(self, dual_learning, senses_concat, center_word_in_window, vocabulary_size):
        mini_batch_size = None  # dynamic mini-batch size
        self._input_center_words = tf.placeholder(dtype=tf.int32, shape=[mini_batch_size], name='input_center_words')

        if self._max_subword_number:  # Enriching Word Vectors with Subword Information
            self._input_subwords = tf.placeholder(dtype=tf.int32, shape=[mini_batch_size, self._max_subword_number], name='input_subwords')

        if self._sense_number == 1:
            self._train_embed = tf.nn.embedding_lookup(params=self._word_embeddings, ids=self._input_center_words)
        else:
            all_word_center_senses = [tf.nn.embedding_lookup(params=self._word_embeddings, ids=self._input_center_words * self._sense_number + s) for s in xrange(self._sense_number)]  # list of tensors of size [batch_size, embedding_size]
            if self._max_subword_number:  # Enriching Word Vectors with Subword Information
                self._all_center_senses = self.get_word_with_subwords_representation_for_each_sense(word_emb=all_word_center_senses)
            else:
                self._all_center_senses = all_word_center_senses

            # prepare context representation
            self._input_contexts = tf.placeholder(dtype=tf.int32, shape=[mini_batch_size, self._window_size], name='input_contexts')

            context_embeddings = []  # list of tensors of shape batch_size x word_embed_size

            for i in xrange(self._window_size):
                if self._context_embedding_source == 0:
                    for s in xrange(self._sense_number):
                        context_embeddings.append(tf.nn.embedding_lookup(params=self._word_embeddings, ids=self._input_contexts[:, i] * self._sense_number + s))
                elif self._context_embedding_source == 1:
                    context_embeddings.append(tf.nn.embedding_lookup(params=self._global_vectors, ids=self._input_contexts[:, i]))
                elif self._context_embedding_source == 2:
                    assert not senses_concat
                    context_embeddings.append(tf.nn.embedding_lookup(params=self._weights_t, ids=self._input_contexts[:, i]))
                else:
                    raise ValueError('Unknown context embedding source: {}.'.format(self._context_embedding_source))

            if center_word_in_window or self._context_rep == 2:
                center_word_embeddings = []
                if self._context_embedding_source == 0:
                    for s in xrange(self._sense_number):
                        center_word_embeddings.append(tf.nn.embedding_lookup(params=self._word_embeddings, ids=self._input_center_words * self._sense_number + s))
                elif self._context_embedding_source == 1:
                    center_word_embeddings.append(tf.nn.embedding_lookup(params=self._global_vectors, ids=self._input_center_words))
                elif self._context_embedding_source == 2:
                    assert not senses_concat
                    center_word_embeddings.append(tf.nn.embedding_lookup(params=self._weights_t, ids=self._input_center_words))
                else:
                    raise ValueError('Unknown context embedding source: {}.'.format(self._context_embedding_source))

            if center_word_in_window:
                context_embeddings.extend(center_word_embeddings)

            if self._probabilistic_sense_assignment == 2 or self._probabilistic_sense_assignment == 3:
                if self._context_rep == 0:
                    context_embedding = tf.add_n(inputs=context_embeddings) / len(context_embeddings)
                elif self._context_rep == 1:
                    context_embedding = tf.concat(axis=1, values=context_embeddings)
                elif self._context_rep == 2:
                    assert not center_word_in_window
                    assert len(center_word_embeddings) == 1
                    avg_context_without_center_embedding = tf.add_n(inputs=context_embeddings) / len(context_embeddings)
                    context_embedding = tf.concat(axis=1, values=[avg_context_without_center_embedding, center_word_embeddings[0]])
                else:
                    raise ValueError('Unknown context representation')

                # context_embedding has the following shape: batch_size x context_embedding_size

                if __debug__:
                    if self._context_rep == 0:
                        expected_emb_size = self._embedding_size
                    elif self._context_rep == 1:
                        expected_emb_size = self._embedding_size * (self._window_size + (1 if center_word_in_window else 0)) * (self._sense_number if self._context_embedding_source == 0 else 1)
                    elif self._context_rep == 2:
                        expected_emb_size = 2 * self._embedding_size

                    assert context_embedding.get_shape().as_list()[1] == expected_emb_size

                if self._context_embedding_source != 1 and not dual_learning:
                    context_embedding = tf.stop_gradient(input=context_embedding)

                if self._context_softmax_weights_penalty:
                    assert self._context_softmax_weights_penalty_val
                else:
                    assert not self._context_softmax_weights_penalty_val

                if self._batch_norm:
                    self._is_training = tf.placeholder(dtype=tf.bool, shape=[], name='is_training')

                if self._probabilistic_sense_assignment == 2:
                    if self._context_softmax_weights_penalty == 1:
                        context_softmax_weights_regularizer = tf.contrib.layers.l1_regularizer(self._context_softmax_weights_penalty_val)
                    elif self._context_softmax_weights_penalty == 2:
                        context_softmax_weights_regularizer = tf.contrib.layers.l2_regularizer(self._context_softmax_weights_penalty_val)
                    else:
                        context_softmax_weights_regularizer = None

                    if self._lower_hidden_units > 0:
                        self._context_hidden_weights = tf.get_variable(name='context_hidden_weights', shape=[context_embedding.get_shape().as_list()[1], self._lower_hidden_units],
                                                                       initializer=tf.contrib.layers.xavier_initializer(), dtype=self._model_dtype)
                        self._saver_variables.append(self._context_hidden_weights)
                        if self._batch_norm == 1:
                            preactivations_before_bn = tf.matmul(a=context_embedding, b=self._context_hidden_weights)
                            hidden_preactivations = tf.contrib.layers.batch_norm(inputs=preactivations_before_bn, is_training=self._is_training, center=True, scale=True, epsilon=EPSILON)
                        else:
                            self._context_hidden_biases = tf.get_variable(name='context_hidden_biases', shape=[self._lower_hidden_units], initializer=tf.contrib.layers.xavier_initializer(), dtype=self._model_dtype)
                            self._saver_variables.append(self._context_hidden_biases)
                            hidden_preactivations = tf.nn.xw_plus_b(x=context_embedding, weights=self._context_hidden_weights, biases=self._context_hidden_biases)
                        context_representation = tf.nn.relu(hidden_preactivations)
                    else:
                        assert self._batch_norm != 1
                        context_representation = context_embedding

                    if self._batch_norm == 2 or self._batch_norm == 3:
                        context_representation = tf.contrib.layers.batch_norm(inputs=context_representation, is_training=self._is_training, center=True, scale=True, epsilon=EPSILON)

                    # init_width = 0.01
                    # self._context_softmax_weights = tf.get_variable(initializer=tf.random_uniform([context_embedding.get_shape().as_list()[1], self._sense_number], -init_width, init_width, dtype=self._model_dtype),
                    #                                                 name='context_softmax_weights', regularizer=context_softmax_weights_regularizer)
                    # self._context_softmax_biases = tf.Variable(tf.random_uniform([self._sense_number], -init_width, init_width, dtype=self._model_dtype), name='context_softmax_biases')
                    self._context_softmax_weights = tf.get_variable(initializer=tf.zeros(shape=[context_representation.get_shape().as_list()[1], self._sense_number], dtype=self._model_dtype),
                                                                    name='context_softmax_weights', regularizer=context_softmax_weights_regularizer)
                    self._context_softmax_biases = tf.Variable(initial_value=tf.zeros(shape=[self._sense_number], dtype=self._model_dtype), name='context_softmax_biases')

                    if self._parallel_penalty:
                        self.pp_loss(w=self._context_softmax_weights)

                    context_logits = tf.nn.xw_plus_b(x=context_representation, weights=self._context_softmax_weights, biases=self._context_softmax_biases)
                elif self._probabilistic_sense_assignment == 3:
                    assert self._lower_hidden_units == 0

                    w_shape = [vocabulary_size, context_embedding.get_shape().as_list()[1], self._sense_number]
                    b_shape = [vocabulary_size, self._sense_number]

                    if os.path.isdir(self._init_model_dir):
                        shared_context_softmax_weights = self.load_init_data(data_name=CTX_SOFTMAX_W, expected_shape=[context_embedding.get_shape().as_list()[1], self._sense_number])
                        shared_context_softmax_biases = self.load_init_data(data_name=CTX_SOFTMAX_B, expected_shape=[self._sense_number])
                        w_initializer = tf.ones(shape=w_shape, dtype=self._model_dtype) * shared_context_softmax_weights
                        b_initializer = tf.ones(shape=b_shape, dtype=self._model_dtype) * shared_context_softmax_biases
                    else:
                        # init_width = init_width = 0.5 / context_embedding.get_shape().as_list()[1] / self._sense_number
                        # w_initializer = tf.random_uniform(shape=w_shape, minval=-init_width, maxval=init_width, dtype=self._model_dtype)
                        # b_initializer = tf.random_uniform(shape=b_shape, minval=-init_width, maxval=init_width, dtype=self._model_dtype)
                        w_initializer = tf.zeros(shape=w_shape, dtype=self._model_dtype)
                        b_initializer = tf.zeros(shape=b_shape, dtype=self._model_dtype)

                    self._context_softmax_weights = tf.get_variable(initializer=w_initializer, name='context_softmax_weights')
                    self._context_softmax_biases = tf.get_variable(initializer=b_initializer, name='context_softmax_biases')

                    assert context_embedding.get_shape().as_list()[1] == self._context_softmax_weights.get_shape().as_list()[1]

                    context_softmax_weights = tf.nn.embedding_lookup(params=self._context_softmax_weights, ids=self._input_center_words)
                    context_softmax_biases = tf.nn.embedding_lookup(params=self._context_softmax_biases, ids=self._input_center_words)

                    if self._context_softmax_weights_penalty == 1:
                        self.l1_loss(w=context_softmax_weights)
                    elif self._context_softmax_weights_penalty == 2:
                        self.l2_loss(w=context_softmax_weights)

                    if self._parallel_penalty:
                        self.pp_loss(w=context_softmax_weights)

                    context_logits = tf.reduce_sum(input_tensor=tf.multiply(x=tf.expand_dims(input=context_embedding, axis=-1), y=context_softmax_weights), axis=1) + context_softmax_biases
                else:
                    raise ValueError('Unknown sense assignment type: {}.'.format(self._probabilistic_sense_assignment))

                if self._learn_ctx_softmax_temp:
                    self._ctx_softmax_temp = tf.Variable(initial_value=[self._learn_ctx_softmax_temp], name='ctx_softmax_temp')
                    self._saver_variables.append(self._ctx_softmax_temp)
                else:
                    self._ctx_softmax_temp = tf.placeholder(dtype=tf.float32, shape=[], name='ctx_softmax_temp')

                if self._relaxed_one_hot:
                    if self._uncertain_penalty:
                        logits = None
                        probs = tf.nn.softmax(logits=context_logits) + 1e-9
                        self.apply_uncertain_penalty(probs=probs)
                    else:
                        logits = context_logits
                        probs = None
                    self._sense_probability_distribution = tf.contrib.distributions.RelaxedOneHotCategorical(temperature=self._ctx_softmax_temp, logits=logits, probs=probs).sample()
                else:
                    context_logits = context_logits / self._ctx_softmax_temp

                    self._sense_probability_distribution = tf.nn.softmax(logits=context_logits)  # batch_size x sense_number

                    if self._uncertain_penalty:
                        self.apply_uncertain_penalty(probs=self._sense_probability_distribution)

                # priors
                self._prior_sense_accumulators = tf.Variable(initial_value=tf.zeros(shape=[vocabulary_size, self._sense_number], dtype=self._model_dtype), name='prior_sense_distributions')
                # one_hot_indices_to_update = tf.one_hot(indices=self._input_center_words, depth=vocabulary_size)
                # deltas = tf.matmul(a=one_hot_indices_to_update, b=self._sense_probability_distribution, transpose_a=True)
                # self._updated_prior_sense_accumulators = prior_sense_accumulators.assign_add(deltas)
                self._updated_prior_sense_accumulators = tf.scatter_add(ref=self._prior_sense_accumulators, indices=self._input_center_words, updates=self._sense_probability_distribution, use_locking=False)

                self._prior_sense_accumulators_new_value = tf.placeholder(dtype=self._model_dtype, shape=[vocabulary_size, self._sense_number], name='prior_sense_accumulators_new_value')
                self._prior_sense_accumulators_assign = tf.assign(self._prior_sense_accumulators, self._prior_sense_accumulators_new_value)

                norm = tf.reduce_sum(input_tensor=self._prior_sense_accumulators, axis=1, keep_dims=True)
                self._prior_sense_distributions = self._prior_sense_accumulators / norm

                self._saver_variables.append(self._context_softmax_weights)
                self._saver_variables.append(self._context_softmax_biases)
                self._saver_variables.append(self._prior_sense_accumulators)
                self._prior_sense_accumulators_initializer = tf.variables_initializer([self._prior_sense_accumulators])
            else:  # either original MSSG or variant 1 of PMSSG
                context_sum = tf.add_n(inputs=context_embeddings)

                norm = tf.sqrt(x=tf.reduce_sum(input_tensor=tf.square(x=context_sum), axis=1, keep_dims=True))
                normalized_context_mean = context_sum / norm

                # calculate similarities of context to center word senses

                if not self._probabilistic_sense_assignment:
                    single_batch_context_cluster_centers = [tf.nn.embedding_lookup(params=self._context_cluster_centers, ids=self._input_center_words * self._sense_number + s) for s in xrange(self._sense_number)]

                similarities = []  # list of tensors of size [batch_size, 1]

                for s in xrange(self._sense_number):
                    if not self._probabilistic_sense_assignment:
                        norm = tf.sqrt(x=tf.reduce_sum(input_tensor=tf.square(x=single_batch_context_cluster_centers[s]), axis=1, keep_dims=True))
                        normalized_context_cluster_centers = single_batch_context_cluster_centers[s] / norm
                        ewm = tf.multiply(x=normalized_context_mean, y=tf.stop_gradient(input=normalized_context_cluster_centers))  # element-wise multiplication of context means and centers
                    else:
                        norm = tf.sqrt(x=tf.reduce_sum(input_tensor=tf.square(x=self._all_center_senses[s]), axis=1, keep_dims=True))
                        normalized_center_senses = self._all_center_senses[s] / norm
                        ewm = tf.multiply(x=normalized_context_mean, y=tf.stop_gradient(normalized_center_senses))  # element-wise multiplication of con

                    sense_similarity = tf.reduce_sum(input_tensor=ewm, axis=1, keep_dims=True)

                    similarities.append(sense_similarity)

                if self._probabilistic_sense_assignment:
                    similarities_sum = tf.add_n(inputs=similarities)
                    all_center_senses_probabilities = [similarities[s] / similarities_sum for s in xrange(self._sense_number)]  # perheps this can be merged with the following line
                    self._sense_probability_distribution = tf.stack(values=all_center_senses_probabilities, axis=1)

            if self._probabilistic_sense_assignment:
                assert self._sense_probability_distribution.get_shape().as_list()[1] == self._sense_number

                all_weighted_center_senses = []  # list of tensors of shape batch_size x embed_size

                for s in xrange(self._sense_number):
                    tiled_sense_probabilities = tf.tile(input=self._sense_probability_distribution[:, s: s + 1], multiples=[1, self._embedding_size])  # batch_size x embed_size

                    if self._probabilistic_sense_assignment == 1 and self._context_embedding_source != 1 and not dual_learning:
                        tiled_sense_probabilities = tf.stop_gradient(input=tiled_sense_probabilities)

                    weighted_center_senses = tf.multiply(x=self._all_center_senses[s], y=tiled_sense_probabilities)

                    all_weighted_center_senses.append(weighted_center_senses)

                self._train_embed = tf.concat(axis=1, values=all_weighted_center_senses) if senses_concat else tf.add_n(inputs=all_weighted_center_senses)

                if self._batch_norm == 3:
                    self._train_embed = tf.contrib.layers.batch_norm(inputs=self._train_embed, is_training=self._is_training, center=True, scale=True, epsilon=EPSILON)

                if self._batch_norm:  # regardless whether variant 1, 2 or 3
                    batch_norm_vars = [v for v in tf.global_variables() if 'BatchNorm' in v.name]
                    expected_bn_var_number = 8 if self._batch_norm == 3 else 4
                    assert len(batch_norm_vars) == expected_bn_var_number  # beta, gamma, moving_mean, moving_variance
                    self._saver_variables.extend(batch_norm_vars)
                    self._updated_moving_mean_and_moving_variance = tf.get_collection(key=tf.GraphKeys.UPDATE_OPS)

            else:  # Original MSSG as defined by Neelakantan et al.
                similarity_matrix = tf.concat(axis=1, values=similarities)

                most_similar_senses = tf.argmax(input=similarity_matrix, axis=1)

                if self._context_embedding_source != 1:
                    most_similar_senses = tf.stop_gradient(input=most_similar_senses)

                # is it necessary to lookup senses once more?
                self._train_embed = tf.nn.embedding_lookup(params=self._word_embeddings, ids=self._input_center_words * self._sense_number + most_similar_senses)  # resultant 3D tensor

                self._updated_context_cluster_centers = tf.scatter_add(ref=self._context_cluster_centers, indices=self._input_center_words * self._sense_number + most_similar_senses, updates=context_sum / len(context_embeddings), use_locking=False)

    def get_train_var_list(self):
        var_list = super(Word2VecSG, self).get_train_var_list()

        if self._probabilistic_sense_assignment == 2 or self._probabilistic_sense_assignment == 3:
            var_list.append(self._context_softmax_weights)
            var_list.append(self._context_softmax_biases)

        if self._probabilistic_sense_assignment == 2 and self._lower_hidden_units > 0:
            var_list.append(self._context_hidden_weights)
            if self._batch_norm != 1:
                var_list.append(self._context_hidden_biases)

        if self._batch_norm:
            batch_norm_trainable_vars = [v for v in tf.global_variables() if 'BatchNorm' in v.name and ('beta' in v.name or 'gamma' in v.name)]
            batch_norm_trainable_vars_expected_number = 4 if self._batch_norm == 3 else 2
            assert len(batch_norm_trainable_vars) == batch_norm_trainable_vars_expected_number
            var_list.extend(batch_norm_trainable_vars)

        if self._max_subword_number:
            var_list.append(self._subword_embeddings)

        if self._learn_ctx_softmax_temp:
            var_list.append(self._ctx_softmax_temp)

        return var_list

    def parse_line(self, line, dictionary, pad_every_sentence):
        """
        Can we take advantage of tf.pad() ?
        """
        unknown_word_id = dictionary[UNKNOWN_WORD]
        sentence = parse_line_base(line, dictionary, self._tokenization_type)
        if pad_every_sentence:
            return [unknown_word_id for i in xrange(self._left_window_size)] + sentence + [unknown_word_id for i in xrange(self._right_window_size)]
        if len(sentence) < self._window_size + 1:  # does this padding make sense in case of skip-gram ???
            pre_pad = [unknown_word_id for i in xrange(self._window_size + 1 - len(sentence))]  # TODO: this only works correctly for left-side windows; should be adapted to both_side windows as well
            return pre_pad + sentence
        return sentence

    def generate_batches(self, requested_batch_size, dictionary, trainset_file, pad_every_sentence):
        """
        It is assumed that trainset_file is a multi-line plain text file
        """
        logging.info('Generating mini-batches')

        batch_id = 0
        context = []
        current_position_in_batch = 0
        center_word_label = np.ndarray(shape=(requested_batch_size, 2), dtype=np.int32)
        unknown_word_id = dictionary[UNKNOWN_WORD]

        with open(trainset_file, 'r') as f:
            for l in f:
                sentence = self.parse_line(l, dictionary, pad_every_sentence)

                for target_word_id in xrange(self._left_window_size, len(sentence) - self._right_window_size):  # TODO: is sentence padded or not? if not then should not it be rather xrange(len(sentence)) ???
                    for context_word_id in generate_context(sentence, target_word_id, self._window_size, self._left_window_size, self._right_window_size):
                        if context_word_id == unknown_word_id:  # in case of Skip-gram model, predicting context word based on an unknown center word does not make sense
                            continue
                        center_word_label[current_position_in_batch, 0] = sentence[target_word_id]
                        center_word_label[current_position_in_batch, 1] = context_word_id
                        current_position_in_batch += 1
                        if current_position_in_batch == requested_batch_size:
                            yield (batch_id, center_word_label.copy())  # we return a COPY of center_word_label ndarray, therefore we do not need to reinitialize it
                            current_position_in_batch = 0
                            batch_id += 1

    def generate_batches_multiple_senses(self, requested_batch_size, dictionary, trainset_file, pad_every_sentence, test_word_id=None, line_number_range=None):
        """
        It is assumed that trainset_file is a multi-line plain text file
        """
        if line_number_range is None:
            logging.info('Generating mini-batches')
        else:
            from multiprocessing import current_process
            logging.info('Generating mini-batches in process {} for docs in range {}.'.format(current_process().name, line_number_range))

        batch_id = 0
        context = []
        current_position_in_batch = 0
        context_center_word_label = np.ndarray(shape=(requested_batch_size, self._window_size + 2), dtype=np.int32)
        unknown_word_id = dictionary[UNKNOWN_WORD]

        with open(trainset_file, 'r') as f:
            for l in enumerate(f):
                if line_number_range is not None:
                    line_number = l[0]
                    if line_number < line_number_range[0]:
                        continue
                    if line_number >= line_number_range[1]:
                        break

                sentence = self.parse_line(l[1], dictionary, pad_every_sentence)

                for target_word_id in xrange(self._left_window_size, len(sentence) - self._right_window_size):
                    if test_word_id and test_word_id != sentence[target_word_id]:
                        continue

                    context = generate_context(sentence, target_word_id, self._window_size, self._left_window_size, self._right_window_size)

                    def add_to_batch(label):
                        context_center_word_label[current_position_in_batch, :-2] = context
                        context_center_word_label[current_position_in_batch, -2] = sentence[target_word_id]
                        context_center_word_label[current_position_in_batch, -1] = label
                        return current_position_in_batch + 1

                    if test_word_id:
                        current_position_in_batch = add_to_batch(-1)
                        if current_position_in_batch == requested_batch_size:
                            yield (batch_id, context_center_word_label.copy())  # we return a COPY of context_center_word_label ndarray, therefore we do not need to reinitialize it
                            current_position_in_batch = 0
                            batch_id += 1
                    else:
                        for context_word_id in context:
                            if context_word_id == unknown_word_id:  # in case of Skip-gram model, predicting context word based on an unknown center word does not make sense
                                continue
                            current_position_in_batch = add_to_batch(context_word_id)
                            if current_position_in_batch == requested_batch_size:
                                yield (batch_id, context_center_word_label.copy())  # we return a COPY of context_center_word_label ndarray, therefore we do not need to reinitialize it
                                current_position_in_batch = 0
                                batch_id += 1

    def analyze_sense_distributions(self, session, requested_batch_size, dictionary, reverse_dictionary, testset_file, test_words, max_dist_to_print, priors):
        for test_word in test_words:
            if priors is not None:
                pruned_senses = priors[dictionary[test_word]] < PRIOR_PRUNING_THRESHOLD
            else:
                pruned_senses = [False for x in xrange(self._sense_number)]

            for sense_id in xrange(self._sense_number):
                if pruned_senses[sense_id]:
                    continue
                logging.info('Sense distribution for word \'{}\' where sense {} dominates:'.format(test_word, sense_id + 1))
                assert test_word == reverse_dictionary[dictionary[test_word]]
                sum_of_distribution = np.ndarray([self._sense_number], dtype=np.float32)
                context_count = 0
                printed_distribution = 0
                for batch in self.generate_batches_multiple_senses(requested_batch_size=requested_batch_size, dictionary=dictionary, trainset_file=testset_file, pad_every_sentence=False, test_word_id=dictionary[test_word]):
                    sense_probability_distribution = self._sense_probability_distribution.eval(session=session, feed_dict=self.get_feed_dict(batch=batch, lr_to_pass=0.0, ctx_softmax_temp=None, is_training=False, word_subwords=None))
                    # print only the first context from the mini-batch biased toward one of the senses
                    example_id = np.argmax(sense_probability_distribution[:, sense_id])
                    if np.argmax(sense_probability_distribution[example_id, :]) == sense_id:
                        example = batch[1][example_id, :]
                        assert example[-1] == -1
                        assert example[-2] == dictionary[test_word]
                        left_window = ' '.join([reverse_dictionary[example[word_id]] for word_id in xrange(self._left_window_size)])
                        right_window = ' '.join([reverse_dictionary[example[self._left_window_size + word_id]] for word_id in xrange(self._right_window_size)])
                        context_to_be_printed = left_window + ' [' + reverse_dictionary[example[-2]] + '] ' + right_window
                        logging.info('  {:<50} - {}'.format(context_to_be_printed, sense_probability_distribution[example_id, :]))
                        printed_distribution += 1
                        sum_of_distribution += sense_probability_distribution[example_id, :]
                        context_count += 1
                    if printed_distribution > max_dist_to_print:
                        break
                if context_count > 0:
                    avg_of_dist = sum_of_distribution / context_count
                    logging.info('Average distribution of word \'{}\': {}'.format(test_word, avg_of_dist))

    def get_global_embeddings(self, session):
        assert self._context_embedding_source == 1
        return self._global_vectors.eval(session=session)

    def get_feed_dict(self, batch, lr_to_pass, ctx_softmax_temp, is_training, word_subwords):
        if self._sense_number == 1:
            batch_inputs = batch[1][:, 0]
            batch_labels = batch[1][:, 1]
            batch_labels = batch_labels.reshape((batch_labels.shape[0], 1))
            feed_dict = {self._input_center_words: batch_inputs, self._input_labels: batch_labels}
        else:
            batch_contexts = batch[1][:, :-2]
            batch_center_words = batch[1][:, -2]
            batch_labels = batch[1][:, -1]
            batch_labels = batch_labels.reshape((batch_labels.shape[0], 1))
            feed_dict = {self._input_contexts: batch_contexts, self._input_center_words: batch_center_words, self._input_labels: batch_labels}
            if self._max_subword_number and is_training:  # TODO do we need subwords when not training (e.g. prior estimation)
                batch_subwords = np.zeros(dtype=np.int32, shape=(batch[1].shape[0], max_subword_number))  # should it be zeros or sth elese?
                for i, w in enumerate(batch_center_words):
                    subwords = word_subwords[w][:self._max_subword_number]  # TODO: add random selection; random selection should be done once before training
                    batch_subwords[i, :len(subwords)] = subwords
                feed_dict[self._input_subwords] = batch_subwords

        feed_dict[self._lr_ph] = lr_to_pass

        if self._batch_norm:
            feed_dict[self._is_training] = is_training

        if not self._learn_ctx_softmax_temp:
            if is_training:
                if len(ctx_softmax_temp) == 1:
                    feed_dict[self._ctx_softmax_temp] = ctx_softmax_temp[0]
                elif len(ctx_softmax_temp) == 2:
                    decay_factor = float(batch[0]) / APPROXIMATELY_WIKI_BATCH_NUMBER
                    start_stop_diff = ctx_softmax_temp[0] - ctx_softmax_temp[1]
                    feed_dict[self._ctx_softmax_temp] = ctx_softmax_temp[0] - start_stop_diff * decay_factor
                else:
                    raise ValueError('Wrong number of ctx softmax temp values')
            else:
                feed_dict[self._ctx_softmax_temp] = 1.0

        return feed_dict

    def get_context_softmax_weights(self, session):
        return self._context_softmax_weights.eval(session=session)

    def get_priors(self, session, dictionary):
        priors = self._prior_sense_distributions.eval(session=session)
        unknown_word_id = dictionary[UNKNOWN_WORD]
        for s in xrange(self._sense_number):
            priors[unknown_word_id, s] = 1.0 / 3  # there is no point in learning priors for the unknown word
        assert not np.isnan(priors).any()  # in case of very small datasets could this assertion be invalid?
        # priors[np.isnan(priors)] = 1.0 / 3
        return priors

    def check_left_context(self, left_context, dictionary):
        unknown_word_id = dictionary[UNKNOWN_WORD]
        return [unknown_word_id for i in xrange(self._left_window_size - len(left_context))] + left_context

    def check_right_context(self, right_context, dictionary):
        unknown_word_id = dictionary[UNKNOWN_WORD]
        return right_context + [unknown_word_id for i in xrange(self._right_window_size - len(right_context))]

    def build_graph_for_sense_distribution_estimation_with_variable_context_size(self):
        self._variable_size_input_contexts = tf.placeholder(dtype=tf.int32, shape=[None], name='variable_size_input_contexts')

        if self._context_embedding_source == 0:
            single_sense_context_embeddings = []

            for s in xrange(self._sense_number):
                single_sense_context_embeddings.append(tf.nn.embedding_lookup(params=self._word_embeddings, ids=(self._variable_size_input_contexts * self._sense_number + s)))

            context_words_embeddings = tf.concat(axis=0, values=single_sense_context_embeddings)
        else:
            if self._context_embedding_source == 1:
                embeddings = self._global_vectors
            elif self._context_embedding_source == 2:
                embeddings = self._weights_t
            else:
                raise ValueError('Unsupported context embedding source: {}.'.format(self._context_embedding_source))

            context_words_embeddings = tf.nn.embedding_lookup(params=embeddings, ids=self._variable_size_input_contexts)

        context_embedding = tf.reduce_mean(input_tensor=context_words_embeddings, axis=0, keep_dims=True)

        if self._context_rep == 2:
            avg_center_senses_embeddings = tf.add_n(inputs=self._all_center_senses) / len(self._all_center_senses)

            context_embedding = tf.concat(axis=1, values=[context_embedding, avg_center_senses_embeddings])

        if self._lower_hidden_units > 0:
            assert self._probabilistic_sense_assignment == 2
            if self._batch_norm == 1:
                preactivations_before_bn = tf.matmul(a=context_embedding, b=self._context_hidden_weights)

                with tf.variable_scope('BatchNorm', reuse=True):
                    m = tf.get_variable('moving_mean')
                    v = tf.get_variable('moving_variance')
                    b = tf.get_variable('beta')
                    g = tf.get_variable('gamma')

                hidden_preactivations = tf.nn.batch_normalization(x=preactivations_before_bn, mean=m, variance=v, offset=b, scale=g, variance_epsilon=EPSILON)
            else:
                hidden_preactivations = tf.nn.xw_plus_b(x=context_embedding, weights=self._context_hidden_weights, biases=self._context_hidden_biases)
            context_representation = tf.nn.relu(hidden_preactivations)
        else:
            context_representation = context_embedding

        if self._batch_norm == 2 or self._batch_norm == 3:
            with tf.variable_scope('BatchNorm', reuse=True):  # even in case of variant 3 of batch_norm this should work because upper softmax BN vars will have _1 suffix; so those without suffix will be lower softmax BN vars (I hope)
                m = tf.get_variable('moving_mean')
                v = tf.get_variable('moving_variance')
                b = tf.get_variable('beta')
                g = tf.get_variable('gamma')

            context_representation = tf.nn.batch_normalization(x=context_representation, mean=m, variance=v, offset=b, scale=g, variance_epsilon=EPSILON)

        if self._probabilistic_sense_assignment == 2:
            weights = self._context_softmax_weights
            biases = self._context_softmax_biases
        elif self._probabilistic_sense_assignment == 3:
            # in case of variable context mini-batch size is 1 so we can do this:
            weights = self._context_softmax_weights[self._input_center_words[0], :, :]
            biases = self._context_softmax_biases[self._input_center_words[0], :]
        else:
            raise ValueError('Unsupported prob. sense assign. type: {}.'.format(self._context_embedding_source))

        context_logits = tf.nn.xw_plus_b(x=context_representation, weights=weights, biases=biases)

        self._sense_probability_distribution_for_variable_context_size = tf.nn.softmax(logits=context_logits)

    def get_feed_dict_for_variable_size_context(self, w, c):
        context_with_central_word = c[0] + [w] + c[1]
        assert len(context_with_central_word) == len(c[0]) + len(c[1]) + 1
        return {self._variable_size_input_contexts: context_with_central_word, self._input_center_words: [w]}  # centr word is needed only for variant 3 of prob. sense. assign.

    def estimate_sense_distribution_single(self, session, w, c, variable_context_size, dictionary, priors):
        if variable_context_size:
            return self._sense_probability_distribution_for_variable_context_size.eval(
                feed_dict=self.get_feed_dict_for_variable_size_context(w=w, c=c), session=session)
        else:
            context_center_word_label = np.ndarray(shape=(1, self._window_size + 2), dtype=np.int32)
            context_left = self.check_left_context(c[0], dictionary)
            context_right = self.check_right_context(c[1], dictionary)
            context_center_word_label[0, :-2] = context_left[-self._left_window_size:] + context_right[:self._right_window_size]
            context_center_word_label[0, -2] = w
            batch = (0, context_center_word_label)
            dist = self._sense_probability_distribution.eval(feed_dict=self.get_feed_dict(batch=batch, lr_to_pass=0.0, ctx_softmax_temp=None, is_training=False, word_subwords=None), session=session)
            if priors is not None:
                pruned_senses = priors[w] < PRIOR_PRUNING_THRESHOLD
                dist[0, pruned_senses] = 0.0
            return dist

    def estimate_sense_distribution(self, session, word1id, word2id, context1, context2, variable_context_size, dictionary, priors):
        """
        There are two possibilities for implementing this: either we can model the same window as during training
        or we can model bigger window - possible as big ad the context provided in the test set
        In the second variant we need to model context as an average of word embeddings instead of concatenation.
        Returns ndarray of shape (2, sense_number)
        """
        return np.vstack((
            self.estimate_sense_distribution_single(session=session, w=word1id, c=context1, variable_context_size=variable_context_size, dictionary=dictionary, priors=priors),
            self.estimate_sense_distribution_single(session=session, w=word2id, c=context2, variable_context_size=variable_context_size, dictionary=dictionary, priors=priors)))


class Word2VecCBOW(Word2VecBase):
    def __init__(self, vocabulary_size, word_embedding_size, num_sampled, loss_function, optimizer_class,
                 window, two_side_window, context_concat, cbow_mean, dropout_prob,
                 write_summary, tokenization_type, model_dtype):
        window_size = window + (window if two_side_window else 0)
        self._hidden_layer_size = word_embedding_size * (window_size if context_concat else 1)
        self.define_common_variables(vocabulary_size=vocabulary_size, embedding_size=word_embedding_size, window=window, two_side_window=two_side_window,
                                     sense_number=1, context_embedding_source=0, tokenization_type=tokenization_type, model_dtype=model_dtype, optimizer_class=optimizer_class, init_model_dir=None)
        self.define_first_layer_activations(context_concat=context_concat, cbow_mean=cbow_mean,
                                            vocabulary_size=vocabulary_size, word_embedding_size=word_embedding_size)
        self.define_second_layer(dropout_prob=dropout_prob, num_sampled=num_sampled, vocabulary_size=vocabulary_size, loss_function=loss_function)
        self._init_op = tf.global_variables_initializer()

        if write_summary:
            self.define_summaries()

    def define_first_layer_activations(self, context_concat, cbow_mean, vocabulary_size, word_embedding_size):
        """
        Returns hidden layer size
        """
        self._input_contexts = tf.placeholder(dtype=tf.int32, shape=[None, self._window_size], name='input_contexts')

        context_embeddings = []

        for i in xrange(self._window_size):
            word_embed = tf.nn.embedding_lookup(params=self._word_embeddings, ids=self._input_contexts[:, i])
            context_embeddings.append(word_embed)

        if context_concat:
            self._train_embed = tf.concat(axis=1, values=context_embeddings)
        else:
            self._train_embed = tf.add_n(inputs=context_embeddings)
            if cbow_mean:
                self._train_embed /= self._window_size

    def get_feed_dict(self, batch, lr_to_pass, ctx_softmax_temp=None, is_training=None):
        batch_contexts = batch[1][:, :-1]
        batch_labels = batch[1][:, -1]
        batch_labels = batch_labels.reshape((batch_labels.shape[0], 1))
        return {self._input_contexts: batch_contexts, self._input_labels: batch_labels, self._lr_ph: lr_to_pass}

    def generate_batches(self, requested_batch_size, dictionary, trainset_file):
        return generate_cbow_batches(requested_batch_size=requested_batch_size, dictionary=dictionary, trainset_file=trainset_file, window_size=self._window_size,
                                     left_window_size=self._left_window_size, right_window_size=self._right_window_size, tokenization_type=self._tokenization_type)


def estimate_priors(dictionary, trainset_file, batch_size, worker_number, pad_every_sentence, work_dir, train_epoch_num, word2VecSGConfig):
    from multiprocessing import Process, Queue, current_process
    logging.info('Building prior probabilities')

    single_process_doc_number = float(file_len(trainset_file)) / worker_number

    def log_progress(batch_id):
        if batch_id % 1000 == 0:
            logging.info('Process: {}, batch: {}.'.format(current_process().name, batch_id))

    def process_batch(session, model, batch):
        feed_dict = model.get_feed_dict(batch=batch, lr_to_pass=0.0, ctx_softmax_temp=None, is_training=False, word_subwords=None)
        model._updated_prior_sense_accumulators.eval(session=session, feed_dict=feed_dict)
        return batch[0]

    def accumulate_priors(pid, queue):
        logging.info('Accumulate priors in process {}.'.format(current_process().name))
        start_doc_id = int(pid * single_process_doc_number)
        end_doc_id = int((pid + 1) * single_process_doc_number)

        with tf.Graph().as_default():
            model = Word2VecSG(config=word2VecSGConfig)
            with tf.Session() as session:
                model.restore_epoch(session=session, work_dir=work_dir, epoch_number=train_epoch_num)
                session.run(model._prior_sense_accumulators_initializer)
                batch_iterator = model.generate_batches_multiple_senses(batch_size, dictionary, trainset_file, pad_every_sentence, line_number_range=[start_doc_id, end_doc_id])
                for batch in batch_iterator:
                    log_progress(process_batch(session=session, model=model, batch=batch))
                logging.info('Putting accumulators to the queue from process {}'.format(current_process().name))
                queue.put(model._prior_sense_accumulators.eval(session=session))

    queue = Queue()

    for i in xrange(worker_number):
        p = Process(target=accumulate_priors, kwargs={'pid': i, 'queue': queue})
        p.start()

    accumulators = []

    for i in xrange(worker_number):
        logging.info('Retrieving accumulators from the queue')
        accumulators.append(queue.get())

    logging.info('Reducing accumulators')
    combined_accumulators = reduce(np.add, accumulators)

    with tf.Graph().as_default():
        model = Word2VecSG(config=word2VecSGConfig)
        with tf.Session() as session:
            model.restore_epoch(session=session, work_dir=work_dir, epoch_number=train_epoch_num)
            session.run(model._prior_sense_accumulators_initializer)
            session.run(model._prior_sense_accumulators_assign, feed_dict={model._prior_sense_accumulators_new_value: combined_accumulators})
            model.save_epoch(session=session, work_dir=work_dir, epoch_number=train_epoch_num)


def pickle_embeddings(embeddings, file_name):
    embed_file = os.path.join(work_dir, file_name)

    logging.info('Pickling embeddings to ' + embed_file + ' ...')

    with open(embed_file, 'wb') as f:
        cPickle.dump((embeddings, dictionary, reverse_dictionary), f)

    logging.info('GZIPing ' + embed_file)
    os.system('gzip -f ' + embed_file)

    logging.info('Embeddings successfully created')


def compute_subword_hashes(word, subwords_range, subword_bucket_number):
    """
    Compute values between [1,subword_bucket_number]. Zero is reserved to pad words having less subwords than max subword number.
    """
    ngrams = []

    for b in xrange(len(word)):
        for e in xrange(b + subwords_range[0], b + subwords_range[1] + 1):
            if e > len(word):
                break
            ngrams.append(word[b:e])

    return [fnv1a_32(ngram) % subword_bucket_number + 1 for ngram in ngrams]


def build_word_subwords(dictionary, subwords_range, subword_bucket_number):
    """
    Buld dictionary mapping word IDs to IDs of buckets containing its subwords
    """
    return {w_id: compute_subword_hashes(word=w, subwords_range=subwords_range, subword_bucket_number=subword_bucket_number) for w, w_id in dictionary.iteritems()}


if __name__ == '__main__':
    from docopt import docopt

    random.seed(123)
    np.random.seed(123)

    arg = docopt(__doc__)
    print(arg)

    work_dir = arg['--work_dir']
    trainset_file = arg['--trainset_file']
    ta_sg = arg['--ta'] == 'SG'
    word_embedding_size = int(arg['--word_dim_num'])
    batch_size = int(arg['--batch_size'])
    num_sampled = int(arg['--num_sampled'])
    train_epoch_num = int(arg['--train_epoch_num'])
    lr_train = map(float, arg['--lr_train'].split(','))
    momentum = map(float, arg['--momentum'].split(','))
    dropout_prob = float(arg['--dropout_prob'])
    write_summary = parse_bool(arg['--write_summary'])
    loss_function = get_loss_function(arg['--loss_function'])
    optimizer_class = get_optimizer(arg['--optimizer'])
    window = int(arg['--window'])
    two_side_window = parse_bool(arg['--two_side_window'])
    center_word_in_window = parse_bool(arg['--center-word-in-win'])
    sense_number = int(arg['--sense_number'])
    probabilistic_sense_assignment = int(arg['--prob_sense_assign'])
    context_embedding_source = int(arg['--ctx-embedding-src'])
    worker_number = int(arg['--worker_number'])
    tokenization_type = arg['--token_type']
    context_rep = int(arg['--context-rep'])
    dual_learning = parse_bool(arg['--dual-learning'])
    senses_concat = parse_bool(arg['--senses-concat'])
    test_words = arg['--test_words']
    model_dtype = tf.float32
    dict_file = arg['--dict_file']
    save_each_epoch = parse_bool(arg['--save_each_epoch'])
    pad_every_sentence = parse_bool(arg['--pad_every_sentence'])
    mode = int(arg['--mode'])
    subwords_range = map(int, arg['--subwords_range'].split('-'))
    subword_bucket_number = int(arg['--subword_bucket_num'])

    def inner_split(part):
        return map(float, part.split(','))

    ctx_softmax_temp = map(inner_split, arg['--ctx_softmax_temp'].split('_'))

    logging.info('Node name: {}'.format(os.uname()[1]))

    logging.info('Loading dict from ' + dict_file)
    with gzip.open(dict_file, 'rb') as f:
        dictionary = cPickle.load(f)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

    if subwords_range == [0, 0]:
        max_subword_number = 0
        word_subwords = None
    else:
        assert subwords_range[0] < subwords_range[1]
        word_subwords = build_word_subwords(dictionary=dictionary, subwords_range=subwords_range, subword_bucket_number=subword_bucket_number)
        max_subword_number = int(arg['--max_subword_number'])
        if not max_subword_number:
            max_subword_number = max([len(v) for v in word_subwords.itervalues()])
        logging.info('Max subword number {}'.format(max_subword_number))
        assert min([min(v) if v else 1 for v in word_subwords.itervalues()]) == 1
        assert max([max(v) if v else 1 for v in word_subwords.itervalues()]) == subword_bucket_number

    if ta_sg:
        word2VecSGConfig = Word2VecSGConfig(vocabulary_size=len(dictionary), word_embedding_size=word_embedding_size, num_sampled=num_sampled,
                                            loss_function=loss_function, optimizer_class=optimizer_class, dropout_prob=dropout_prob, write_summary=write_summary,
                                            window=window, two_side_window=two_side_window, center_word_in_window=center_word_in_window, sense_number=sense_number,
                                            probabilistic_sense_assignment=probabilistic_sense_assignment, context_embedding_source=context_embedding_source,
                                            dual_learning=dual_learning, context_rep=context_rep, senses_concat=senses_concat, tokenization_type=tokenization_type,
                                            model_dtype=model_dtype, lower_hidden_units=int(arg['--lower_hidden_units']),
                                            batch_norm=int(arg['--batch_norm']), context_softmax_weights_penalty=int(arg['--lower_penalty']),
                                            context_softmax_weights_penalty_val=float(arg['--lower_penalty_val']), uncertain_penalty=float(arg['--uncertain_penalty']),
                                            parallel_penalty=float(arg['--parallel_penalty']), huber_loss=int(arg['--huber_loss']), init_model_dir=arg['--init_model_dir'],
                                            relaxed_one_hot=int(arg['--relaxed_one_hot']), subword_bucket_number=subword_bucket_number, max_subword_number=max_subword_number,
                                            separate_subwords=int(arg['--separate_subwords']), learn_ctx_softmax_temp=float(arg['--learn_ctx_smx_temp']),
                                            check_num_batch=int(arg['--check_num_batch']))
    if mode in (0, 2, 3, 4, 5, 6, 7):
        with tf.Graph().as_default():
            tf.set_random_seed(123)
            if ta_sg:
                model = Word2VecSG(config=word2VecSGConfig)
            else:
                model = Word2VecCBOW(vocabulary_size=len(dictionary), word_embedding_size=word_embedding_size, num_sampled=num_sampled,
                                     loss_function=loss_function, optimizer_class=optimizer_class, window=window,
                                     two_side_window=two_side_window, context_concat=context_rep == 1, cbow_mean=parse_bool(arg['--cbow_mean']),
                                     dropout_prob=dropout_prob, write_summary=write_summary, tokenization_type=tokenization_type, model_dtype=model_dtype)

            with tf.Session() as session:
                if mode == 0:
                    if train_epoch_num == 1 or not save_each_epoch:
                        model.init(session=session)

                    if save_each_epoch and train_epoch_num > 1:
                        model.restore_epoch(session=session, work_dir=work_dir, epoch_number=train_epoch_num - 1)

                    model.train(session=session, dictionary=dictionary, reverse_dictionary=reverse_dictionary,
                                trainset_file=trainset_file, train_epoch_num=train_epoch_num, batch_size=batch_size,
                                lr_train=lr_train, momentum=momentum, write_summary=write_summary, work_dir=work_dir,
                                worker_number=worker_number, save_each_epoch=save_each_epoch, pad_every_sentence=pad_every_sentence,
                                ctx_softmax_temp=ctx_softmax_temp, word_subwords=word_subwords)
                elif mode in (2, 3, 4, 5, 6, 7):
                    model.restore_epoch(session=session, work_dir=work_dir, epoch_number=train_epoch_num)
                    if mode == 2:
                        test_words = test_words.split(',')
                        if probabilistic_sense_assignment == 2 or probabilistic_sense_assignment == 3:
                            priors = model.get_priors(session=session, dictionary=dictionary)
                            estimate_sense_distribution_function = lambda w1, w2, c1, c2, full_context: model.estimate_sense_distribution(session, w1, w2, c1, c2, variable_context_size=full_context, dictionary=dictionary, priors=priors)
                        else:
                            priors = None
                            estimate_sense_distribution_function = None
                        logging.info('Test embeddings correlation')
                        test_correlation_and_print_nn(embeddings=model.get_word_embeddings(session=session), priors=priors,
                                                      dictionary=dictionary, reverse_dictionary=reverse_dictionary, sense_number=sense_number, test_words=test_words,
                                                      estimate_sense_distribution_function=estimate_sense_distribution_function, window_size=window, two_side_window=two_side_window,
                                                      work_dir=work_dir, epoch_number=train_epoch_num)
                        logging.info('Test output embeddings correlation')
                        test_correlation_and_print_nn(embeddings=model.get_output_embeddings(session=session), priors=None,
                                                      dictionary=dictionary, reverse_dictionary=reverse_dictionary, sense_number=1, test_words=test_words,
                                                      estimate_sense_distribution_function=estimate_sense_distribution_function, window_size=window, two_side_window=two_side_window)
                        if context_embedding_source == 1:
                            logging.info('Test global embeddings correlation')
                            test_correlation_and_print_nn(embeddings=model.get_global_embeddings(session=session), priors=None,
                                                          dictionary=dictionary, reverse_dictionary=reverse_dictionary, sense_number=1, test_words=test_words,
                                                          estimate_sense_distribution_function=estimate_sense_distribution_function, window_size=window, two_side_window=two_side_window)
                        if probabilistic_sense_assignment == 2 or probabilistic_sense_assignment == 3:
                            model.analyze_sense_distributions(session=session, requested_batch_size=batch_size, dictionary=dictionary, reverse_dictionary=reverse_dictionary,
                                                              testset_file=trainset_file, test_words=test_words, max_dist_to_print=25, priors=priors)
                    elif mode in (3, 7):
                        wsi_dir = arg['--wsi_dir']
                        estimate_sense_distribution_model_fct = lambda w, c, variable_context_size: model.estimate_sense_distribution_single(
                            session=session, w=w, c=c, variable_context_size=variable_context_size, dictionary=dictionary, priors=priors).flatten()
                        if mode == 3:
                            embeddings = model.get_word_embeddings(session=session)
                        elif mode == 7:
                            ctx_smx_w = model.get_context_softmax_weights(session=session)
                            expected_shape = (ctx_smx_w.shape[0] * ctx_smx_w.shape[2], ctx_smx_w.shape[1])
                            embeddings = np.transpose(ctx_smx_w, (0, 2, 1)).reshape(expected_shape)
                            # alternatively this could be done the following way:
                            # to_concat = [ctx_smx_w[:, :, s] for s in xrange(ctx_smx_w.shape[2])]
                            # embeddings = np.concatenate(to_concat, axis=1).reshape(expected_shape)
                        else:
                            raise Exception('Unsupported mode {}'.format(mode))
                        output_embeddings = model.get_output_embeddings(session=session)
                        priors = model.get_priors(session=session, dictionary=dictionary)
                        estimate_sense_distribution_cosine_fct = lambda w, c, variable_context_size, prior_weighting, ctx_based_on_out_emb: estimate_probability_distribution_cosine(
                            embeddings=embeddings, sense_number=sense_number, context_embeddings=(output_embeddings if ctx_based_on_out_emb else embeddings), context_sense_number=(1 if ctx_based_on_out_emb else sense_number),
                            dictionary=dictionary, word_id=w, context=c, window_size=(None if variable_context_size else window),
                            two_side_window=two_side_window, priors=priors, prior_weighting=prior_weighting)
                        test_wsi(wsi_dir=wsi_dir, dictionary=dictionary, estimate_sense_distribution_model_fct=estimate_sense_distribution_model_fct, estimate_sense_distribution_cosine_fct=estimate_sense_distribution_cosine_fct)
                    elif mode == 4:
                        assert train_epoch_num == 1
                        model.dump_all_parameters(session=session, work_dir=work_dir)
                    elif mode == 5:
                        reduce_dims(embeddings=model.get_word_embeddings(session=session), work_dir=work_dir, epoch_number=train_epoch_num)
                    elif mode == 6:
                        doc_corpus_dir = arg['--doc_corpus_dir']
                        doc_represent_dir = arg['--doc_represent_dir']
                        doc_rep_disamb = int(arg['--doc_rep_disamb'])
                        doc_rep_disamb_out = int(arg['--doc_rep_disamb_out'])
                        word_embeddings = model.get_word_embeddings(session=session)
                        if doc_rep_disamb_out:
                            context_embeddings = model.get_output_embeddings(session=session)
                            context_sense_number = 1
                        else:
                            context_embeddings = word_embeddings
                            context_sense_number = sense_number

                        build_document_representation(dictionary=dictionary, dimmensionality=word_embedding_size,
                                                      word_embeddings=word_embeddings, context_embeddings=context_embeddings,
                                                      sense_number=sense_number, context_sense_number=context_sense_number,
                                                      dataset=os.path.join(doc_corpus_dir, 'train.txt'),
                                                      labels_file=os.path.join(doc_corpus_dir, 'train_labels.txt'),
                                                      document_representation=os.path.join(doc_represent_dir, 'train.pkl.gz'),
                                                      disambiguation=doc_rep_disamb)
                        build_document_representation(dictionary=dictionary, dimmensionality=word_embedding_size,
                                                      word_embeddings=word_embeddings, context_embeddings=context_embeddings,
                                                      sense_number=sense_number, context_sense_number=context_sense_number,
                                                      dataset=os.path.join(doc_corpus_dir, 'test.txt'),
                                                      labels_file=os.path.join(doc_corpus_dir, 'test_labels.txt'),
                                                      document_representation=os.path.join(doc_represent_dir, 'test.pkl.gz'),
                                                      disambiguation=doc_rep_disamb)
                    else:
                        raise ValueError('Unsupported mode: {}!'.format(mode))
                else:
                    raise ValueError('Unknown mode: {}!'.format(mode))
    elif mode == 1:
            estimate_priors(dictionary=dictionary, trainset_file=trainset_file, batch_size=batch_size, worker_number=worker_number,
                            pad_every_sentence=pad_every_sentence, work_dir=work_dir, train_epoch_num=train_epoch_num, word2VecSGConfig=word2VecSGConfig)
    else:
        raise ValueError('Unknown mode: {}!'.format(mode))
