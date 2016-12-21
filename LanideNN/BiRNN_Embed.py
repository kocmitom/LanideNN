import os

import numpy as np
import tensorflow as tf
import time
import logging
from tensorflow.models.rnn import rnn, rnn_cell, seq2seq

from LanideNN import NNModel


class Model(NNModel.Model):
    def __init__(self, sess, params, vocabs_size):
        NNModel.Model.__init__(self, vocabs_size)

        self.params = params

        self.batch_size = self.params.get("batch_size")
        self.max_length = self.params.get("max_length")
        self.size = self.params.get("size")
        self.num_layers = self.params.get("num_layers")
        # the learning rate could be a float, but this way we can adjust it during training
        # self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
        self.learning_rate = self.params.get("learning_rate")
        self.embedding_size = self.params.get("embedding_size")
        # self.global_step = tf.Variable(0, trainable=False)
        self.incorrect = [0] * self.max_length
        self.global_step = 0
        self.corpus_name = self.params.get("corpus_name")

        logging.info(
            "BiRNN model created with {0} layers of {1} cells. Embedding = {2}. Vocabulary sizes = {3}, length = {4}, batch = {5}.".format(self.num_layers, self.size, self.embedding_size, vocabs_size,
                                                                                                                                           self.max_length, self.batch_size))

        # forward RNN
        with tf.variable_scope('forward'):
            fcell = rnn_cell.GRUCell(self.size, input_size=self.embedding_size)
            forward_cell = fcell
            if self.num_layers > 1:
                fcell2 = rnn_cell.GRUCell(self.size)
                forward_cell = rnn_cell.MultiRNNCell([fcell] + ([fcell2] * self.num_layers))

        # backward RNN
        with tf.variable_scope('backward'):
            bcell = rnn_cell.GRUCell(self.size, input_size=self.embedding_size)
            backward_cell = bcell
            if self.num_layers > 1:
                bcell2 = rnn_cell.GRUCell(self.size)
                backward_cell = rnn_cell.MultiRNNCell([bcell] + ([bcell2] * self.num_layers))

        #seq_len = tf.fill([self.batch_size], constant(self.max_length, dtype=tf.int64))

        # self.inputs = tf.placeholder(tf.float32, shape=[self.max_length, self.batch_size, self.vocab_sizes[0]], name="inputs")
        self.inputs = [tf.placeholder(tf.int32, shape=[None], name="inputs{0}".format(i)) for i in range(self.max_length)]
        self.targets = [tf.placeholder(tf.int32, shape=[None], name="targets{0}".format(i)) for i in range(self.max_length)]

        self.sentence_lengths = tf.placeholder(tf.int64, shape=[None], name="sequence_lengths")
        self.dropout_placeholder = tf.placeholder(tf.float32, shape=[], name="dropout")

        self.word_embeddings = tf.Variable(tf.random_uniform([self.vocab_sizes[0], self.embedding_size], -1.0, 1.0))
        embedded_inputs = [tf.nn.embedding_lookup(self.word_embeddings, input_) for input_ in self.inputs]
        dropped_embedded_inputs = [tf.nn.dropout(i, self.dropout_placeholder) for i in embedded_inputs]  # dropout je realny cislo

        weights = {
            # Hidden layer weights => 2*n_hidden because of foward + backward cells
            # 'hidden': tf.Variable(tf.random_uniform([self.vocab_sizes[0], 2 * size]), name="hidden-weight"),
            'out': tf.Variable(tf.random_uniform([2 * self.size, self.vocab_sizes[1]]), name="out-weight")
        }
        biases = {
            # 'hidden': tf.Variable(tf.random_uniform([2 * size]), name="hidden-bias"),
            'out': tf.Variable(tf.random_uniform([self.vocab_sizes[1]]), name="out-bias")
        }

        # hack to omit information from RNN creation
        logging.getLogger().setLevel(logging.CRITICAL)
        with tf.variable_scope('BiRNN-net'):
            # bidi_layer = BidirectionalRNNLayer(forward_cell, backward_cell, dropped_embedded_inputs, self.sentence_lengths)
            # with tf.variable_scope('forward'):
            #     output_fw, last_state = rnn.rnn(cell=forward_cell, inputs=dropped_embedded_inputs, dtype=tf.float32, sequence_length=self.sentence_lengths)
            #
            # with tf.variable_scope('backward'):
            #     outputs_rev_rev, last_state_rev = rnn.rnn(cell=backward_cell, inputs=rnn._reverse_seq(dropped_embedded_inputs, self.sentence_lengths), dtype=tf.float32,
            #                                               sequence_length=self.sentence_lengths)
            #     output_bw = self.rnn._reverse_seq(outputs_rev_rev, self.sentence_lengths)
            #
            # outputs = [array_ops.concat(1, [fw, bw]) for fw, bw in zip(output_fw, output_bw)]
            outputs = rnn.bidirectional_rnn(forward_cell, backward_cell, dropped_embedded_inputs, sequence_length=self.sentence_lengths, dtype=tf.float32)

        logging.getLogger().setLevel(logging.INFO)

        self.out = []
        self.probs = []
        # after switch to TF 0.8 it started outputing some merges for FC a BC
        for o in outputs[0]:
            # TODO ############# pridat tf.nn.relu(MATMUL+BIAs) ???
            intermediate_out = tf.matmul(o, weights['out']) + biases['out']
            self.out.append(intermediate_out)
            self.probs.append(tf.nn.softmax(intermediate_out))

        loss = seq2seq.sequence_loss_by_example(self.out, self.targets, [tf.ones([self.batch_size])] * self.max_length, self.vocab_sizes[1])

        self.cost = tf.reduce_sum(loss) / self.batch_size

        tf.scalar_summary("Cost", self.cost)

        self.updates = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

        self.saver = tf.train.Saver(max_to_keep=0) # don't remove old models

        self.summaries = tf.merge_all_summaries()
        self.sum_writer = tf.python.training.summary_io.SummaryWriter("tmp", sess.graph)

        # Initializing the variables & Launch the graph

        sess.run(tf.initialize_all_variables())
        logging.info("BiRNN model initialized.")

    def feed(self, inputs, sentence_lengths, dropout, targets=None):
        input_feed = {}
        for l in range(self.max_length):
            input_feed[self.inputs[l].name] = inputs[l]
            if targets is not None:
                input_feed[self.targets[l].name] = targets[l]

        input_feed[self.sentence_lengths.name] = sentence_lengths
        input_feed[self.dropout_placeholder.name] = dropout
        return input_feed

    def eval_probs(self, session, inputs, sentence_lengths):
        input_feed = self.feed(inputs, sentence_lengths, 1)

        outs = session.run(self.probs, input_feed)

        return outs

    def eval(self, session, inputs, sentence_lengths, langs_mask=None):
        input_feed = self.feed(inputs, sentence_lengths, 1)

        outs = session.run(self.out, input_feed)

        outputs = np.zeros((self.max_length, self.batch_size), dtype=np.int)
        for j in range(self.batch_size):
            for i in range(self.max_length):
                if langs_mask is not None:
                    for max in np.argsort(-outs[i][j]):
                        if langs_mask[max] == 1:
                            outputs[i][j] = max # don't forget that the values can be negative
                            break
                else:
                    outputs[i][j] = np.argmax(outs[i][j])



        return outputs

    def run(self, forward_only, session, inputs, targets, sentence_lengths, dropout, rev_vocab=False):
        input_feed = self.feed(inputs, sentence_lengths, dropout, targets=targets)

        # Fit training using batch data
        if not forward_only:
            session.run(self.updates, input_feed)
            # result = session.run([self.summaries, self.updates], input_feed)
            # self.sum_writer.add_summary(result[0], self.global_step)
            # self.global_step += 1
        else:
            return self.compute_correctness(session.run(self.out, input_feed), targets, rev_vocab)

    def compute_correctness(self, outputs, targets, rev_vocab=False):
        # voted_outputs = self.output_crop(outputs)

        correct = 0
        total = 0
        for j in range(self.batch_size):
            for i in range(self.max_length):
                # print('Guess: ', np.argmax(outputs[i][j]), ' Trg:', targets[i][j])
                if np.argmax(outputs[i][j]) == targets[i][j]:
                    correct += 1
                else:
                    # print("incorrectly: ", np.argmax(outputs[i][j]) , targets[i][j])
                    self.incorrect[i] += 1

                total += 1
        # print(self.incorrect)
        # print(correct, "/", float(total))
        return [correct, total]

    def save(self, session, step, result):
        directory = "models/" + self.corpus_name
        if not os.path.exists(directory):
            os.makedirs(directory)

        filename = directory + "/BiRNN.%d.step%d.m%d_s%d-l%d.model" % (time.time(), step, self.max_length, self.size, self.num_layers)

        self.saver.save(session, filename)
        self.params.save_result(result, filename)
        print("Saving model into " + filename)
