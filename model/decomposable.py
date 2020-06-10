from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
class Decomposable(object):
    def __init__(self, seq_length, hidden_size, batch_size, learning_rate, embeddings):
        # model init
        self._parameter_init(seq_length, hidden_size, batch_size, learning_rate)
        self._placeholder_init()
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self._embedding_init(embeddings)
        # model operation
        self.logits = self._logits_op()
        self.mean_loss = self._loss_op()
        self.accuracy = self._acc_op()
        self.train = self._training_op()

    # init hyper-parameters
    def _parameter_init(self, seq_length, hidden_size, batch_size, learning_rate):
        """
        :param seq_length: max sentence length
        :param hidden_size: hidden dims
        :param batch_size: batch size
        :param learning_rate: learning rate
        :param clip_value: if gradients value bigger than this value, clip it
        """
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate

    # placeholder declaration
    def _placeholder_init(self):
        """
        premise_len: actual length of premise sentence
        hypothesis_len: actual length of hypothesis sentence
        embed_matrix: with shape (n_vocab, embedding_size)
        dropout_keep_prob: dropout keep probability
        :return:
        """
        self.premise = tf.placeholder(tf.int32, [None, self.seq_length], 'premise')
        self.hypothesis = tf.placeholder(tf.int32, [None, self.seq_length], 'hypothesis')

        self.premise_len = tf.placeholder(tf.int32, [None], 'premise_len')
        self.hypothesis_len = tf.placeholder(tf.int32, [None], 'hypothesis_len')

        self.target = tf.placeholder(tf.int64, [None], name="target")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

    def _embedding_init(self,embeddings):
        self.Embedding = tf.constant(embeddings, name = 'Embedding')
        #self.Embedding = tf.get_variable('Embedding',embeddings.shape,initializer=tf.constant_initializer(embeddings))

    @staticmethod
    def ffnn_layer(inputs, output_size, dropout_keep_prob, scope, scope_reuse=False):
        with tf.variable_scope(scope, reuse=scope_reuse):
            inputs = tf.nn.dropout(inputs, dropout_keep_prob)
            outputs = tf.layers.dense(inputs, output_size, tf.nn.relu, kernel_initializer=tf.orthogonal_initializer())
            outputs = tf.nn.dropout(outputs, dropout_keep_prob)
            resluts = tf.layers.dense(outputs, output_size, tf.nn.relu, kernel_initializer=tf.orthogonal_initializer())
        return resluts

    @staticmethod
    def premise_hypothesis_similarity_matrix(premise, hypothesis):
        # [batch_size, dim, p_len]
        p2 = tf.transpose(premise, perm=[0, 2, 1])
        # [batch_size, h_len, p_len]
        similarity = tf.matmul(hypothesis, p2, name='similarity_matrix')
        return similarity

    @staticmethod
    def attend_hypothesis(similarity_matrix, premise, premise_len, maxlen):
        # similarity_matrix: [batch_size, h_len, p_len]
        # premise: [batch_size, p_len, dim]
        # masked similarity_matrix
        mask_p = tf.sequence_mask(premise_len, maxlen, dtype=tf.float32)  # [batch_size, p_len]
        mask_p = tf.expand_dims(mask_p, 1)  # [batch_size, 1, p_len]
        similarity_matrix = similarity_matrix * mask_p + -1e9 * (1 - mask_p)  # [batch_size, h_len, p_len]
        # [batch_size, h_len, p_len]
        attention_weight_for_p = tf.nn.softmax(similarity_matrix, dim=-1)
        # [batch_size, a_len, dim]
        attended_hypothesis = tf.matmul(attention_weight_for_p, premise)
        return attended_hypothesis

    @staticmethod
    def attend_premise(similarity_matrix, hypothesis, hypothesis_len, maxlen):
        # similarity_matrix: [batch_size, h_len, p_len]
        # hypothesis: [batch_size, h_len, dim]
        # masked similarity_matrix
        mask_h = tf.sequence_mask(hypothesis_len, maxlen, dtype=tf.float32)  # [batch_size, h_len]
        mask_h = tf.expand_dims(mask_h, 2)  # [batch_size, h_len, 1]
        similarity_matrix = similarity_matrix * mask_h + -1e9 * (1 - mask_h)  # [batch_size, h_len, p_len]
        # [batch_size, p_len, h_len]
        attention_weight_for_h = tf.nn.softmax(tf.transpose(similarity_matrix, perm=[0, 2, 1]), dim=-1)
        # [batch_size, p_len, dim]
        attended_premise = tf.matmul(attention_weight_for_h, hypothesis)
        return attended_premise

    def embedding_layer(self):
        self.premise_embedded = tf.nn.embedding_lookup(self.Embedding, self.premise)
        self.hypothesis_embedded = tf.nn.embedding_lookup(self.Embedding, self.hypothesis)
        print("shape of premise_embedded: {}".format(self.premise_embedded.get_shape()))
        print("shape of hypothesis_embedded: {}".format(self.hypothesis_embedded.get_shape()))

    def encoding_layer(self):
        with tf.variable_scope("encoding_layer") as vs:
            self.premise_output = self.ffnn_layer(self.premise_embedded, self.hidden_size, self.dropout_keep_prob, 'F', scope_reuse=False)
            self.hypothesis_output = self.ffnn_layer(self.hypothesis_embedded, self.hidden_size, self.dropout_keep_prob, 'F', scope_reuse=True)

    def matching_layer(self):
        with tf.variable_scope("matching_layer") as vs:
            similarity = self.premise_hypothesis_similarity_matrix(self.premise_output,
                                                                   self.hypothesis_output)  # [batch_size, answer_len, question_len]
            self.attended_premise = self.attend_premise(similarity, self.hypothesis_output, self.hypothesis_len,
                                                   self.seq_length)  # [batch_size, maxlen, dim]
            self.attended_hypothesis = self.attend_hypothesis(similarity, self.premise_output, self.premise_len,
                                                         self.seq_length)  # [batch_size, maxlen, dim]

    def comparing_layer(self):
        with tf.variable_scope("comparing_layer") as vs:
            premise_concat = tf.concat([self.premise_embedded, self.attended_premise], axis=2)
            hypothesis_concat = tf.concat([self.hypothesis_embedded, self.attended_hypothesis], axis=2)
            self.premise_concat_ff = self.ffnn_layer(premise_concat, self.hidden_size, self.dropout_keep_prob, 'G', scope_reuse=False)
            self.hypothesis_concat_ff = self.ffnn_layer(hypothesis_concat, self.hidden_size, self.dropout_keep_prob, 'G', scope_reuse=True)

    def aggregation_layer(self):
        with tf.variable_scope("aggregation_layer") as vs:
            premise_concat_sum = tf.reduce_sum(self.premise_concat_ff, axis=1)
            hypothesis_concat_sum = tf.reduce_sum(self.hypothesis_concat_ff, axis=1)
            joined_feature = tf.concat([premise_concat_sum, hypothesis_concat_sum], axis=1)
            full_out = self.ffnn_layer(joined_feature, self.hidden_size, self.dropout_keep_prob, 'H', scope_reuse=False)
            logits = tf.layers.dense(full_out, 2, kernel_initializer=tf.orthogonal_initializer())
            print("shape of logits: {}".format(logits.get_shape()))
        return logits

    def _logits_op(self):
        self.embedding_layer()
        self.encoding_layer()
        self.matching_layer()
        self.comparing_layer()
        logits = self.aggregation_layer()
        return logits

    # calculate classification loss
    def _loss_op(self):
        with tf.name_scope('cost'):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.target)
            mean_loss = tf.reduce_mean(losses, name="mean_loss")
        return mean_loss

    # calculate classification accuracy
    def _acc_op(self):
        with tf.name_scope('acc'):
            self.probs = tf.nn.softmax(self.logits, name="prob")   # [batch_size, n_class(3)]
            correct_prediction = tf.equal(tf.argmax(self.probs, 1), self.target)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='Accuracy')
        return accuracy

    # define optimizer
    def _training_op(self):
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step,5000, 0.96, staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        grads_and_vars = optimizer.compute_gradients(self.mean_loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)
        return train_op