from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

class ESIM(object):
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
    def _parameter_init(self, seq_length,hidden_size, batch_size, learning_rate):
        """
        :param seq_length: max sentence length
        :param hidden_size: hidden dims
        :param batch_size: batch size
        :param learning_rate: learning rate
        """
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate

    # placeholder declaration
    def _placeholder_init(self):
        """
        premise_mask: actual length of premise sentence
        hypothesis_mask: actual length of hypothesis sentence
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

    # biLSTM unit
    @staticmethod
    def lstm_layer(inputs, input_seq_len, rnn_size, dropout_keep_prob, scope, scope_reuse=False):
        with tf.variable_scope(scope, reuse=scope_reuse) as vs:
            fw_cell = tf.contrib.rnn.LSTMCell(rnn_size, forget_bias=1.0, state_is_tuple=True, reuse=scope_reuse)
            fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell, output_keep_prob=dropout_keep_prob)
            bw_cell = tf.contrib.rnn.LSTMCell(rnn_size, forget_bias=1.0, state_is_tuple=True, reuse=scope_reuse)
            bw_cell = tf.contrib.rnn.DropoutWrapper(bw_cell, output_keep_prob=dropout_keep_prob)
            rnn_outputs, rnn_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell, cell_bw=bw_cell,
                                                                      inputs=inputs,
                                                                      sequence_length=input_seq_len,
                                                                      dtype=tf.float32)
            return rnn_outputs, rnn_states

    @staticmethod
    def ffnn_layer(inputs, output_size, dropout_keep_prob, scope, scope_reuse=False):
        with tf.variable_scope(scope, reuse=scope_reuse):
            input_size = inputs.get_shape()[-1].value
            W = tf.get_variable("W_trans", shape=[input_size, output_size], initializer=tf.orthogonal_initializer())
            b = tf.get_variable("b_trans", shape=[output_size, ], initializer=tf.zeros_initializer())
            outputs = tf.nn.relu(tf.einsum('aij,jk->aik', inputs, W) + b)
            outputs = tf.nn.dropout(outputs, keep_prob=dropout_keep_prob)
        return outputs

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
        with tf.device('/cpu:0'):
            premise_embedded = tf.nn.embedding_lookup(self.Embedding, self.premise)
            hypothesis_embedded = tf.nn.embedding_lookup(self.Embedding, self.hypothesis)
            self.premise_embedded    = tf.nn.dropout(premise_embedded, keep_prob=self.dropout_keep_prob)
            self.hypothesis_embedded = tf.nn.dropout(hypothesis_embedded, keep_prob=self.dropout_keep_prob)
            print("shape of premise_embedded: {}".format(premise_embedded.get_shape()))
            print("shape of hypothesis_embedded: {}".format(hypothesis_embedded.get_shape()))

    def encoding_layer(self):
        with tf.variable_scope("encoding_layer") as vs:
            rnn_scope_name = "bidirectional_rnn"
            p_rnn_output, p_rnn_states = self.lstm_layer(self.premise_embedded, self.premise_len, self.hidden_size, self.dropout_keep_prob, rnn_scope_name, scope_reuse=False)   # [batch_size, sequence_length, rnn_size(200)]
            self.premise_output = tf.concat(axis=2, values=p_rnn_output)     # [batch_size, maxlen, rnn_size*2]
            h_rnn_output, h_rnn_states = self.lstm_layer(self.hypothesis_embedded, self.hypothesis_len, self.hidden_size, self.dropout_keep_prob, rnn_scope_name, scope_reuse=True)
            self.hypothesis_output = tf.concat(axis=2, values=h_rnn_output)   # [batch_size, maxlen, rnn_size*2]
            print('Incorporate single_lstm_layer successfully.')

    def matching_layer(self):
        with tf.variable_scope("matching_layer") as vs:
            similarity = self.premise_hypothesis_similarity_matrix(self.premise_output, self.hypothesis_output)  # [batch_size, answer_len, question_len]
            attended_premise = self.attend_premise(similarity, self.hypothesis_output, self.hypothesis_len,self.seq_length)  # [batch_size, maxlen, dim]
            attended_hypothesis = self.attend_hypothesis(similarity, self.premise_output, self.premise_len,self.seq_length)  # [batch_size, maxlen, dim]

            m_p = tf.concat(axis=2,
                            values=[self.premise_output, attended_premise, tf.multiply(self.premise_output, attended_premise),
                                    self.premise_output - attended_premise])
            m_h = tf.concat(axis=2, values=[self.hypothesis_output, attended_hypothesis,
                                            tf.multiply(self.hypothesis_output, attended_hypothesis),
                                            self.hypothesis_output - attended_hypothesis])

            # m_ffnn
            m_input_size = m_p.get_shape()[-1].value
            m_output_size = m_input_size
            m_p = self.ffnn_layer(m_p, m_output_size, self.dropout_keep_prob, "m_ffnn", scope_reuse=False)
            m_h = self.ffnn_layer(m_h, m_output_size, self.dropout_keep_prob, "m_ffnn", scope_reuse=True)
            print('Incorporate ffnn_layer after cross attention successfully.')
            rnn_scope_cross = 'bidirectional_rnn_cross'
            rnn_size_layer_2 = self.hidden_size
            rnn_output_p_2, rnn_states_p_2 = self.lstm_layer(m_p, self.premise_len, rnn_size_layer_2, self.dropout_keep_prob,
                                                        rnn_scope_cross, scope_reuse=False)
            rnn_output_h_2, rnn_states_h_2 = self.lstm_layer(m_h, self.hypothesis_len, rnn_size_layer_2,
                                                        self.dropout_keep_prob, rnn_scope_cross, scope_reuse=True)
            self.premise_output_cross = tf.concat(axis=2, values=rnn_output_p_2)  # [batch_size, sequence_length, 2*rnn_size(400)]
            self.hypothesis_output_cross = tf.concat(axis=2, values=rnn_output_h_2)

    def aggregation_layer(self):
        with tf.variable_scope("aggregation_layer") as vs:
            premise_max = tf.reduce_max(self.premise_output_cross, axis=1)  # [batch_size, 2*rnn_size(400)]
            hypothesis_max = tf.reduce_max(self.hypothesis_output_cross, axis=1)
            premise_mean = tf.reduce_mean(self.premise_output_cross, axis=1)  # [batch_size, 2*rnn_size(400)]
            hypothesis_mean = tf.reduce_mean(self.hypothesis_output_cross, axis=1)
            joined_feature = tf.concat(axis=1, values=[premise_max, hypothesis_max, premise_mean,hypothesis_mean])  # [batch_size, 8*rnn_size(1600)]
            print("shape of joined feature: {}".format(joined_feature.get_shape()))
        return joined_feature

    def prediction_layer(self,joined_feature):
        with tf.variable_scope("prediction_layer"):
            hidden_output_size = 256
            joined_feature = tf.nn.dropout(joined_feature, keep_prob=self.dropout_keep_prob)
            full_out = tf.contrib.layers.fully_connected(joined_feature, hidden_output_size,
                                                         activation_fn=tf.nn.relu,
                                                         reuse=False,
                                                         trainable=True,
                                                         scope="projected_layer")  # [batch_size, hidden_output_size(256)]
            full_out = tf.nn.dropout(full_out, keep_prob=self.dropout_keep_prob)
            last_weight_dim = full_out.get_shape()[1].value
            print("last_weight_dim: {}".format(last_weight_dim))
            bias = tf.Variable(tf.constant(0.1, shape=[3]), name="bias")
            s_w = tf.get_variable("s_w", shape=[last_weight_dim, 3], initializer=tf.contrib.layers.xavier_initializer())
            logits = tf.matmul(full_out, s_w) + bias  # [batch_size, 3]
            print("shape of logits: {}".format(logits.get_shape()))
        return logits

    def _logits_op(self):
        self.embedding_layer()
        self.encoding_layer()
        self.matching_layer()
        joined_feature = self.aggregation_layer()
        logits = self.prediction_layer(joined_feature)
        return logits

    # calculate classification loss
    def _loss_op(self):
        l2_loss = tf.constant(0.0)
        l2_reg_lambda = 0.0
        regularizer = tf.contrib.layers.l2_regularizer(l2_reg_lambda)
        with tf.name_scope('cost'):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.target)
            mean_loss = tf.reduce_mean(losses, name="mean_loss") + l2_reg_lambda * l2_loss + sum(
                                                              tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
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