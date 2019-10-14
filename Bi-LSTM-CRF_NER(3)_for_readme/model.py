import model_utils
import ner_data_utils
import metrics
import tensorflow as tf
import pandas as pd
import numpy as np
import config
import warnings

warnings.filterwarnings('ignore')


class NERTagger:
    def __init__(self, mdl_path, final_mdl_path, num_units, optimizer, learning_rate, epochs, batch_size, activation,
                 keep_probability, keep_probability_d):
        """

        :param mdl_path
        :param final_mdl_path
        :param num_units: The number of LSTM cells
        :param optimizer: optimizer
        :param learning_rate: <dtype: float>
        :param epochs: <dtype: integer>
        :param batch_size: <dtype: integer>
        :param activation: activation function
        :param keep_probability: <dtype: float> keep probability of DropoutWrapper
        :param keep_probability_d: <dtype: float> keep probability of dropout on output layer
        """
        self.mdl_path = mdl_path
        self.final_mdl_path = final_mdl_path
        self.num_units = num_units
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.activation = activation
        self.keep_probability = keep_probability
        self.keep_probability_d = keep_probability_d

    def _placeholder(self):
        with tf.variable_scope("placeholder"):
            # shape = (batch, batch_seq_len, input_size)
            self.input_data = tf.placeholder(tf.float32,
                                             [None, None, self.input_size],
                                             name="input_data")
            # shape = (batch, sentence)
            self.labels = tf.placeholder(tf.int32,
                                         shape=[None, None],
                                         name="labels")

            # max sequence length in a batch
            self.batch_sequence_length = tf.placeholder(tf.int32)

            # the number of morphemes in each sents.
            self.original_sequence_lengths = tf.placeholder(tf.int32, [None])

            # keep_prob in lstm cells
            self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")

    def _lstm(self, num_units):
        """
        Creates LSTM cell
        :return: lstm cell wrapped with dropout.
        """
        lstm = tf.nn.rnn_cell.LSTMCell(num_units)  # lstm cell
        lstm = tf.nn.rnn_cell.DropoutWrapper(lstm,
                                             input_keep_prob=self.keep_prob,
                                             output_keep_prob=self.keep_prob,
                                             state_keep_prob=self.keep_prob)
        return lstm

    def _bi_lstm(self):
        """
        Creates a bi-directional lstm.
        :return: Tuple of fw and bw output.
        """
        fw_cell = self._lstm(self.num_units)
        bw_cell = self._lstm(self.num_units)

        (output_fw, output_bw), states = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell,
                                                                         cell_bw=bw_cell,
                                                                         inputs=self.input_data,
                                                                         sequence_length=self.original_sequence_lengths,
                                                                         dtype=tf.float32,
                                                                         scope="BiLSTM")
        return output_fw, output_bw

    def _build_graph(self):
        self._placeholder()

        with tf.variable_scope("bi_lstm"):
            _outputs = self._bi_lstm()
            outputs = tf.concat(list(_outputs), axis=2)  # [output_fw, output_bw], need checking

        with tf.variable_scope("projection"):
            training = tf.placeholder_with_default(False, shape=(), name='training')
            he_init = tf.contrib.layers.variance_scaling_initializer()

            outputs_flat = tf.reshape(outputs, [-1, 2 * self.num_units])
            outputs_flat_drop = tf.layers.dropout(inputs=outputs_flat,
                                                  rate=self.keep_probability_d,
                                                  training=training)
            self.pred = tf.layers.dense(inputs=outputs_flat_drop,
                                        units=self.num_classes,
                                        activation=self.activation,
                                        kernel_initializer=he_init,
                                        name="pred")
            self.logits = tf.reshape(self.pred, [-1, self.batch_sequence_length, self.num_classes])

        with tf.variable_scope('crf'):
            log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(self.logits,
                                                                                  self.labels,
                                                                                  self.original_sequence_lengths)
            self.loss = tf.reduce_mean(-log_likelihood)

            # Compute the viterbi sequence and score (used for prediction and test time).
            self.viterbi_sequence, self.viterbi_score = tf.contrib.crf.crf_decode(self.logits, transition_params,
                                                                                  self.original_sequence_lengths)

        with tf.name_scope("train"):
            optimizer = self.optimizer(learning_rate=self.learning_rate)
            self.train_op = optimizer.minimize(self.loss)

        for op in (self.viterbi_sequence, self.input_data, self.labels, self.batch_sequence_length,
                   self.original_sequence_lengths, self.keep_prob):
            tf.add_to_collection("operations", op)

    def train(self, train, valid):
        X_tr, self.y_tr, whole_seq_len_tr, self.input_size, self.num_classes = train
        X_val, y_val, whole_seq_len_val, input_size_val, _ = valid

        self._build_graph()

        best_loss_val = np.infty
        checks = 0
        max_checks = config.max_checks

        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(self.epochs):
                i = 0
                self.y_pred_tr = []
                for batch_data, batch_labels, batch_seq_len, batch_sequence_lengths \
                        in model_utils.mini_batch(X_tr, self.y_tr, self.input_size, whole_seq_len_tr, self.batch_size):
                    tf_viterbi_sequence, _ = sess.run([self.viterbi_sequence, self.train_op],
                                                      feed_dict={self.input_data: batch_data,
                                                                 self.labels: batch_labels,
                                                                 self.batch_sequence_length: batch_seq_len,
                                                                 self.original_sequence_lengths: batch_sequence_lengths,
                                                                 self.keep_prob: self.keep_probability})
                    self.y_pred_tr.extend(tf_viterbi_sequence)
                    if i % self.batch_size == 0:
                        for batch_data, batch_labels, batch_seq_len, batch_sequence_lengths \
                                in model_utils.mini_batch(X_val, y_val, input_size_val, whole_seq_len_val, batch_size=1):
                            loss_val = self.loss.eval(feed_dict={self.input_data: batch_data,
                                                                 self.labels: batch_labels,
                                                                 self.batch_sequence_length: batch_seq_len,
                                                                 self.original_sequence_lengths: batch_sequence_lengths,
                                                                 self.keep_prob: 1})
                            # print("loss_val: ", loss_val, "best_loss_val: ", best_loss_val)
                        if loss_val < best_loss_val:
                            saver.save(sess, self.mdl_path, global_step=epoch)
                            best_loss_val = loss_val
                            checks = 0
                        else:
                            checks += 1
                        i += 1

                print("epoch {} \t\tbest_loss_val: {:.6f}".format(epoch, best_loss_val))
                if checks > max_checks:
                    print("Early stop.")
                    break

            saver.save(sess, self.final_mdl_path)

    def test(self, test):  # for evaluation
        X_te, self.y_te, whole_seq_len_te, input_size_te, self.num_classes = test

        viterbi_sequence, input_data, labels, batch_sequence_length, original_sequence_lengths, keep_prob = tf.get_collection("operations")

        saver = tf.train.import_meta_graph(self.final_mdl_path + ".meta")
        with tf.Session() as sess:
            saver.restore(sess, self.final_mdl_path)
            _y_pred_te = []
            for batch_data, batch_labels, batch_seq_len, batch_sequence_lengths \
                    in model_utils.mini_batch(X_te, self.y_te, input_size_te, whole_seq_len_te, batch_size=1):
                tf_viterbi_sequence = sess.run(viterbi_sequence,
                                               feed_dict={input_data: batch_data,
                                                          labels: batch_labels,
                                                          batch_sequence_length: batch_seq_len,
                                                          original_sequence_lengths: batch_sequence_lengths,
                                                          keep_prob: 1})
                _y_pred_te.extend(tf_viterbi_sequence)
            self.y_pred_te = _y_pred_te

    def test_real(self, test):  # for real test data set(without NE tags.) => generate result.
        X_te, whole_seq_len_te, input_size_te, self.num_classes = test

        # viterbi_sequence, input_data, labels, batch_sequence_length, original_sequence_lengths, keep_prob = tf.get_collection("operations")

        saver = tf.train.import_meta_graph(self.final_mdl_path + ".meta")

        viterbi_sequence, input_data, labels, batch_sequence_length, original_sequence_lengths, keep_prob = tf.get_collection("operations")
        with tf.Session() as sess:
            saver.restore(sess, self.final_mdl_path)
            _y_pred_te = []
            for batch_data, batch_seq_len, batch_sequence_lengths \
                    in model_utils.mini_batch_te(X_te, input_size_te, whole_seq_len_te, batch_size=1):
                tf_viterbi_sequence = sess.run(viterbi_sequence,
                                               feed_dict={input_data: batch_data,
                                                          batch_sequence_length: batch_seq_len,
                                                          original_sequence_lengths: batch_sequence_lengths,
                                                          keep_prob: 1})
                _y_pred_te.extend(tf_viterbi_sequence)
            self.y_pred_te = _y_pred_te

    def metrics(self, op):
        """ evaluation
        :param op: <dtype: str>
        """
        # if op == "train":
        #     metrics.metrics(self.y_tr, self.y_pred_tr)
        if op == "test":
           metrics.metrics(self.y_te, self.y_pred_te)

    def result(self):
        """ make result file.
        """
        y_pred = self.y_pred_te

        sents = metrics.convertv2s(y_pred)

        wh_sents = []
        for sent in sents:
            for morph in sent:
                wh_sents.append(morph)
            wh_sents.append(np.nan)  # end of sentence

        pred_TAG = {'pred_TAG': wh_sents}  # predicted NE tags
        pred_TAG = pd.DataFrame(pred_TAG)

        org = ner_data_utils._read_as_df(config._NER_SENT_TEST_REAL_, config._NAMES_TEST_)

        res = pd.merge(org, pred_TAG, left_index=True, right_index=True)

        res.to_csv(config.result, index=False, sep='\t', encoding='cp949')


if __name__ == "__main__":
    # [ for test ] #
    NERTagger = NERTagger()

    # print("train(main)")
    # NERTagger.train()
    print("test(main)")
    NERTagger.test()
    print("metrics(main)")
    NERTagger.metrics()

    print("result")
    NERTagger.result()

