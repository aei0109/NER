import numpy as np
from keras.preprocessing.sequence import pad_sequences


def padding(batch_sents, max_len):
    """
    Padding with '0' to make all sentences in a batch to have same length.
    :param batch_sents: sentences in a batch.
    :param max_len: the length of the longest sentence in a batch.
    :return padded_batch_sent: padded sentences.
    """
    padded_batch_sent = pad_sequences(sequences=batch_sents, maxlen=max_len, padding="post", value=0)

    return padded_batch_sent


def pad_x(sentence, max_length, input_size):
    pad_len = max_length - len(sentence)
    if pad_len <= 0: return sentence
    padding = [np.zeros(input_size) for _ in range(pad_len)]
    return np.concatenate((sentence, padding))


def mini_batch(X, y, input_size, whole_seq_len, batch_size):
    """
    mini batch
    :param X: <dtype: numpy.ndarray> input
    :param y: <dtype: numpy.ndarray> output. (not use in testing)
    :param input_size: <dtype: integer> embedding size
    :param whole_seq_len: <dtype: list> original sequence length of whole sentences.
    :param batch_size: hyper-parameter.
    :var len(X): the number of whole sentences.
    :var n_batch: the number of mini batches.
    :returns X_batch, y_batch: padded input, output batch sentences.
    :return batch_max_len: the length of the longest sentence in a batch.
    :return batch_seq_len: <dtype: list> the list of the length of sentences in a batch.
    """
    n_batch = int(np.ceil(len(X) / batch_size))
    idx = 0
    for i in range(n_batch):
        batch_seq_len = whole_seq_len[idx:idx + batch_size]
        X_batch_sents = X[idx:idx + batch_size]
        y_batch_sents = y[idx:idx + batch_size]

        batch_max_len = max([len(sent) for sent in X_batch_sents])
        # X_batch = padding(X_batch_sents, batch_max_len)
        X_batch= [pad_x(X_batch, batch_max_len, input_size)for X_batch in X_batch_sents]
        y_batch = padding(y_batch_sents, batch_max_len)

        idx += batch_size

        yield X_batch, y_batch, batch_max_len, batch_seq_len


def mini_batch_te(X, input_size, whole_seq_len, batch_size):
    """
    mini batch
    :param X: <dtype: numpy.ndarray> input
    :param y: <dtype: numpy.ndarray> output. (not use in testing)
    :param input_size: <dtype: integer> embedding size
    :param whole_seq_len: <dtype: list> original sequence length of whole sentences.
    :param batch_size: hyper-parameter.
    :var len(X): the number of whole sentences.
    :var n_batch: the number of mini batches.
    :returns X_batch, y_batch: padded input, output batch sentences.
    :return batch_max_len: the length of the longest sentence in a batch.
    :return batch_seq_len: <dtype: list> the list of the length of sentences in a batch.
    """
    n_batch = int(np.ceil(len(X) / batch_size))
    idx = 0
    for i in range(n_batch):
        batch_seq_len = whole_seq_len[idx:idx + batch_size]
        X_batch_sents = X[idx:idx + batch_size]

        batch_max_len = max([len(sent) for sent in X_batch_sents])
        # X_batch = padding(X_batch_sents, batch_max_len)
        X_batch= [pad_x(X_batch, batch_max_len, input_size)for X_batch in X_batch_sents]

        idx += batch_size

        yield X_batch, batch_max_len, batch_seq_len

