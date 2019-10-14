import numpy as np
import pandas as pd
import pickle
import os
from gensim.models import Word2Vec
import embedding_utils
import config


# Path
_PATH_CORPUS_ = ".\\Corpus4model\\"  # you should've have corpus for embedding in this folder.
_PATH_SENTS_ = ".\\sentences\\"  # this folder will be generated in this program.
_NER_SENT_TRAIN_ = "exobrain_train.bio_ner"
_NER_SENT_TEST_ = "exobrain_test.bio_ner"
_NER_SENT_VALID_ = "exobrain_dev.bio_ner"
_NAMES_ = ['RAW', 'MORPH', 'POS', 'TAG']
_DICTIONARY_ = ".\\Dictionary\\ne_dict.pickle"


def _read_as_df(fname, names):
    """ Read data as pandas data frame.
    :param fname: <dtype: str> type of data set
    :param names: column names. (Training set must have at least 'MORPH' and 'TAG' columns)
    :return data: data table
    """
    data = pd.read_table(config._PATH_CORPUS_ + fname, names=names,
                         delimiter='\t', quotechar=';', skip_blank_lines=False, encoding='utf-8')
    # data = data[data.iloc[:, 0] != '_']  # except space
    data = data.reset_index(drop=True)
    # print(data)

    return data


def tokens(data):
    """ Return the type and the number of tags in the data set.
    :param data: data table
    :return tags: <dtype: list> type of NE tags
    :return n_tags: the number of NE tags in the data set
    """
    words = list(set(data["MORPH"].values))
    n_words = len(words)
    # print("number of words: ", n_words)

    tags = list(set(data["TAG"].values))  # exclude duplication
    tags.remove(np.nan)  # except 'nan' tag
    n_tags = len(tags)
    # print("number of tags: ", n_tags)

    return tags, n_tags


def tag2idx(tags):
    """ Converge NE tags to index.
    :param tags: <dtype: list> type of NE tags
    :return tag2idx: <dtype: dict>
    """
    tags.remove('O')
    tag2idx = {t: i for i, t in enumerate(tags, start=1)}
    tag2idx['O'] = 0
    print("tag2idx: ", tag2idx)
    print("Success in transfer tags to index.")
    return tag2idx


# # Embedding
def _read_data(fname):
    """ Read data file.
    :param fname: <dtype: str> type of data set
    :return ner_corpus:
    """
    with open(config._PATH_CORPUS_ + fname, 'r', encoding='utf-8-sig') as fin:
        ner_corpus = fin.readlines()
    return ner_corpus


def preproc_data(ner_corpus, dataset):
    """
    :param ner_corpus: ['RAW\tMORPH\tPOS\tTAG\n', ..., '...'] <- if dataset == "train_data", "valid_data"
                     or ['MORPH\tPOS\n', ..., '...']               <- if dataset == "test_data"
    :fout: sentences: [[('morph', 'pos', 'tag')...('morph', 'pos', 'tag')], ..., [...]]
    :return: None
    """
    sent, sentences = [], []
    for line in ner_corpus:
        line = line.strip().split()
        if len(line) == 4:  # line: ['RAW', 'MORPH', 'POS', 'TAG']
            # if line[1] == '_':  # except space
            #     continue
            mpt = tuple([line[1], line[2], line[3]])  # mpt; morpheme, pos, ne_tag
            sent.append(mpt)
        # elif len(line) == 3:  # line: ['MORPH', 'POS', 'TAG']
        #     mpt = tuple([line[0], line[1], line[2]])  # mpt; morpheme, pos, ne_tag
        #     sent.append(mpt)
        elif len(line) == 2:  # line: ['MORPH', 'POS']
            mp = tuple([line[0], line[1]])  # mpt; morpheme, pos
            sent.append(mp)

        else:  # sentence delimiter
            sentences.append(sent)
            sent = []

    with open(config._PATH_SENTS_ + dataset + "_sentences.pickle", 'wb') as fout:
        pickle.dump(sentences, fout)


def _load_sentences(dataset):
    # load pre-processed sentences
    with open(config._PATH_SENTS_ + dataset + "_sentences.pickle", "rb") as fin:
        return pickle.load(fin)


def _load_embedding_mdl():
    """
    Load embedding models.
    :return wd_emb: word embedding model
    :return pos_emb: pos embedding model
    """
    mp_emb = Word2Vec.load(embedding_utils._PATH_EMB_RESULT_ + config._EMB_MP_)
    pos_emb = Word2Vec.load(embedding_utils._PATH_EMB_RESULT_ + config._EMB_POS_)
    char_emb = Word2Vec.load(embedding_utils._PATH_EMB_RESULT_ + config._EMB_CHAR_)

    return mp_emb, pos_emb, char_emb


# Embedding input data(X, morph/tag)
def input_embedding(sentences):
    """ # of sentences
    sent_length(= number of morphemes)
    :fin: sentences: [[('morph', 'pos', 'tag')...('morph', 'pos', 'tag')], ..., [...]]
    :return input_data, sequence_lengths, emb_size:
    """
    mp_emb, pos_emb, char_emb = _load_embedding_mdl()  # load embedding models
    with open(config._DICTIONARY_, 'rb') as fin:
        ne_dict = pickle.load(fin)

    input_data, sent_lengths = [], []  # input_data: embedded sentences, sent_lengths: list of each length of sentence.
    for s in sentences:
        sent_lengths.append(len(s))
        sent = []
        for mpt in s:  # mpt: ('morpheme', 'pos', 'tag')
            morph = []  # embedding of one morpheme.

            # 1. word embedding
            try:
                morph.extend(mp_emb.wv['/'.join([mpt[0], mpt[1]])])  # mpt[0]: morph, mpt[1]: pos
            except KeyError:
                morph.extend(np.random.rand(config.mp_emb_size))

            # 2. pos embedding
            try:
                morph.extend(pos_emb.wv[mpt[1]])  # mpt[1]: pos
            except KeyError:
                morph.extend(np.random.rand(config.pos_emb_size))

            # 3. charter based word embedding


            # 4. ne dictionary feature
            try:
                df = pd.DataFrame(sorted(ne_dict[mpt[0]].items()))  # if mpt[0] is in the dictionary, make data frame by key order.
                ne_dict_feature = df[1].tolist()  # convert feature column to list, ne_dict_feature = [0, 0,...,1, 0]
                morph.extend(ne_dict_feature)
            except KeyError:
                morph.extend([0 for i in range(10)])  # if mpt[0] is not in the dictionary, the features will be [0, ..., 0]

            sent.append(morph)
        input_data.append(sent)

#                                          # morph                      # sent                                 # sents
    input_data = np.array(input_data)  # [[[emb, emb, ..., emb], ..., []], ..., [[emb, emb, ..., emb], ..., []]]
    emb_size = config.mp_emb_size + config.pos_emb_size + 10  # need revise
    # n_sent = len(sentences)  # The number of sentences.
    sequence_lengths = np.array(sent_lengths)  # [length, length, ..., length]

    return input_data, sequence_lengths, emb_size  #, n_sent


# Embedding target data(y, label)
def label_embedding(sentences):
    with open("tag_enc.pickle", 'rb') as fin:
        tag_enc = pickle.load(fin)

    label_data = [[tag_enc[w[2]] if w[2] in tag_enc else -1 for w in s] for s in sentences]
#                                         # sent                       # sents
    label_data = np.array(label_data)  # [[idx, idx, ..., idx], ..., []]

    return label_data


def data_set(dataset):
    if dataset == "train_data":
        fname = config._NER_SENT_TRAIN_
    if dataset == "test_data":
        fname = config._NER_SENT_TEST_
    if dataset == "test_data_real":
        fname = config._NER_SENT_TEST_REAL_
    if dataset == "valid_data":
        fname = config._NER_SENT_VALID_

    # pre-processed sentence
    if not os.path.exists(config._PATH_SENTS_ + dataset + "_sentences.pickle"):
        if not os.path.exists(config._PATH_SENTS_):
            os.makedirs(config._PATH_SENTS_)
        print("There is no processed sentences. Start processing sentences.")
        preproc_data(_read_data(fname), dataset)
    else:
        print("Processed sentences exists. Use this sentences in embedding")

    sentences = _load_sentences(dataset)
    if not dataset == "test_data_real":
        tags, n_tags = tokens(_read_as_df(fname, config._NAMES_))

    # embedded tag dictionary
    if not os.path.exists("tag_enc.pickle"):  # run only once when dataset is "train_data"
        print("# tag_enc: File doesn't exists. Start to transfer tags to index.")
        with open("tag_enc.pickle", 'wb') as fout:
            pickle.dump(tag2idx(tags), fout)  # tag2idx(tags) = {'tag': idx, 'tag': idx, ..., 'tag': idx}
    else:
        with open("tag_enc.pickle", 'rb') as fin:
            tag_enc = pickle.load(fin)
        n_tags = len(tag_enc)
        # print("tag2idx(tag_enc.pickle): ", tag_enc)
        # print("# tag_enc:↑File already exists. Use this tag in labeling.")

    # Embedding
    input_data, sequence_lengths, emb_size = input_embedding(sentences)

    # print("[type] input_data:", type(input_data), ", label_data: ", type(label_data))
    print("<shape>\ninput_data: ", np.shape(input_data))
    if not dataset == "test_data_real":
        label_data = label_embedding(sentences)
        print("label_data: ", np.shape(label_data))
        return input_data, label_data, sequence_lengths, emb_size, n_tags
    else:
        return input_data, sequence_lengths, emb_size, n_tags


if __name__ == "__main__":
    # [ for test ] #
    data_set("train_data")


