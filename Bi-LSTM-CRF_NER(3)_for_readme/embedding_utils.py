from gensim.models import Word2Vec
import pandas as pd
import pickle
import re
import os
import config

"""
On this program, you can get 'embedding models'.

< Explanation about folders >
- Corpus4embedding: having raw corpus.          <= _PATH_EMB_CORPUS_
- embeddings: saving embedding results here.    <= _PATH_EMB_RESULT_
"""

# Path
_PATH_EMB_CORPUS_ = ".\\Corpus4embedding\\"  # you should've have 'corpus for embedding' in this folder.
_PATH_EMB_RESULT_ = ".\\embeddings\\"  # embedding results. (generated in this program) -> offered
_PATH_DICTIONARY_ = ".\\Dictionary\\"  # you should've have 'Named Entity dictionary' in this folder.
_EMB_MP_ = "morph_pos_emb.model"
_EMB_POS_ = "pos_emb.model"
_EMB_CHAR_ = "char_emb.model"
_DICTIONARY_ = "ne_dictionary.dict"  # Named Entity dictionary

# Hyper-parameter for Word Embedding
mp_emb_size = 64  # 'morph/pos' embedding size
pos_emb_size = 16  # 'pos' embedding size
window = 5
cbow = 0
iteration = 15
min_count = 0


# # Pre-processing
def preproc4file():
    """
    fin: emb_corpus: 'morph/pos morph/pos ... morph/pos'
    fout: sentences: [[(word, pos), (word, pos), ... (word, pos)][]...[]]
    """

    # add checking file existence part.
    with open(_PATH_EMB_CORPUS_ + "emb_corpus.txt", 'r', encoding='utf-8') as fin:
        emb_corpus = fin.readlines()

    sentences = []
    for line in emb_corpus:
        if len(line) < 2000:    # pruning long sentences(for preventing memory error).
                                # ↑when the system is stable, try to use longer sentences.
            sent = []
            line = line.split(' ')
            for token in line:
                token = token.split('/')  # token[0]: morph, token[1]: pos
                num = re.search(r"(_+\d+)", token[0])  # search shoulder number
                pos = token[1]
                if num:
                    morph = token[0].replace(num.group(), "")  # excepting shoulder number
                else:
                    morph = token[0]

                sent.append((morph, pos))
            sentences.append(sent)

    with open(_PATH_EMB_CORPUS_ + "emb_corpus_prun.pickle", 'wb') as fout:
        pickle.dump(sentences, fout)


def preproc4emb():
    """
    :param sentences: <dtype: list>, 2D
                        [[(word, pos), (word, pos), ... (word, pos)][]...[]]
    :return morph_pos_sents: [['word/pos', 'word/pos', ..., 'word/pos'], ..., [...]]
    :return pos_sents: [['pos', 'pos', ..., 'pos'], ..., []]
    """
    with open(_PATH_EMB_CORPUS_ + "emb_corpus_prun.pickle", "rb") as fin:
        sentences = pickle.load(fin)

    morph_pos_sents, pos_sents, char_sents = [], [], []

    for s in sentences:  # for s in sentences:
        wd_sent, pos_sent, char_sent = [], [], []
        for w in s:
            wd = '/'.join([w[0], w[1]])  # w[0]: word, w[1]: pos => wd = 'word/pos'
            wd_sent.append(wd)
            pos = w[1]  # Where occur memory error
            pos_sent.append(pos)
            for c in w[0]:
                char_sent.append(c)
        morph_pos_sents.append(wd_sent)
        pos_sents.append(pos_sent)
        char_sents.append(char_sent)

    print("End pre-processing for embedding.")

    return morph_pos_sents, pos_sents, char_sents


# # Embedding
# 1. word embedding
def morph_pos_emb(morph_pos_sents):
    if os.path.exists(_PATH_EMB_RESULT_ + _EMB_MP_):
        print("# morph_pos_emb.model: File already exists.")
    else:
        print("# morph_pos_emb.model: File doesn't exists. Start Embedding.")
        model = Word2Vec(morph_pos_sents, size=config.mp_emb_size, window=config.window, sg=config.cbow,
                         iter=config.iteration, min_count=config.min_count)
        model.init_sims(replace=True)  # unload useless memory
        model.save(_PATH_EMB_RESULT_ + _EMB_MP_)
        print("Embedding success(morph_pos_emb).")


# 2. pos embedding
def pos_emb(pos_sents):
    if os.path.exists(_PATH_EMB_RESULT_ + _EMB_POS_):
        print("# pos_emb.model: File already exists.")
    else:
        print("# pos_emb.model: File doesn't exists. Start Embedding.")
        model = Word2Vec(pos_sents, size=config.pos_emb_size, window=config.window, sg=config.cbow,
                         iter=config.iteration, min_count=config.min_count)
        model.init_sims(replace=True)
        model.save(_PATH_EMB_RESULT_ + _EMB_POS_)
        print("Embedding success(pos_emb.model).")


# 3. char embedding; for charter based word embedding
def char_emb(char_sents):
    if os.path.exists(_PATH_EMB_RESULT_ + _EMB_CHAR_):
        print("# char_emb.model: File already exists.")
    else:
        print("# char_emb.model: File doesn't exists. Start Embedding.")
        model = Word2Vec(char_sents, size=config.mp_emb_size, window=config.window, sg=config.cbow,
                         iter=config.iteration, min_count=config.min_count)
        model.init_sims(replace=True)
        model.save(_PATH_EMB_RESULT_ + _EMB_CHAR_)
        print("Embedding success(char_emb.model).")


# 4. ne dictionary feature
def _init_feature_vector():
    tags = ['DAT', 'DUR', 'LOC', 'MNY', 'NOH', 'ORG', 'PER', 'PNT', 'POH', 'TIM']
    feature_vec = {}
    for tag in tags:
        feature_vec[tag] = 0  # feature_vec = {'DAT': 0, 'DUR': 0, ..., 'TIM':0}, total 10 tags.
    return feature_vec


def dict_feature():
    if os.path.exists(_PATH_DICTIONARY_ + "ne_dict.pickle"):
        print("# ne_dict.pickle: File already exists.")
    else:
        print("# ne_dict.pickle: File doesn't exists. Start  ")
        ne_dictionary = pd.read_csv(_PATH_DICTIONARY_ + _DICTIONARY_, 'r', names=['WORD', 'TAG'], delimiter='\t', encoding='utf-8')
        length = len(ne_dictionary["WORD"])

        dictionary = {}
        for i in range(length):
            key = ne_dictionary.iloc[i][0]  # word
            value = ne_dictionary.iloc[i][1]  # ne_tag
            try:
                dictionary[key][value] += 1
            except KeyError:
                dictionary[key] = _init_feature_vector()
                dictionary[key][value] += 1

        with open(_PATH_DICTIONARY_ + "ne_dict.pickle", 'wb') as fout:
            pickle.dump(dictionary, fout)

        print("Making dictionary success. (ne_dict.pickle).")


def mk_embedding_mdl():
    """
    Consists of 2 parts; 'Pre-processing' and 'Embedding'
    [ Pre-processing ]
    #
    - preproc4file:
    - preproc4emb:

    [ Embedding ]
    #
    - morph_pos_emb:
    - pos_emb:
    """
    # Pre-processing
    if not os.path.exists(_PATH_EMB_CORPUS_ + "emb_corpus_prun.pickle"):
        preproc4file()
    else:
        print("emb_corpus_prun.pickle: File already exists.")

    morph_pos_sents, pos_sents, char_sents = preproc4emb()

    # Embedding
    if not os.path.exists(_PATH_EMB_RESULT_):
        os.makedirs(_PATH_EMB_RESULT_)  # Folder for saving embedding results.
    morph_pos_emb(morph_pos_sents)
    pos_emb(pos_sents)
    char_emb(char_sents)
    dict_feature()


if __name__ == "__main__":
    # [ for test ] #
    mk_embedding_mdl()