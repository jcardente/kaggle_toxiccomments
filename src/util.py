# ------------------------------------------------------------
# util.py
#
# Common code for Kaggle Toxic Comment Challenge
#
# J. Cardente
# 2018
# ------------------------------------------------------------
from gensim.models import KeyedVectors
import pandas as pd
import spacy
import pickle

def keep_token(t):
    return not (t.is_space or t.is_punct or 
                t.is_stop or t.like_num or
                t.like_url or t.like_email)

def lematize_comment(comment):
    return [ t.lemma_ for t in comment if keep_token(t)]


def lematize_comments(comments, nlp, nthreads=4, batch_size=1000):
    docs = []
    for c in nlp.pipe(comments, batch_size=batch_size, n_threads=nthreads):
        lc = lematize_comment(c)
        docs.append(lc)
    return docs


def load_nlp():
     return spacy.load('en_core_web_md', disable=['ner','parser'])

def load_data(fname):
    data = pd.read_csv(fname)
    data['comment_text'].fillna('', inplace=True)
    return data


def load_fasttext(fname):
    # NB - this loads the FastText embeddings downloaded from
    #      the FastText website. It's in an text format that
    #      is slow to load. Use the binary version converted
    #      to Gensim KeyedVectors format.
    ft_model = KeyedVectors.load_word2vec_format(fname)
    return ft_model


def load_embedding(fname):
    ft_mode = KeyedVectors.load(fname)
    return ft_mode


def load_chi2(fname):
    f = open(fname, 'rb')
    term_scores = pickle.load(f)
    return term_scores
