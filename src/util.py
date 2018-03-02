# ------------------------------------------------------------
# util.py
#
# Common code for Kaggle Toxic Comment Challenge
#
# J. Cardente
# 2018
# ------------------------------------------------------------

import spacy
 
def keep_token(t):
    return t.has_vector and not (t.is_space or t.is_punct or 
                                 t.is_stop or t.like_num)


def lematize_comment(comment):
    return [ t.lemma_ for t in comment if keep_token(t)]


def lematize_comments(comments, nlp, nthreads=4, batch_size=1000):
    docs = []
    for c in nlp.pipe(comments, batch_size=batch_size, n_threads=nthreads):
        lc = lematize_comment(c)
        docs.append(lc)
    return docs

def load_nlp():
     return spacy.load('en_core_web_md', disable=['parser', 'ner', 'tagger'])

