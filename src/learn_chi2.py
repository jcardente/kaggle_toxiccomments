# ------------------------------------------------------------
# learn_chi2.py
#
# Utility to score terms based on a Chi2 fit against labels. 
#
# J. Cardente
# 2018
# ------------------------------------------------------------

import argparse
import pandas as pd
import pickle

from gensim.corpora import Dictionary
from gensim.models.tfidfmodel import TfidfModel
from gensim.matutils import corpus2csc
from sklearn.feature_selection import chi2, SelectFdr
from util import load_data, load_embedding
from collections import Counter

FLAGS = None


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-t',type=str,
                        required=True,
                        dest='trainfile',
                        help='Training file')

    parser.add_argument('-e',type=str,
                        required=True,
                        dest='embedfile',
                        help='Embedding file')
    
    
    parser.add_argument('-c',type=str,
                        dest='chi2file',
                        default='models/chi2scores.pkl')
    
    FLAGS, unparsed = parser.parse_known_args()

    print('Reading data...')
    data = load_data(FLAGS.trainfile)
    comments_text = data['comment_text']
    comments_text = comments_text.tolist()

    print('Finding tokens with embeddings...')
    ft_model = load_embedding(FLAGS.embedfile)
    docs = [c.split(' ') for c in comments_text]
    for i in range(len(docs)):
        docs[i] = [t for t in docs[i] if t in ft_model.vocab]
        
    print('Building dictionary...')
    comments_dictionary = Dictionary(docs)
    comments_corpus     = [comments_dictionary.doc2bow(d) for d in docs]

    print("Creating tfidf model...")        
    model_tfidf     = TfidfModel(comments_corpus)

    print("Converting to tfidf vectors...")
    comments_tfidf  = model_tfidf[comments_corpus]
    comments_vecs   = corpus2csc(comments_tfidf).T

    print('Finding important terms...')
    labelcols = data.columns.tolist()[2:]
    terms = Counter()
    for l in labelcols:
        cl = data[l]
        model_fdr = SelectFdr(chi2, alpha=0.025)
        model_fdr.fit(comments_vecs, cl)
        ids = model_fdr.get_support(indices=True)
        for i in ids:
            terms[i] += model_fdr.scores_[i]

    print('Saving results...')
    with open(FLAGS.chi2file, 'wb') as f:
        pickle.dump(terms, f, protocol=pickle.HIGHEST_PROTOCOL)
    
            
