# ------------------------------------------------------------
# learn_important_unknown.py
#
# Learns tokens that appear to be informative for classification
# but don't have an embedding
#
# J. Cardente
# 2018
# ------------------------------------------------------------

import argparse
import sys
import pandas as pd
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
                        help='Train file')

    parser.add_argument('-o', type=str,
                        required=True,
                        dest='outfile',
                        help='Output file')
    
    parser.add_argument('-e',type=str,
                        required=True,
                        dest='embedfile',
                        help='Embedding file')
        
    FLAGS, unparsed = parser.parse_known_args()

    print('Reading data...')
    train_data = load_data(FLAGS.trainfile)
    comments_text = train_data['comment_text']
    comments_text = comments_text.tolist()

    print('Finding unknown tokens...')
    ft_model = load_embedding(FLAGS.embedfile)
    docs = [c.split(' ') for c in comments_text]
    for i in range(len(docs)):
        docs[i] = [t for t in docs[i] if not t in ft_model.vocab]
        
    print('Building dictionary...')
    comments_dictionary = Dictionary(docs)
    comments_corpus     = [comments_dictionary.doc2bow(d) for d in docs]

    print("Creating tfidf model...")        
    model_tfidf     = TfidfModel(comments_corpus)

    print("Converting to tfidf vectors...")
    comments_tfidf  = model_tfidf[comments_corpus]
    comments_vecs   = corpus2csc(comments_tfidf).T

    print('Finding important terms...')
    labelcols = train_data.columns.tolist()[2:]
    unknowns = Counter()
    for l in labelcols:
        cl = train_data[l]
        model_fdr = SelectFdr(chi2, alpha=0.025)
        model_fdr.fit(comments_vecs, cl)
        ids = model_fdr.get_support(indices=True)
        for i in ids:
            unknowns[i] += model_fdr.scores_[i]

    print('Saving results...')
    unknowns_ranked = [(i, s, comments_dictionary.dfs[i]) for i,s in unknowns.items()]
    unknowns_ranked.sort(key=lambda x: x[2], reverse=True)
    unknowns_dict = {
        'scores': [s for i,s,c in unknowns_ranked],
        'ndocs' : [c for i,s,c in unknowns_ranked],
        'tokens': [comments_dictionary[i] for i,s,c in unknowns_ranked]
        }
    
    df = pd.DataFrame(unknowns_dict)
    df.to_csv(FLAGS.outfile, index=False)
