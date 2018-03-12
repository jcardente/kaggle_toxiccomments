# ------------------------------------------------------------
# learn_vocab.py
#
# Learn the vocabulary for both the test and train data sets.
# Also learns informative terms from the training set
#
# J. Cardente
# 2018
# ------------------------------------------------------------

import argparse
import pandas as pd
import numpy as np
from gensim.corpora import Dictionary
from gensim.models.tfidfmodel import TfidfModel
from gensim.matutils import corpus2csc
from sklearn.feature_selection import chi2, SelectFdr
from util import load_data
from collections import Counter

FLAGS = None

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--train',type=str,
                        required=True,
                        dest='trainfile',
                        help='Train file')

    parser.add_argument('--test',type=str,
                        required=True,
                        dest='testfile',
                        help='Test file')
    
    parser.add_argument('-v',type=str,
                        required=True,
                        dest='vocabfile',
                        help='vocab file')
            
    FLAGS, unparsed = parser.parse_known_args()


    print('Reading data...')
    train_data = load_data(FLAGS.trainfile)
    test_data  = load_data(FLAGS.testfile)
    train_comments = train_data['comment_text'].tolist()
    test_comments  = test_data['comment_text'].tolist()
    labelcols      = train_data.columns.tolist()[2:]    

    ntrain  = len(train_comments)
    ntest   = len(test_comments)
    nlabels = len(labelcols)

    
    print('Building dictionary...')
    train_docs = [c.split(' ') for c in train_comments]
    test_docs  = [c.split(' ') for c in test_comments]
    all_docs = train_docs + test_docs
    all_dictionary = Dictionary(all_docs)
    all_dictionary.filter_extremes(no_below=10, no_above=0.5)
    all_dictionary.compactify()
    nterms = len(all_dictionary)

    print("Creating tfidf models...")
    train_corpus = [all_dictionary.doc2bow(d) for d in train_docs]
    test_corpus  = [all_dictionary.doc2bow(d) for d in test_docs]
    
    train_tfidf_model  = TfidfModel(train_corpus, dictionary=all_dictionary)
    test_tfidf_model   = TfidfModel(test_corpus,  dictionary=all_dictionary)

    print("Converting to tfidf vectors...")
    train_tfidf  = train_tfidf_model[train_corpus]
    train_vecs   = corpus2csc(train_tfidf, num_terms=nterms).T

    
    print('Scoring train terms using Chi2...')
    chi2_scores = np.zeros((nterms, nlabels), dtype=np.float)
    for i,l in enumerate(labelcols):
        cl = train_data[l]
        scores, _ = chi2(train_vecs, cl)
        chi2_scores[:,i] = scores
    np.nan_to_num(chi2_scores,copy=False)
    chi2_min = np.min(chi2_scores[np.nonzero(chi2_scores)])
    chi2_scores[np.where(chi2_scores == 0)] = chi2_min
    

    print('Scoring train terms using IDF delta...')
    idfdelta_scores = np.zeros((nterms, nlabels), dtype=np.float)

    for i,l in enumerate(labelcols):
        cl = train_data[l]

        pos = train_vecs[np.where(cl==1)[0],:]
        neg = train_vecs[np.where(cl==0)[0],:]

        pos_df = (pos > 0.0).sum(0)
        neg_df = (neg > 0.0).sum(0)

        npos = pos.shape[0]
        nneg = neg.shape[0]

        v1 = np.multiply((npos - pos_df + 0.5), (neg_df + 0.5))
        v2 = np.multiply((nneg - neg_df + 0.5), (pos_df + 0.5))
        
        idf_delta = 
        
        model_fdr = SelectFdr(chi2, alpha=0.025)
        model_fdr.fit(train_vecs, cl)
        ids = model_fdr.get_support(indices=True)
        for i in ids:
            terms[all_dictionary[i]] += model_fdr.scores_[i]

    print('Scoring terms using tfidf difference...')
            
    print('Saving results...')
    ids   = range(len(all_dictionary))
    vocab = pd.DataFrame()
    vocab['ids'] = ids
    vocab['tokens'] = [all_dictionary[i] for i in ids]
    vocab['scores'] = [terms[t] if t in terms else 0 for t in vocab['tokens']]
    vocab.to_csv(FLAGS.vocabfile, index=False)
    
