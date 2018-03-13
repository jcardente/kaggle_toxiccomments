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
from sklearn.feature_selection import chi2
from util import load_data


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

    train_data['any'] =  np.max(train_data.iloc[:,2:], axis=1)    
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
    tokens_df = pd.DataFrame([all_dictionary[i] for i in range(nterms)], columns=['token'])
    
    print("Creating tfidf models...")
    train_corpus = [all_dictionary.doc2bow(d) for d in train_docs]
    test_corpus  = [all_dictionary.doc2bow(d) for d in test_docs]
    
    train_tfidf_model  = TfidfModel(train_corpus, dictionary=all_dictionary)
    test_tfidf_model   = TfidfModel(test_corpus,  dictionary=all_dictionary)

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

    chi2_scores = np.log(chi2_scores)
    chi2_scores -= np.min(chi2_scores)
    chi2_scores /= np.max(chi2_scores)
    chi2_df = pd.DataFrame(data=chi2_scores, columns=['chi2_' + l for l in labelcols])

    
    print('Scoring train terms using IDF delta...')
    idfdelta_scores = np.zeros((nterms, nlabels), dtype=np.float)
    for i,l in enumerate(labelcols):
        cl = train_data[l]

        pos = train_vecs[np.where(cl==1)[0],:]
        neg = train_vecs[np.where(cl==0)[0],:]

        pos_df = np.squeeze(np.array((pos > 0.0).sum(0)))
        neg_df = np.squeeze(np.array((neg > 0.0).sum(0)))

        npos = pos.shape[0]
        nneg = neg.shape[0]

        v1 = np.multiply((npos - pos_df + 0.5), (neg_df + 0.5))
        v2 = np.multiply((nneg - neg_df + 0.5), (pos_df + 0.5))

        idfdelta_scores[:,i]  = np.log(np.divide(v2,v1))

    idfdelta_scores -= np.min(idfdelta_scores)
    idfdelta_scores /= np.max(idfdelta_scores)
    idfdelta_df = pd.DataFrame(data=idfdelta_scores, columns=['idfdelta_' + l for l in labelcols])

    print('Saving results...')
    all_df = pd.concat([tokens_df, chi2_df, idfdelta_df], axis=1)
    all_df.index.name = 'id'
    all_df.to_csv(FLAGS.vocabfile, index=True)
    
