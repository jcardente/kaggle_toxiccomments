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
    
    print('Building dictionary...')
    train_docs = [c.split(' ') for c in train_comments]
    test_docs  = [c.split(' ') for c in test_comments]
    docs = train_docs + test_docs
    comments_dictionary = Dictionary(docs)
    comments_dictionary.filter_extremes(no_below=10, no_above=0.5)
    comments_dictionary.compactify()

    print("Creating tfidf model...")
    train_corpus = [comments_dictionary.doc2bow(d) for d in train_docs]    
    model_tfidf  = TfidfModel(train_corpus)

    print("Converting to tfidf vectors...")
    train_tfidf  = model_tfidf[train_corpus]
    train_vecs   = corpus2csc(train_tfidf).T
    
    print('Finding important terms...')
    labelcols = train_data.columns.tolist()[2:]
    terms = Counter()
    for l in labelcols:
        cl = train_data[l]
        model_fdr = SelectFdr(chi2, alpha=0.025)
        model_fdr.fit(train_vecs, cl)
        ids = model_fdr.get_support(indices=True)
        for i in ids:
            terms[comments_dictionary[i]] += model_fdr.scores_[i]

    print('Saving results...')
    ids   = range(len(comments_dictionary))
    vocab = pd.DataFrame()
    vocab['ids'] = ids
    vocab['tokens'] = [comments_dictionary[i] for i in ids]
    vocab['scores'] = [terms[t] if t in terms else 0 for t in vocab['tokens']]
    vocab.to_csv(FLAGS.vocabfile, index=False)
    
