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
import numpy as np
from gensim.corpora import Dictionary
from gensim.models.tfidfmodel import TfidfModel
from gensim.matutils import corpus2csc
from sklearn.feature_selection import chi2, SelectFdr
from util import load_data, load_embedding, Vocab
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
                
    parser.add_argument('-e',type=str,
                        required=True,
                        dest='embedfile',
                        help='Embedding file')

    parser.add_argument('-o', type=str,
                        required=True,
                        dest='outfile',
                        help='Output file')
    
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

    print('Loading vocab and embeddings...')
    vocab = Vocab(FLAGS.vocabfile)
    vocab.add_embeddings(FLAGS.embedfile)

    token_ids = set(vocab.vdf['id'].tolist())

    unknown_ids = token_ids.difference(vocab.hasEmbedding)
    unknown_scored = [(i,vocab.id2quickscore(i)) for i in unknown_ids]

    unknown_scored.sort(key=lambda x: x[1], reverse=True)

    unknown_scored = [(vocab.id2token(i), s) for i,s in unknown_scored]

    print('Saving results...')
    unknown_dict = {
        'score': [s for t,s in unknown_scored],
        'token': [t for t,s in unknown_scored],
        }
    
    df = pd.DataFrame(unknown_dict)
    df.to_csv(FLAGS.outfile, index=False)
    
