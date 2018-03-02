# ------------------------------------------------------------
# learn_vocab.py
#
# Learn the vocabulary for both the test and train data sets.
#
# J. Cardente
# 2018
# ------------------------------------------------------------

import argparse
import pandas as pd
import pickle
import spacy
from gensim.corpora import Dictionary
 
FLAGS = None


def keep_token(t):
    return t.has_vector and not (t.is_space or t.is_punct or 
                 t.is_stop or t.like_num)

def lematize_comment(comment):
    return [ t.lemma_ for t in comment if keep_token(t)]
            

def lematize_comments(comments, nlp, nthreads=4):
    docs = []
    for c in nlp.pipe(comments, batch_size=100, n_threads=nthreads):
        lc = lematize_comment(c)
        if len(lc) == 0:
            lc =['--NONE--']
        docs.append(lc)
    return docs



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

    parser.add_argument('-d',type=str,
                        dest='dictfile',
                        default='models/vocab.dat')
    
    parser.add_argument('-n',type=int,
                        default=4,
                        dest='nthreads',
                        help='Number of threads')
    
    FLAGS, unparsed = parser.parse_known_args()



    print('Reading data...')
    train_data = pd.read_csv(FLAGS.trainfile)
    test_data  = pd.read_csv(FLAGS.testfile)
    comments_text = pd.concat([train_data['comment_text'], test_data['comment_text']], axis=0)
    
    print('Lematizing...')
    nlp = spacy.load('en_core_web_md', disable=['parser'])
    docs = lematize_comments(comments_text, nlp, nthreads=FLAGS.nthreads)

    print('Building dictionary...')
    comments_dictionary = Dictionary(docs)
    comments_dictionary.filter_extremes(no_below=10, no_above=0.3)
    comments_dictionary.compactify()

    print('Saving dictionary...')
    comments_dictionary.save(FLAGS.dictfile)    
