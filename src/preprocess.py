# ------------------------------------------------------------
# preprocess.py
#
# Utility to lemmatize and detect phrases in comment text.
# Saves result to a new csv file.
#
# J. Cardente
# 2018
# ------------------------------------------------------------

import argparse
import numpy as np
import pandas as pd
import spacy

from gensim.models.phrases import Phrases, Phraser

FLAGS = None

 
def keep_token(t):
    return not (t.is_space or t.is_punct or 
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


if  __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i',type=str,
                        required=True,
                        dest='infile',
                        help='Input file to process')
    
    parser.add_argument('-o',type=str,
                        required=True,
                        dest='outfile',
                        help='Output file')
    
    parser.add_argument('-n',type=int,
                        default=4,
                        dest='nthreads',
                        help='Number of threads')

    FLAGS, unparsed = parser.parse_known_args()

    print('Reading data...')
    data = pd.read_csv(FLAGS.infile)

    comments_text = data['comment_text']

    print('Lemmatizing...')
    nlp = spacy.load('en', disable=['ner'])
    docs = lematize_comments(comments_text, nlp, nthreads=FLAGS.nthreads)

    bigram_transformer = Phraser(Phrases(docs))
    new_comments = [' '.join(d) for d in bigram_transformer[docs]]
    data['comment_text'] = new_comments

    print('Saving results...')    
    data.to_csv(FLAGS.outfile, index=False)
    
    
