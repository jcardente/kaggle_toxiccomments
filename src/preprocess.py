# ------------------------------------------------------------
# preprocess.py
#
# Utility to lemmatize comment text and saves result to a new csv
# file. Only keeps lemmas that have a spaCy embedding vector.
#
# J. Cardente
# 2018
# ------------------------------------------------------------

import argparse
import pandas as pd
import spacy
from   util import lematize_comments, load_nlp

FLAGS = None


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
    nlp  = load_nlp()
    docs = lematize_comments(comments_text, nlp, nthreads=FLAGS.nthreads)

    new_comments = [' '.join(d) for d in docs]
    data['comment_text'] = new_comments

    print('Saving results...')    
    data.to_csv(FLAGS.outfile, index=False)
    
    
