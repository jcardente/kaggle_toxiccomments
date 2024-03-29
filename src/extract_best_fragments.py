# ------------------------------------------------------------
# extract_best_fragments.py
#
# Utility to pick most informative text fragments
# based on a Chi2 fit against labels. 
#
# J. Cardente
# 2018
# ------------------------------------------------------------

import argparse
import numpy as np
import pandas as pd
import pickle

from gensim.corpora import Dictionary
from gensim.models.tfidfmodel import TfidfModel
from gensim.matutils import corpus2csc
from sklearn.feature_selection import chi2
 
FLAGS = None


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-i',type=str,
                        required=True,
                        dest='infile',
                        help='Input file to convert')
    parser.add_argument('-o',type=str,
                        required=True,
                        dest='outfile',
                        help='Output file to store results')
    parser.add_argument('--maxwords',type=int,
                        default=50,
                        dest='maxwords',
                        help='Maximum number of words')
    parser.add_argument('--windowsize',type=int,
                        default=5,
                        dest='windowsize',
                        help='Size of window fragments (one side)')
    
    parser.add_argument('-c',type=str,
                        dest='chi2file',
                        default='models/chi2scores.pkl')
    
    FLAGS, unparsed = parser.parse_known_args()

    print('Reading data...')
    data = pd.read_csv(FLAGS.infile)

    labelColnames =  data.columns.tolist()[2:]        
    labels   = data[labelColnames].apply(lambda x: int(any(x)), axis=1)
        
    comments_text = data['comment_text']
    docs = [c.split(' ') for c in comments_text]

    with open(FLAGS.chi2file, 'rb') as f:
        term_scores = pickle.load(f)
        
        
    print('Extracting fragments...')
    new_comments = []
    for d in docs:
        new_d = []        
        if len(d) <= FLAGS.maxwords:
            new_d = d
        else:
            d_chivals = [(i,term_scores[t]) if t in term_scores else (i,0) for i,t in enumerate(d)]
            d_chivals.sort(reverse=True, key=lambda x: x[1])

            extents = [(max(0,i-FLAGS.windowsize), min(len(d)-1,i+FLAGS.windowsize)) for i,_ in d_chivals]
            keep = set()
            while len(extents) > 0:
               ext  = extents.pop(0)
               idxs = range(ext[0],ext[1]+1)
               if (len(idxs) + len(keep)) > FLAGS.maxwords:
                   break
               keep.update(idxs)

            new_d = [d[i] for i in sorted(keep)]
                
        new_comments.append(' '.join(new_d))


    data['comment_text'] = new_comments

    print('Saving results...')    
    data.to_csv(FLAGS.outfile, index=False)
            
