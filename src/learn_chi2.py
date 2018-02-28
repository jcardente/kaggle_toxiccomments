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
from sklearn.feature_selection import chi2
 
FLAGS = None


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-t',type=str,
                        required=True,
                        dest='trainfile',
                        help='Training file')

    parser.add_argument('-c',type=str,
                        dest='chi2file',
                        default='models/chi2scores.pkl')
    
    FLAGS, unparsed = parser.parse_known_args()

    print('Reading data...')
    data = pd.read_csv(FLAGS.trainfile)

    labelColnames =  data.columns.tolist()[2:]        
    labels   = data[labelColnames].apply(lambda x: int(any(x)), axis=1)
        
    comments_text = data['comment_text']
    docs = [c.split(' ') for c in comments_text]

    print('Building dictionary...')
    comments_dictionary = Dictionary(docs)
    comments_corpus     = [comments_dictionary.doc2bow(d) for d in docs]

    print("Creating tfidf model...")        
    model_tfidf     = TfidfModel(comments_corpus)

    print("Converting to tfidf vectors...")
    comments_tfidf  = model_tfidf[comments_corpus]
    comments_vecs   = corpus2csc(comments_tfidf).T

    print('Calculating Chi2 scores...')
    chivals, pvals = chi2(comments_vecs, labels)
    term_scores    = {t:chivals[i] for i,t in comments_dictionary.iteritems()}

    print('Saving chi2 scores...')
    with open(FLAGS.chi2file, 'wb') as f:
        pickle.dump(term_scores, f, protocol=pickle.HIGHEST_PROTOCOL)
            
