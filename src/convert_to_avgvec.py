# ------------------------------------------------------------
# convert_to_avgvec.py
#
# Utility to convert input data for the Kaggle Toxic Comment
# challenge into a TF-IDF weighted average word2vec vector.
#
# Also does feature selection using Chi2 criteria
#
# J. Cardente
# 2018
# ------------------------------------------------------------

import argparse
import numpy as np
import pandas as pd
import spacy

from gensim.corpora import Dictionary
from gensim.models.tfidfmodel import TfidfModel
from gensim.matutils import sparse2full

from sklearn.feature_selection import SelectFpr, chi2
 
FLAGS = None




def keep_token(t):
    return (t.is_alpha and 
            not (t.is_space or t.is_punct or 
                 t.is_stop or t.like_num))

def lematize_comment(comment):
    return [ t.lemma_ for t in comment if keep_token(t)]
            

def lematize_comments(comments):
    docs = []
    for c in nlp.pipe(comments, batch_size=100, n_threads=4):
        docs.append(lematize_comment(c))
    return docs


if  __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i',type=str,
                        required=True,
                        dest='infile',
                        help='Input file to convert')

    parser.add_argument('-o',type=str,
                        required=True,
                        dest='outfile',
                        help='File to store output')

    parser.add_argument('--convertonly',
                        action='store_true',
                        default=False,
                        dest='convertOnly',
                        help='Only do the conversion')

    parser.add_argument('-d',type=str,
                        dest='dictFile',
                        default='data/derived/comments_dict.dat')

    parser.add_argument('-t',type=str,
                        dest='tfidfFile',
                        default='data/derived/comments_tfidf.dat')

    parser.add_argument('-c',type=str,
                        dest='chi2File',
                        default='data/derived/comments_chi2indices.npy')
    
    FLAGS, unparsed = parser.parse_known_args()

    # NB - a convert only flag makes sense as a command line
    #      argument but in code it is easier to work with a
    #      doTrain flag.
    doTrain = not FLAGS.convertOnly

    print("Reading data...")
    data = pd.read_csv(FLAGS.infile)

    # XXX - uncomment for debugging
    #data = data.iloc[0:10000,:]
    
    if doTrain:
        labelColnames =  data.columns.tolist()[2:]        
        data['any']   = data[labelColnames].apply(lambda x: int(any(x)), axis=1)
        
    nlp  = spacy.load('en_core_web_md')

    print("Lematizing comments....")
    comments_text = data['comment_text']
    data.drop(['comment_text'], inplace=True, axis=1)
    docs = lematize_comments(comments_text)

    comments_dictionary = None
    if doTrain:
        print("Creating dictionary....")
        comments_dictionary = Dictionary(docs)
        comments_dictionary.filter_extremes(no_below=10, no_above=0.3)
        comments_dictionary.compactify()
        comments_dictionary.save(FLAGS.dictFile)
    else:
        print("Loading dictionary...")
        comments_dictionary = Dictionary.load(FLAGS.dictFile)        

    print("Converting to BOW vectors...")
    comments_corpus = [comments_dictionary.doc2bow(d) for d in docs]

    model_tfidf = None
    if doTrain:
        print("Creating tfidf model...")
        model_tfidf = TfidfModel(comments_corpus)
        model_tfidf.save(FLAGS.tfidfFile)
    else:
        print("Loading tfidf model...")
        model_tfidf = TfidfModel.load(FLAGS.tfidfFile)

    print("Converting to tfidf vectors...")
    comments_tfidf  = model_tfidf[comments_corpus]
    comments_vecs   = np.vstack([sparse2full(c, len(comments_dictionary)) for c in comments_tfidf])

    chi2_features = None
    if doTrain:
        # Find most descrimitive words for any of the labels
        print("Finding discrimitive features...")
        labels = np.array(data['any'])
        model_fpr = SelectFpr(chi2, alpha=0.025)
        model_fpr.fit(comments_vecs, labels)
        chi2_features = model_fpr.get_support(indices=True)
        np.save(FLAGS.chi2File, chi2_features)
        
    else:
        print("Loading discrimitive features data...")
        chi2_features = np.load(FLAGS.chi2File)


    print("Calculating tfidf weighted word2vec vectors...")
    chi2_vecs  = comments_vecs[:,chi2_features]
    fpr_tokens = [nlp(t) for t in [comments_dictionary[i] for i in chi2_features]]
    avg_vecs   = []
    num_scores = len(chi2_features)
    for i in range(chi2_vecs.shape[0]):
        scores = chi2_vecs[i,:]
        avgvec = np.sum([fpr_tokens[j].vector * scores[j] for j in range(num_scores)], axis=0, keepdims=True)
        avg_vecs.append(avgvec)
        
    avg_vecs = np.vstack(avg_vecs)
    avg_vecs = pd.DataFrame(avg_vecs)
    avg_vecs.rename(columns=lambda x: 'F'+str(x), inplace=True)

    if doTrain:
        data.drop(['any'],axis=1,inplace=True)

    print("Saving converted data...")
    data_converted = pd.concat([data, avg_vecs], axis=1)
    data_converted.to_pickle(FLAGS.outfile)

    
