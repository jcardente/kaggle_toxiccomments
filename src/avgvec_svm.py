# ------------------------------------------------------------
# avgvec_svm.py
#
# Simple SVM model to predict toxic comment labels
# from a TFIDF weighted word2vec vector using an
# SVM model.
#
# J. Cardente
# 2018
# ------------------------------------------------------------

import argparse
import numpy as np
import pandas as pd
import pickle

from sklearn import svm
from sklearn import metrics

FLAGS = None


if  __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-t',type=str,
                        default='data/train.csv',
                        dest='trainfile',
                        help='Training data file')

    parser.add_argument('-p',type=str,
                        default='data/test.csv',
                        dest='testfile',
                        help='Test data file')

    parser.add_argument('--train',
                        action='store_true',
                        default=False,
                        dest='doTrain',
                        help='Train a model')

    parser.add_argument('--test',
                        action='store_true',
                        default=False,
                        dest='doTest',
                        help='Predict test data')
    
    parser.add_argument('-o',type=str,
                        dest='subfile',
                        default='submission/submission.csv',
                        help='File to store test submission output')

    parser.add_argument('-m',type=str,
                        required=True,
                        dest='modelfile',
                        help='File to load/store model from/to')
    
    FLAGS, unparsed = parser.parse_known_args()


    models     = {}
    if FLAGS.doTrain:
        print('Loading training data....')
        data = pd.read_pickle(FLAGS.trainfile)

        colnames = data.columns.tolist()
        vecStart = colnames.index('F0')

        avg_vecs = np.array(data.iloc[:,vecStart:])
        data = data[colnames[:vecStart]]

        print('Training models...')
        categories =  data.columns.tolist()[1:]
        for cat in categories:
            print('\t{}....'.format(cat))
            labels = data[cat]
            models[cat] = svm.SVC(probability=True, kernel='rbf')
            models[cat].fit(avg_vecs, labels) 

        with open(FLAGS.modelfile, 'wb') as f:
            pickle.dump(models, f)

        print('Model saved to file {}'.format(FLAGS.modelfile))

        print('Evaluating fit to training data...')
        for cat in categories:
            labels    = data[cat]
            predicted = models[cat].predict(avg_vecs)
            print('{}\t Accuracy: {:.3f} F1 {:.3f}'.format(cat, metrics.accuracy_score(labels, predicted), metrics.f1_score(labels, predicted)))
            
    else:
        with open(FLAGS.modelfile, 'rb') as f:
            models = pickle.load(f)


    if FLAGS.doTest:
        data = pd.read_pickle(FLAGS.testfile)

        colnames = data.columns.tolist()
        vecStart = colnames.index('F0')

        avg_vecs = np.array(data.iloc[:,vecStart:])
        data = data[colnames[:vecStart]]

        categories =  models.keys()
        for cat in categories:
            predicted = models[cat].predict_proba(avg_vecs)
            data[cat] = predicted[:,1]

        # NB - to be safe, make sure columns are in the same order as the sample
        #      submission
        data = data[['id','toxic','severe_toxic','obscene','threat','insult','identity_hate']]
        data.to_csv(FLAGS.subfile, index=False, float_format='%.5f')

        print('Submission saved as file {}'.format(FLAGS.subfile))
