# ------------------------------------------------------------
# learn_fasttext.py
#
# Utility to learn fasttext embeddings from comment text.
# Saves result for use by other utilities and models
#
# J. Cardente
# 2018
# ------------------------------------------------------------

import argparse
import pandas as pd

from gensim.models.fasttext import FastText


FLAGS = None


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-i',type=str,
                        required=True,
                        dest='infile',
                        help='Input file to convert')
    parser.add_argument('-e',type=str,
                        required=True,
                        dest='embedfile',
                        help='Name of embeddings file')
    parser.add_argument('-s',type=int,
                        default=300,
                        dest='size',
                        help='Embedding vector size')
    parser.add_argument('-w',type=int,
                        default=4,
                        dest='nworkers',
                        help='Number of workers')
    
    FLAGS, unparsed = parser.parse_known_args()

    print("Reading data...")
    data = pd.read_csv(FLAGS.infile)
    comments_text = data['comment_text']
    docs = [c.split(' ') for c in comments_text]
                     
    print('Learning embeddings...')    
    ft_model = FastText(sentences=docs, size=FLAGS.size, workers=FLAGS.nworkers)

    print('Saving model...')
    ft_model.save(FLAGS.embedfile)
