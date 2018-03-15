# ------------------------------------------------------------
# clean_text.py
#
# Clean up some recuring issues after lematization. Based
# on observations of the processed data.
# 
#
# J. Cardente
# 2018
# ------------------------------------------------------------

import argparse
import pandas as pd
import re
from util import load_data

FLAGS = None


def clean_token(token):

    # Get rid of things that look like IP addresses.
    token = re.sub(r'[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+', '', token)

    # Get rid of things that look like a time
    token = re.sub(r'[0-9]{1,2}:[0-9]{1,2}','', token)

    # Get rid of anything not in latin character set
    token = regex.sub(r'[^\p{Latin}]','',token)
    
    # Get rid of trailing equals
    token = re.sub(r'[=]+$','',token)

    # Get rid of repeated punctuations like exclamation points.
    token = re.sub(r'[!]+','', token)

    # Get rid of leading |
    token = re.sub(r'^[|]','',token)

    # Get rid of trailing |
    token = re.sub(r'[|]$','',token)
    
    # Get rid of leading -, often in names
    token = re.sub(r'[-]+','',token)
    
    return token


    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-i',type=str,
                        required=True,
                        dest='infile',
                        help='input file')

    parser.add_argument('-o',type=str,
                        required=True,
                        dest='outfile',
                        help='Ouput file')
        
    FLAGS, unparsed = parser.parse_known_args()

    print('Reading data...')
    train_data = load_data(FLAGS.trainfile)
    comments_text = train_data['comment_text']
    comments_text = comments_text.tolist()

    for i in range(len(comments_text)):
        comment = comments_text[i]

        # Get rid of things that look like IP addresses.
        comment = re.sub(r'[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+', '', comment)

        # Get rid of things that look like a time
        comment = re.sub(r'[0-9]{1,2}:[0-9]{1,2}','', comment)

        # Get rid of trailing equals
        comment = re.sub(r'[=]+$','',comment)
        
        # Get rid of repeated punctuations like exclamation points.
        comment = re.sub(r'[!]+','', comment)

        # Replace
