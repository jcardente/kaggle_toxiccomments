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
import regex
from util import load_data

FLAGS = None


replacement_regexs = [
    (r'f[*]{3}', 'fuck'),
    (r'fu[*]k', 'fuck'),    
    (r'f[*]+[c]{0,1}[k]{0,1}', 'fuck'),
    (r'fukkers', 'fuckers'),
    (r'sh[!*]t', 'shit'),
    (r'b[*]tch', 'bitch'),
    (r'tw[*]t', 'twat'),
    (r'a[*]+hole', 'asshole'),
    (r'faggit', 'faggot'),
    (r'shouldn[;\'d]t', 'should not'),
    (r'-(PRON|pron)-', '')
]

def replace_word(token):
    for p,s in replacement_regexs:
        token = re.sub(p,s,token)
    return token
    

def clean_token(token):

    # Get rid of things that look like IP addresses.
    token = re.sub(r'[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+', '', token)

    # Get rid of things that look like a time
    token = re.sub(r'[0-9]{1,2}:[0-9]{1,2}[a-zA-Z]*$','', token)

    # Get rid of things that look like ethernet mac addresses
    token = re.sub('^[0-9a-fA-F:]+$','',token)
    
    # Get rid of things that look like style tags or markup
    token = re.sub(r'(style|class|colspan|valign|vspan|width|rowspan)[=].*$','',token)
    token = re.sub(r'(border|cellspacing|align|cellpadding)[=].*$','',token)
    token = re.sub(r'^.*(border|color|padding|spacing):.*$','', token)
    token = re.sub(r'color:[#][0-9a-fA-F]{6}','',token)
    token = re.sub(r'bgcolor[=].*$','', token)

    token = re.sub(r'index\.php\?title','', token)
    
    # Get rid of anything not in latin character set
    token = regex.sub(r'[^\p{Latin}]','',token)
    
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
    data = load_data(FLAGS.infile)
    comments_text = data['comment_text']
    comments_text = comments_text.tolist()

    print('Cleaning text...')
    for i in range(len(comments_text)):
        comment = comments_text[i]
        new_tokens = []
        for token in comment.split(' '):
            token = replace_word(token)
            token = clean_token(token)

            if len(token) > 0:
                new_tokens.append(token)
        new_comment = ' '.join(new_tokens)
        comments_text[i] = new_comment

    print('Saving results...')
    data['comment_text'] = comments_text
    data.to_csv(FLAGS.outfile, index=False)
