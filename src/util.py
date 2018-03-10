# ------------------------------------------------------------
# util.py
#
# Common code for Kaggle Toxic Comment Challenge
#
# J. Cardente
# 2018
# ------------------------------------------------------------
from gensim.models import KeyedVectors
import pandas as pd
import numpy as np
import spacy
import pickle
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split


def splitData(df, PARAMS):
    val_data   = None
    train_data = None
    if PARAMS['validationPercentage'] > 0.0:
        labels = df.iloc[:,2:]
        any = np.max(labels, axis=1)
        vp  = float(PARAMS['validationPercentage']) / 100
        train_data, val_data = train_test_split(df, test_size=vp, stratify=any)
    else:
        train_data = df

    return train_data, val_data


def get_epoch_val(params, numepochs, currepoch):
    tmpCount = 0
    pval = None
    for i in range(len(params)):
        tmpCount += numepochs[i]
        if currepoch < tmpCount:
            pval = params[i]
            break
    return pval

def keep_token(t):
    return not (t.is_space or t.is_punct or 
                t.is_stop or t.like_num or
                t.like_url or t.like_email)

def lematize_comment(comment):
    return [ t.lemma_ for t in comment if keep_token(t)]


def lematize_comments(comments, nlp, nthreads=4, batch_size=1000):
    docs = []
    for c in nlp.pipe(comments, batch_size=batch_size, n_threads=nthreads):
        lc = lematize_comment(c)
        docs.append(lc)
    return docs


def load_nlp():
     return spacy.load('en_core_web_md', disable=['ner','parser'])

def load_data(fname):
    data = pd.read_csv(fname)
    data['comment_text'].fillna('', inplace=True)
    return data


def load_fasttext(fname):
    # NB - this loads the FastText embeddings downloaded from
    #      the FastText website. It's in an text format that
    #      is slow to load. Use the binary version converted
    #      to Gensim KeyedVectors format.
    ft_model = KeyedVectors.load_word2vec_format(fname)
    return ft_model


def load_embedding(fname):
    ft_mode = KeyedVectors.load(fname)
    return ft_mode


def load_vocab(fname):
    vocab = pd.read_csv(fname)
    t2id = {t:i for t,i in zip(vocab['tokens'],vocab['ids'])}
    id2t = {i:t for t,i in zip(vocab['tokens'],vocab['ids'])}
    id2score = {i:s for i,s in zip(vocab['ids'],vocab['scores']) if s > 0}
    return vocab, t2id, id2t, id2score


def load_chi2(fname):
    f = open(fname, 'rb')
    term_scores = pickle.load(f)
    return term_scores


def score_predictions(categories, labels, probs, PARAMS):
    scores = {}
    
    if labels.shape != probs.shape:
        print('Uh oh, labels and probabilities shapes dont match')
        return scores

    accuracies = []
    precisions = []
    recalls    = []
    f1s        = []
    rocs       = []
    npos       = 0
    for i in range(probs.shape[1]):
        class_labels  = labels[:,i]
        class_probs   = probs[:,i]
        class_pred    = np.zeros_like(class_probs)# round(class_probs)
        class_pred[np.where(class_probs >= PARAMS['threshold'])] = 1
        class_npos    = np.sum(class_labels == 1)    

        tmp = {}
        tmp['npos'] = class_npos
        tmp['accuracy'] = metrics.accuracy_score(class_labels, class_pred)
        if np.any(class_labels == 1) and np.any(class_pred == 1):
            tmp['precision'] = metrics.precision_score(class_labels, class_pred)
            tmp['recall']    = metrics.recall_score(class_labels, class_pred)
            tmp['f1']        = metrics.f1_score(class_labels, class_pred)
        else:
            tmp['precision'] = 0
            tmp['recall']    = 0
            tmp['f1']        = 0
                                                                    
        try:
            tmp['roc'] = metrics.roc_auc_score(class_labels, class_probs)
        except ValueError:
            tmp['roc'] = 0

        npos += class_npos
        accuracies.append(tmp['accuracy'])
        precisions.append(tmp['precision'])
        recalls.append(tmp['recall'])
        f1s.append(tmp['f1'])            
        rocs.append(tmp['roc'])

        scores[categories[i]] = tmp

    scores['all'] = {}
    scores['all']['npos']      = npos
    scores['all']['accuracy']  = np.mean(accuracies)
    scores['all']['precision'] = np.mean(precisions)
    scores['all']['recall']    = np.mean(recalls)
    scores['all']['f1']        = np.mean(f1s)
    scores['all']['roc']   = np.mean(rocs)
    
    return scores
