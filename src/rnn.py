# ------------------------------------------------------------
# rnn.py
#
# Model to predict toxic comment labels
# from word vectors using a TensorFlow RNN model.
#
# J. Cardente
# 2018
# ------------------------------------------------------------

import os
from os.path import join, isfile, isdir
import sys
import argparse
import numpy as np
import pandas as pd
import spacy
import pickle
import tensorflow as tf
import sklearn.metrics as metrics
import time
from timeit import default_timer as timer
#from gensim.models.fasttext import FastText
from util import load_data, load_embedding, load_chi2
import rnn_models as models

FLAGS = None

PARAMS = {
    'learningRates': [0.001,0.0001],    
    'numEpochs' : [1,1],
    'batchSize': 512,
    'validationPercentage': 0,
    'threshold': 0.5,
    'maxwords': 100,
    'lossReweight': True,
    'lossBoost': 1,
    'nonetoken': '--EMPTY--',
    'categories':  ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
}


def splitData(df, PARAMS):
    val_data = pd.DataFrame()
    train_data = pd.DataFrame()
    if PARAMS['validationPercentage'] > 0.0:
        numTotal = len(df.index)
        numValidation = int(PARAMS['validationPercentage']/100*numTotal)

        indices = np.arange(numTotal)
        np.random.shuffle(indices)

        val_indices  = indices[:numValidation]
        train_indices = indices[numValidation:]

        val_data = df.iloc[val_indices,:]
        train_data = df.iloc[train_indices,:]

    else:
        train_data = df

    return train_data, val_data


def text2ids(text, vocab_scores, t2id, maxwords):
    tokens = text.split(' ')
    tokens = [t for t in tokens if t in t2id]
    scores = [(i, vocab_scores[t]) if t in vocab_scores else (i,0) for i,t in enumerate(tokens)]
    scores.sort(key=lambda x: x[1], reverse=True)

    # NB - preserve order of words in text
    keep    = [i for i,_ in scores[0:maxwords]]
    keep.sort()
    token_ids = [t2id[tokens[i]] for i in keep]
    ntokens   = len(token_ids)

    if ntokens == 0:
        # NB - this happens when the comment contains
        #      non-english text using different character
        #      sets (eg Arabic). For now, we'll just use
        #      the NONE vector
        token_ids  = [t2id[PARAMS['nonetoken']]]
        ntokens    = 1
        
    if ntokens < maxwords:
        topad = maxwords - ntokens
        padded = np.ones(maxwords, dtype=np.int) * t2id[PARAMS['nonetoken']]
        padded[:ntokens]  = token_ids
        token_ids = padded

    return ntokens, token_ids


def inputGenerator(df, vocab_scores, t2id, class_probs, PARAMS, randomize=False):
    # NB - df is a pandas dataframe
    columns    = df.columns.tolist()
    rowidxs    = np.arange(df.shape[0])
    if randomize:
        np.random.shuffle(rowidxs)
    epochSize  = len(df.index)
    batchStart = 0
    batchSize  = PARAMS['batchSize'] 
    while batchStart < epochSize:
        batchIdxs = rowidxs[batchStart:(batchStart+batchSize)]
        batchData = df.iloc[batchIdxs,:]
        ids       = batchData['id'].tolist()
        comments  = batchData['comment_text'].tolist()
        
        batch = {}
        batch['docids']  = batchData['id'].tolist()

        tmp = [text2ids(c, vocab_scores, t2id, PARAMS['maxwords']) for c in comments]
        batch['lengths']  = np.array([t[0] for t in tmp])
        batch['tokenids'] = np.stack([t[1] for t in tmp])
        
        if len(columns) > 2:            
            labels  = np.array(batchData.iloc[:,2:])

            if PARAMS['lossReweight']:
                probs   = np.tile(class_probs, (len(batchIdxs), 1))
                weights = labels * (1-probs) * PARAMS['lossBoost'] + (1-labels) * probs
            else:
                weights = np.ones_like(labels)
            
            batch['labels'] = labels
            batch['loss_weights'] = weights
           
            
        batchStart += batchSize            
        yield batch
            


def calc_class_probs(df):
    labelcols = df.columns.tolist()[2:]
    labelvals = df[labelcols].as_matrix()
    probs = np.mean(labelvals, axis=0)
    return probs

        
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



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-t',type=str,
                        default='data/train.csv',
                        dest='trainfile',
                        help='Training data file')

    parser.add_argument('-p',type=str,
                        default='data/test.csv',
                        dest='testfile',
                        help='Test data file')

    parser.add_argument('-e',type=str,
                        default='data/ftvecs.pkl',
                        dest='embedfile',
                        help='FastText embeddings')
    
    parser.add_argument('-c',type=str,
                        dest='chi2file',
                        default='models/chi2scores.pkl')

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

    parser.add_argument('--checkpoint',
                        action='store_true',
                        default=False,
                        dest='checkpoint',
                        help='Dont save a checkpoint')
        
    parser.add_argument('--checkdir', type=str, default='./chkpoints',
                        dest='checkpointDir',
                        help='Directory to store checkpoints')
    
    parser.add_argument('--checkname', type=str, default='model.ckpt',
                        dest='checkpointName',
                        help='Checkpoint filename')
        
    parser.add_argument('--val', type=int,
                        default=0,
                        dest='validationPercentage',
                        help='Validation percentage')
    
    FLAGS, unparsed = parser.parse_known_args()
    PARAMS['validationPercentage'] = FLAGS.validationPercentage

    categories = PARAMS['categories'] 

    # Load word embeddings and chi2 scores
    embeddings   = load_embedding(FLAGS.embedfile)
    chi2scores   = load_chi2(FLAGS.chi2file)
    vocab_scores = {k:v for k,v in chi2scores.items() if k in embeddings.vocab}

    # Pick the best terms to use and then get the embeddings for
    # them
    vocab = [PARAMS['nonetoken']] + [w for w in embeddings.vocab.keys()]
    t2id  = {w:(i+1) for i,w in enumerate(vocab[1:])}
    t2id[PARAMS['nonetoken']] = 0
    embedding_size   = embeddings.get_vector(vocab[1]).shape[0]    
    vocab_embeddings = np.stack([np.zeros(embedding_size)] + [embeddings.get_vector(w) for w in vocab[1:]])
    
    
    # DEFINE THE GRAPH
    tf.reset_default_graph()
    isTraining    = tf.placeholder(tf.bool, name='istraining')
    input_ids     = tf.placeholder(tf.int32, shape=[None, PARAMS['maxwords']], name='input_ids')
    input_lengths = tf.placeholder(tf.int32, shape=[None], name='input_lengths')
    input_labels  = tf.placeholder(tf.int32, shape=[None,len(categories)], name='input_labels')
    input_loss_weights = tf.placeholder(tf.float32, shape=[None, len(categories)], name='input_weights')
    input_embeddings   = tf.placeholder(tf.float32, shape=vocab_embeddings.shape)
    
    learning_rate = tf.placeholder(tf.float32, [], name='learning_rate')
    threshold     = tf.constant(PARAMS['threshold'], dtype=tf.float32, name='probit_threshold')
    global_step   = tf.Variable(0, name='global_step',trainable=False)
    
    with tf.device("/gpu:0"):

        # NB - put embeddings on GPU!
        tfembeddings = tf.Variable(input_embeddings, trainable=False, name='embedding_vectors')
        input_vecs   = tf.gather(tfembeddings, input_ids)
        logits       = models.stacked_bidir_lstm(input_vecs, input_lengths, PARAMS)
        
        loss   = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                         labels=tf.cast(input_labels,dtype=tf.float32))
        loss   = tf.multiply(loss, input_loss_weights)
        loss   = tf.reduce_sum(loss, name="loss")
        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
        
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            training_op = optimizer.minimize(loss, global_step=global_step)
        
        probits     = tf.nn.sigmoid(logits, name="predictions")
        cond        = tf.greater(probits, tf.ones_like(probits) * threshold)
        predictions = tf.where(cond,
                               tf.ones_like(probits,dtype=tf.int32),
                               tf.zeros_like(probits, dtype=tf.int32))        
        accuracy    = tf.reduce_mean(tf.cast(tf.equal(input_labels, predictions), dtype=tf.float32))
                
    tf.summary.scalar('loss',loss)
    tf.summary.scalar('accuracy',accuracy)
    summary_op = tf.summary.merge_all()

    #writer              = tf.summary.FileWriter('logs', graph=tf.get_default_graph())    
    saver               = tf.train.Saver()    
    init_op_global      = tf.global_variables_initializer()
    init_op_local       = tf.local_variables_initializer()
    batchCount          = 0
    batchReportInterval = 10
    epochLearningRate   = 0.001
    trainTimeStart      = timer()    
    with tf.Session() as sess:
        sess.run([init_op_global, init_op_local], feed_dict={input_embeddings: vocab_embeddings})

        if FLAGS.doTrain:
            
            print("Reading training data...")
            data = load_data(FLAGS.trainfile)
            class_probs = calc_class_probs(data)
            
            train_data, val_data = splitData(data, PARAMS)
            print('Training samples {}..'.format(len(train_data)))
            print('Validation samples {}..'.format(len(val_data)))

            for epoch in range(sum(PARAMS['numEpochs'])):            
                print("Epoch " + str(epoch))

                tmpCount = 0
                for i in range(len(PARAMS['numEpochs'])):
                    tmpCount += PARAMS['numEpochs'][i]
                    if epoch < tmpCount:
                        epochLearningRate = PARAMS['learningRates'][i]
                        break

                timeStart = timer()
                for batch in inputGenerator(train_data, vocab_scores, t2id, class_probs, PARAMS, randomize=True):
                    feed_dict = {learning_rate: epochLearningRate,
                                 input_lengths: batch['lengths'],
                                 input_ids:  batch['tokenids'],
                                 input_labels: batch['labels'],
                                 input_loss_weights: batch['loss_weights'],
                                 isTraining: 1}

                    _ , batch_loss, batch_probs, batch_preds, batch_accuracy = sess.run([training_op, loss, probits, predictions, accuracy], feed_dict=feed_dict)
                    batchCount += 1                
                    if batchCount % batchReportInterval == 0:
                        timeEnd = timer()
                        trainRate = float(batchReportInterval* PARAMS['batchSize']) / (timeEnd - timeStart)
                        print("Batch {} loss {} accuracy {} rate {}".format(batchCount, batch_loss, batch_accuracy, trainRate))
                        #writer.add_summary(summary, step)                    
                        timeStart = timer()

            trainTimeEnd = timer()
            print("Total Training Time {:.2f}m".format((trainTimeEnd-trainTimeStart)/60))

            if FLAGS.checkpoint:
                chkpFullName = os.path.join(FLAGS.checkpointDir, FLAGS.checkpointName)
                save_path    = saver.save(sess, chkpFullName)
            
            if not val_data.empty:
                print("Starting validation....")
                batchCount   = 0
                valTimeStart = timer()            
                timeStart    = valTimeStart
                val_probits  = []
                val_labels   = []
                for batch in inputGenerator(val_data, vocab_scores, t2id, class_probs, PARAMS, randomize=False):
                    feed_dict = {learning_rate: epochLearningRate,
                                 input_ids:  batch['tokenids'],
                                 input_lengths: batch['lengths'],                                 
                                 input_labels: batch['labels'],
                                 isTraining: 0}

                    batch_probits, batch_preds, batch_accuracy = sess.run([probits, predictions, accuracy], feed_dict=feed_dict)
                    val_probits.append(batch_probits)
                    val_labels.append(batch['labels'])
                    
                    batchCount += 1            
                    if batchCount % batchReportInterval == 0:
                        timeEnd = timer()
                        valRate = float(batchReportInterval* PARAMS['batchSize']) / (timeEnd - timeStart)
                        print("Batch {} Accuracy {} Rate {:.2f}".format(batchCount, batch_accuracy, valRate))
                        timeStart = timer()

                valTimeEnd  = timer()
                print("Validation Total Time {:.2f}m".format((valTimeEnd-valTimeStart)/60))


                val_probits = np.vstack(val_probits)
                val_labels  = np.vstack(val_labels) #val_data.iloc[:, 2:].as_matrix()
                scores      = score_predictions(categories, val_labels, val_probits, PARAMS)

                cols        = ['npos','accuracy','precision','recall','f1','roc']
                headers     = '{:^13}' + ''.join(['{:^10}'] * len(cols))
                print(headers.format('',*cols))
                rows        = categories +  ['all']
                rowfmt      = '{:^13}'+ ''.join(['{:^10.2f}'] * len(cols))
                for i in range(len(rows)):
                    cn = rows[i]
                    cs = scores[cn]
                    cv = [cn] + [cs[x] for x in cols]
                    print(rowfmt.format(*cv))
                    

        if FLAGS.doTest:

            if not FLAGS.doTrain:
                chkpFullName = os.path.join(FLAGS.checkpointDir, FLAGS.checkpointName)
                metaFile     = chkpFullName + '.meta'                                
                print('Restoring model {}'.format(chkpFullName))
                if not isfile(metaFile):
                    print("Error, checkpoint file doesnt exist")
                    sys.exit(1)
                saver.restore(sess,chkpFullName)
            

            print('Loading test data....')
            test_data = pd.read_csv(FLAGS.testfile)    

            print("Starting inference....")
            batchCount   = 0
            infTimeStart = timer()            
            timeStart    = infTimeStart
            inf_ids      = []
            inf_probits  = []
            for batch in inputGenerator(test_data, vocab_scores, t2id, None, PARAMS, randomize=False):
                feed_dict = {learning_rate: epochLearningRate,
                             input_ids:  batch['tokenids'],
                             input_lengths: batch['lengths'],                             
                             isTraining: 0}

                batch_probits = sess.run(probits, feed_dict=feed_dict)
                inf_ids.extend(batch['docids'])
                inf_probits.append(batch_probits)

                batchCount += 1            
                if batchCount % batchReportInterval == 0:
                    timeEnd = timer()
                    infRate = float(batchReportInterval* PARAMS['batchSize']) / (timeEnd - timeStart)
                    print("Batch {} Rate {:.2f}".format(batchCount, infRate))
                    timeStart = timer()

            infTimeEnd = timer()
            print("Total Inference Time {:.2f}m".format((infTimeEnd-infTimeStart)/60))


            inf_probits = np.vstack(inf_probits)
            df = pd.DataFrame(data=inf_probits, index=inf_ids, columns = categories)
            df.index.name = 'id'

            print("Saving submission....")
            
            df.to_csv(FLAGS.subfile, index=True, float_format='%.5f')



            
                

