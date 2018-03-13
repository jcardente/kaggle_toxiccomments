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
import time
from timeit import default_timer as timer
from util import load_data, Vocab, get_epoch_val, score_predictions, splitData
import rnn_models as models

FLAGS = None

PARAMS = {
    'numEpochs' : [10,2],
    'batchSize': 512,
    'validationPercentage': 0,
    'threshold': 0.5,
    'maxwords': 50,
    'optLearningRates': [0.001,0.0001],
    'optBeta1': 0.8,
    'optBeta2': 0.99,
    'optEpsilon': 0.001,
    'weightEmbeddings': True,
    'trainEmbeddings': False,
    'lossReweight': True,
    'lossWeightAdjust': 1,
    'categories':  ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
}


def vprint(*args):
    if FLAGS.verbose:
        print(*args)
        

def text2ids(text, vocab, PARAMS):
    tokens    = text.split(' ')
    token_ids = [vocab.token2id(t) for t in tokens]
    token_ids = [id for id in token_ids if id != None]
    maxwords  = PARAMS['maxwords']
    
    ntokens = len(token_ids)
    if ntokens == 0:
        token_ids    = np.zeros(1, dtype=np.int)
        token_scores = np.zeros((1,vocab.id2scores(1).shape[0]), dtype=np.float32)
        ntokens      = 1
        
    else:
        # Score the tokens
        scored    = [(i,id,vocab.id2quickscore(id, mode='mean')) for i,id in enumerate(token_ids)]
        scored.sort(key=lambda x: x[2], reverse=True)
        scored    = scored[0:maxwords]

        # NB - preserve order of words in text
        scored.sort(key=lambda x: x[0])

        token_ids    = [s[1] for s in scored]
        token_scores = np.vstack([vocab.id2scores(id) for id in token_ids])
        ntokens = len(token_ids)
        
    
    if ntokens < maxwords:
        topad = maxwords - ntokens
        token_ids    = np.pad(token_ids, (0,topad), 'constant', constant_values=0)
        token_scores = np.pad(token_scores, [(0,topad),(0,0)], 'constant', constant_values=0)

    return ntokens, token_ids, token_scores



def inputGenerator(df, vocab, class_loss_weights, PARAMS, randomize=False):
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

        tmp = [text2ids(c, vocab, PARAMS) for c in comments]
        batch['lengths']      = np.array([t[0] for t in tmp])
        batch['tokenids']     = np.stack([t[1] for t in tmp])
        batch['term_scores']  = np.stack([t[2] for t in tmp])
        
        if len(columns) > 2:            
            labels  = np.array(batchData.iloc[:,2:])

            if PARAMS['lossReweight']:
                weights = np.tile(class_loss_weights, (len(batchIdxs), 1))
                batch_weights = weights * labels * PARAMS['lossWeightAdjust'] + (1-labels) * np.ones_like(labels) 
            else:
                batch_weights = np.ones_like(labels)
            
            batch['labels'] = labels
            batch['loss_weights'] = batch_weights

        batchStart += batchSize            
        yield batch
            

        
def calc_class_loss_weights(df):
    labelcols = df.columns.tolist()[2:]
    labelvals = df[labelcols].as_matrix()
    probs = np.mean(labelvals, axis=0)
    log_odds_ratio = np.log((1-probs)/probs)
    return log_odds_ratio


            

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
    
    parser.add_argument('-v',type=str,
                        dest='vocabfile',
                        default='models/vocab.csv')

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

    parser.add_argument('--noverbose',
                        action='store_false',
                        default=True,
                        dest='verbose',
                        help='Disble verbose output')
                            
    FLAGS, unparsed = parser.parse_known_args()
    
    PARAMS['validationPercentage'] = FLAGS.validationPercentage
    categories = PARAMS['categories'] 

    # Load word embeddings and vocab
    vprint('Loading vocabulary...')
    vocab = Vocab(FLAGS.vocabfile)
    vocab.add_embeddings(FLAGS.embedfile)
    scores_size = vocab.id2scores(1).shape[0]
    embed_shape = vocab.embed_vecs.shape
    
        
    # DEFINE THE GRAPH
    tf.reset_default_graph()
    isTraining         = tf.placeholder(tf.bool, name='istraining')
    input_ids          = tf.placeholder(tf.int32, shape=[None, PARAMS['maxwords']], name='input_ids')
    input_term_scores  = tf.placeholder(tf.float32, shape=[None, PARAMS['maxwords'], scores_size], name='input_term_scores')
    input_lengths      = tf.placeholder(tf.int32, shape=[None], name='input_lengths')
    input_labels       = tf.placeholder(tf.int32, shape=[None,len(categories)], name='input_labels')
    input_loss_weights = tf.placeholder(tf.float32, shape=[None, len(categories)], name='input_weights')
    input_embeddings   = tf.placeholder(tf.float32, shape=embed_shape)
    
    learning_rate = tf.placeholder(tf.float32, [], name='learning_rate')
    threshold     = tf.constant(PARAMS['threshold'], dtype=tf.float32, name='probit_threshold')
    global_step   = tf.Variable(0, name='global_step',trainable=False)
    
    with tf.device("/gpu:0"):

        # NB - put embeddings on GPU!
        tfembeddings = tf.Variable(input_embeddings, trainable=PARAMS['trainEmbeddings'], name='embedding_vectors')
        input_vecs   = tf.gather(tfembeddings, input_ids)

        if PARAMS['weightEmbeddings']:
            #term_weights = tf.reduce_mean(input_term_scores, axis=2)
            #vec_weights  = tf.tile(tf.expand_dims(term_weights, 2), [1,1,embed_shape[1]])

            # NB - Use a 1D convolution to determine how best to combine weights
            term_weights = tf.layers.conv1d(input_term_scores, filters=1, kernel_size=1, strides=1)
            vec_weights  = tf.tile(term_weights, [1,1,embed_shape[1]])
            input_vecs   = tf.multiply(vec_weights, input_vecs)
        
        logits       = models.bidir_gru_pooled(input_vecs, input_lengths, isTraining, PARAMS)
        #logits       = models.stacked_lstm(input_vecs, input_lengths, isTraining, PARAMS)        
        
        loss   = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                         labels=tf.cast(input_labels,dtype=tf.float32))
        loss   = tf.multiply(loss, input_loss_weights)
        loss   = tf.reduce_sum(loss, name="loss")
        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate,
                                           beta1=PARAMS['optBeta1'],
                                           beta2=PARAMS['optBeta2'],
                                           epsilon=PARAMS['optEpsilon'])
        
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
        sess.run([init_op_global, init_op_local], feed_dict={input_embeddings: vocab.embed_vecs})

        if FLAGS.doTrain:
            
            vprint("Reading training data...")
            data = load_data(FLAGS.trainfile)
            class_loss_weights = calc_class_loss_weights(data)
            
            train_data, val_data = splitData(data, PARAMS)
            vprint('Training samples {}..'.format(len(train_data)))
            vprint('Validation samples {}..'.format(len(val_data)))

            for epoch in range(sum(PARAMS['numEpochs'])):
                vprint("Epoch " + str(epoch))

                epochLearningRate = get_epoch_val(PARAMS['optLearningRates'], PARAMS['numEpochs'], epoch)
                timeStart = timer()
                for batch in inputGenerator(train_data, vocab, class_loss_weights, PARAMS, randomize=True):
                    feed_dict = {learning_rate: epochLearningRate,
                                 input_lengths: batch['lengths'],
                                 input_ids:  batch['tokenids'],
                                 input_labels: batch['labels'],
                                 input_term_scores: batch['term_scores'],
                                 input_loss_weights: batch['loss_weights'],
                                 isTraining: 1}

                    _ , batch_loss, batch_probs, batch_preds, batch_accuracy = sess.run([training_op, loss, probits, predictions, accuracy], feed_dict=feed_dict)
                    batchCount += 1                
                    if FLAGS.verbose and batchCount % batchReportInterval == 0:
                        timeEnd = timer()
                        trainRate = float(batchReportInterval* PARAMS['batchSize']) / (timeEnd - timeStart)
                        print("Batch {} loss {} accuracy {} rate {}".format(batchCount, batch_loss, batch_accuracy, trainRate))
                        #writer.add_summary(summary, step)                    
                        timeStart = timer()

            trainTimeEnd = timer()
            vprint("Total Training Time {:.2f}m".format((trainTimeEnd-trainTimeStart)/60))

            if FLAGS.checkpoint:
                chkpFullName = os.path.join(FLAGS.checkpointDir, FLAGS.checkpointName)
                save_path    = saver.save(sess, chkpFullName)
            
            if not val_data.empty:
                vprint("Starting validation....")
                batchCount   = 0
                valTimeStart = timer()            
                timeStart    = valTimeStart
                val_probits  = []
                val_labels   = []
                for batch in inputGenerator(val_data, vocab, class_loss_weights, PARAMS, randomize=False):
                    feed_dict = {learning_rate: epochLearningRate,
                                 input_ids:  batch['tokenids'],
                                 input_lengths: batch['lengths'],                                 
                                 input_labels: batch['labels'],
                                 input_term_scores: batch['term_scores'],                                 
                                 isTraining: 0}

                    batch_probits, batch_preds, batch_accuracy = sess.run([probits, predictions, accuracy], feed_dict=feed_dict)
                    val_probits.append(batch_probits)
                    val_labels.append(batch['labels'])
                    
                    batchCount += 1            
                    if FLAGS.verbose and batchCount % batchReportInterval == 0:
                        timeEnd = timer()
                        valRate = float(batchReportInterval* PARAMS['batchSize']) / (timeEnd - timeStart)
                        print("Batch {} Accuracy {:.5f} Rate {:.2f}".format(batchCount, batch_accuracy, valRate))
                        timeStart = timer()

                valTimeEnd  = timer()
                vprint("Validation Total Time {:.2f}m".format((valTimeEnd-valTimeStart)/60))

                # Print results of validation
                val_probits = np.vstack(val_probits)
                val_labels  = np.vstack(val_labels) 
                scores      = score_predictions(categories, val_labels, val_probits, PARAMS)

                cols        = ['npos','accuracy','precision','recall','f1','roc']
                headers     = '{:^13}' + ''.join(['{:^10}'] * len(cols))
                print(headers.format('',*cols))
                rows        = categories +  ['all']
                rowfmt      = '{:^13}'+ ''.join(['{:^10.4f}'] * len(cols))
                for i in range(len(rows)):
                    cn = rows[i]
                    cs = scores[cn]
                    cv = [cn] + [cs[x] for x in cols]
                    print(rowfmt.format(*cv))
                    

        if FLAGS.doTest:

            if not FLAGS.doTrain:
                chkpFullName = os.path.join(FLAGS.checkpointDir, FLAGS.checkpointName)
                metaFile     = chkpFullName + '.meta'                                
                vprint('Restoring model {}'.format(chkpFullName))
                if not isfile(metaFile):
                    print("Error, checkpoint file doesnt exist")
                    sys.exit(1)
                saver.restore(sess,chkpFullName)
            

            vprint('Loading test data....')
            test_data = load_data(FLAGS.testfile)    

            vprint("Starting inference....")
            batchCount   = 0
            infTimeStart = timer()            
            timeStart    = infTimeStart
            inf_ids      = []
            inf_probits  = []
            for batch in inputGenerator(test_data, vocab, None, PARAMS, randomize=False):
                feed_dict = {learning_rate: epochLearningRate,
                             input_ids:  batch['tokenids'],
                             input_lengths: batch['lengths'],
                             input_term_scores: batch['term_scores'],                             
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
            vprint("Total Inference Time {:.2f}m".format((infTimeEnd-infTimeStart)/60))


            inf_probits = np.vstack(inf_probits)
            df = pd.DataFrame(data=inf_probits, index=inf_ids, columns = categories)
            df.index.name = 'id'

            vprint("Saving submission....")
            
            df.to_csv(FLAGS.subfile, index=True, float_format='%.5f')



            
                

