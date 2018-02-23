# ------------------------------------------------------------
# rnn.py
#
# Model to predict toxic comment labels
# from word vectors using a TensorFlow RNN model.
#
# J. Cardente
# 2018
# ------------------------------------------------------------

import argparse
import numpy as np
import pandas as pd
import spacy
import pickle
import tensorflow as tf
import sklearn.metrics as metrics
import time
from timeit import default_timer as timer

from gensim.models.fasttext import FastText

FLAGS = None

PARAMS = {
    'learningRates': [0.01,0.001,0.0001],    
    'numEpochs' : [10,5,2],
    'batchSize': 256,
    'validationPercentage': 0,
    'threshold': 0.5,
    'maxwords': 50,
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


def text2vecs(text, embeddings, maxwords):
    tokens = text.split(' ')
    tokens = [t for t in tokens if t in embeddings]
    tokens = tokens[0:maxwords]
    vecs   = [embeddings[t] for t in tokens]
    nvecs  = len(vecs)

    if nvecs == 0:
        # NB - this happens when the comment contains
        #      non-english text using different character
        #      sets (eg Arabic). For now, we'll just use
        #      the NONE vector
        vecs = [embeddings['--NONE--']]
        nvecs = 1
        
    vecs = np.vstack(vecs)        
    if nvecs < maxwords:
        topad = maxwords - nvecs
        vecs  = np.pad(vecs, [[0,topad],[0,0]], 'constant')

    return nvecs, vecs


def inputGenerator(df, embeddings, PARAMS):
    # NB - df is a pandas dataframe
    columns    = df.columns.tolist()
    epochSize  = len(df.index)
    batchStart = 0
    batchSize  = PARAMS['batchSize'] 
    while batchStart < epochSize:
        batchData = df.iloc[batchStart:(batchStart+batchSize),:]
        ids       = batchData['id'].tolist()
        comments  = batchData['comment_text'].tolist()
        
        batch = {}
        batch['ids']  = batchData['id'].tolist()

        tmp = [text2vecs(c, embeddings, PARAMS['maxwords']) for c in comments]

        batch['lengths'] = np.array([t[0] for t in tmp])
        batch['vecs']    = np.stack([t[1] for t in tmp])
        
        if len(columns) > 2:
            batch['labels'] = np.array(batchData.iloc[:,2:])
            
        batchStart += batchSize            
        yield batch
            

def score_predictions(labels, probs):
    scores = {
        'accuracy': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'f1': 0.0,
        'roc_auc': 0.0,
        }
    
    if labels.shape != probs.shape:
        print('Uh oh, labels and probabilities shapes dont match')
        return scores

    accuracies = []
    precisions = []
    recalls    = []
    f1s        = []
    rocs       = []
    for i in range(probs.shape[1]):
        class_labels = labels[:,i]
        class_probs  = probs[:,i]
        class_pred   = np.round(class_probs)
        
        accuracies.append(metrics.accuracy_score(class_labels, class_pred))
        if np.any(class_labels == 1) and np.any(class_pred == 1):
            precisions.append(metrics.precision_score(class_labels, class_pred))
            recalls.append(metrics.recall_score(class_labels, class_pred))
            f1s.append(metrics.f1_score(class_labels, class_pred))
        else:
            precisions.append(0)
            recalls.append(0)
            f1s.append(0)
            
        try:
            roc = metrics.roc_auc_score(class_labels, class_probs)
        except ValueError:
            roc = 0
        rocs.append(roc)

    scores['accuracy'] = np.mean(accuracies)
    scores['precision'] = np.mean(precisions)
    scores['recall'] = np.mean(recalls)
    scores['f1'] = np.mean(f1s)
    scores['roc_auc'] = np.mean(rocs)
    
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
                        default='models/embeddings.dat',
                        dest='embedfile',
                        help='FastText embeddings')
    
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

    # Load word embeddings
    embeddings = FastText.load(FLAGS.embedfile)
    dictionary = {w:i for i,w in enumerate(list(embeddings.wv.vocab))}


    # DEFINE THE GRAPH
    tf.reset_default_graph()
    isTraining    = tf.placeholder(tf.bool, name='istraining')
    input_vecs    = tf.placeholder(tf.float32, shape=[None, PARAMS['maxwords'], embeddings.vector_size], name='input_vecs')
    input_lengths = tf.placeholder(tf.int32, shape=[None], name='input_lengths')
    input_labels  = tf.placeholder(tf.int32, shape=[None,len(categories)], name='input_labels')        
    learning_rate = tf.placeholder(tf.float32, [], name='learning_rate')
    threshold     = tf.constant(PARAMS['threshold'], dtype=tf.float32, name='probit_threshold')
    global_step   = tf.Variable(0, name='global_step',trainable=False)
    
    with tf.device("/gpu:0"):

        #used = tf.sign(tf.reduce_max(tf.abs(input_vecs), 2))
        #length = tf.reduce_sum(used, 1)
        #length = tf.cast(length, tf.int32)        

        layers    = [128, 128, 128]
        rnn_cells = [tf.contrib.rnn.LSTMCell(num_units=n, use_peepholes=True) for n in layers]
        stacked_cell = tf.contrib.rnn.MultiRNNCell(rnn_cells)
        outputs, states = tf.nn.dynamic_rnn(cell=stacked_cell,
                                            inputs=input_vecs,
                                            dtype=tf.float32)
        flat_states = tf.concat(states[len(layers)-1], axis=1) 

        dense1 = tf.layers.dense(flat_states, units=1024, activation=tf.nn.relu)
        dense2 = tf.layers.dense(dense1, units=1024, activation=tf.nn.relu)        
        logits = tf.layers.dense(dense2, units=len(categories))

        loss   = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                         labels=tf.cast(input_labels,dtype=tf.float32))
        loss   = tf.reduce_mean(loss, axis=0)
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
        sess.run([init_op_global, init_op_local])

        if FLAGS.doTrain:
            
            print("Reading training data...")
            data = pd.read_csv(FLAGS.trainfile)
            train_data, val_data = splitData(data, PARAMS)

            for epoch in range(sum(PARAMS['numEpochs'])):            
                print("Epoch " + str(epoch))

                tmpCount = 0
                for i in range(len(PARAMS['numEpochs'])):
                    tmpCount += PARAMS['numEpochs'][i]
                    if epoch < tmpCount:
                        epochLearningRate = PARAMS['learningRates'][i]
                        break

                timeStart = timer()
                for batch in inputGenerator(train_data, embeddings, PARAMS):
                    feed_dict = {learning_rate: epochLearningRate,
                                 input_lengths: batch['lengths'],
                                 input_vecs:  batch['vecs'],
                                 input_labels: batch['labels'],
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
                for batch in inputGenerator(val_data, embeddings, PARAMS):
                    feed_dict = {learning_rate: epochLearningRate,
                                 input_vecs:  batch['vecs'],
                                 input_lengths: batch['lengths'],                                 
                                 input_labels: batch['labels'],
                                 isTraining: 0}

                    batch_probits, batch_preds, batch_accuracy = sess.run([probits, predictions, accuracy], feed_dict=feed_dict)
                    val_probits.append(batch_probits)

                    batchCount += 1            
                    if batchCount % batchReportInterval == 0:
                        timeEnd = timer()
                        valRate = float(batchReportInterval* PARAMS['batchSize']) / (timeEnd - timeStart)
                        print("Batch {} Accuracy {} Rate {:.2f}".format(batchCount, batch_accuracy, valRate))
                        timeStart = timer()

                valTimeEnd  = timer()
                val_probits = np.vstack(val_probits)
                val_labels  = val_data.iloc[:, 2:].as_matrix()
                scores      = score_predictions(val_labels, val_probits)            
                print("Validation Total Time {:.2f}m".format((valTimeEnd-valTimeStart)/60))
                print('    Accuracy: {:.2f}'.format(scores['accuracy']))
                print('   Precision: {:.2f}'.format(scores['precision']))
                print('      Recall: {:.2f}'.format(scores['recall']))
                print('          F1: {:.2f}'.format(scores['f1']))
                print('     ROC AUC: {:.2f}'.format(scores['roc_auc']))


        if FLAGS.doTest:

            if not FLAGS.doTrain:
                chkpFullName = os.path.join(FLAGS.checkpointDir, FLAGS.checkpointName)
                print('Restoring model {}'.format(chkpFullName))            
                if not isfile(chkpFullName):
                    print("Error, checkpoint file doesnt exist {}")
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
            for batch in inputGenerator(test_data, embeddings, PARAMS):
                feed_dict = {learning_rate: epochLearningRate,
                             input_vecs:  batch['vecs'],
                             input_lengths: batch['lengths'],                             
                             isTraining: 0}

                batch_probits = sess.run(probits, feed_dict=feed_dict)
                inf_ids.extend(batch['ids'])
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



            
                

