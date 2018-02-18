# ------------------------------------------------------------
# avgvec_dnn.py
#
# Model to predict toxic comment labels
# from a TFIDF weighted word2vec vector using a
# TensorFlow DNN model.
#
# J. Cardente
# 2018
# ------------------------------------------------------------

import argparse
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf

import time
from timeit import default_timer as timer


FLAGS = None

PARAMS = {
    'learningRates': [0.001,0.0001],    
    'numEpochs' : [8,2],
    'batchSize': 64,
    'validationPercentage': 10
}


def getVectorSize(df):
    colnames   = df.columns.tolist()
    vecStart   = colnames.index('F0')
    return len(colnames) - vecStart
    
def getCategories(df):
    colnames   = df.columns.tolist()
    vecStart   = colnames.index('F0')
    return df.columns.tolist()[1:vecStart]    
    

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


def inputGenerator(df, PARAMS):
    # NB - df is a pandas dataframe
    colnames   = df.columns.tolist()
    vecStart   = colnames.index('F0')
    epochData  = []
    epochSize  = len(df.index)
    batchStart = 0
    batchSize  = PARAMS['batchSize'] 
    while batchStart < epochSize:
        batchData = df.iloc[batchStart:(batchStart+batchSize),:]
        batch = {}
        batch['ids'] = batchData.iloc[:,0].tolist()
        batch['vecs'] = np.array(batchData.iloc[:,vecStart:])
        batch['labels'] = np.array(batchData.iloc[:,1:vecStart])        
        batchStart += batchSize            
        yield batch
            

        
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

    # parser.add_argument('-m',type=str,
    #                     required=True,
    #                     dest='modelfile',
    #                     help='File to load/store model from/to')
    
    FLAGS, unparsed = parser.parse_known_args()
    
    print('Loading training data....')
    data = pd.read_pickle(FLAGS.trainfile)
    
    train_data, val_data = splitData(data, PARAMS)
    categories = getCategories(train_data)

    tf.reset_default_graph()
    isTraining    = tf.placeholder(tf.bool, name='istraining')
    input_vecs    = tf.placeholder(tf.float32, shape=[None, getVectorSize(train_data)], name='input_vecs')
    input_labels  = tf.placeholder(tf.int32, shape=[None,len(categories)], name='input_labels')        
    learning_rate = tf.placeholder(tf.float32, [], name='learning_rate')
    global_step = tf.Variable(0, name='global_step',trainable=False)    
    with tf.device("/gpu:0"):
        dense1 = tf.layers.dense(inputs=input_vecs, units = 1024, activation=tf.nn.relu)
        drop1  = tf.layers.dropout(inputs=dense1, rate=0.4, training=isTraining)

        dense2 = tf.layers.dense(inputs=dense1, units = 1024, activation=tf.nn.relu)
        drop2  = tf.layers.dropout(inputs=dense2, rate=0.4, training=isTraining)

        dense3 = tf.layers.dense(inputs=dense2, units = 1024, activation=tf.nn.relu)
        drop3  = tf.layers.dropout(inputs=dense3, rate=0.4, training=isTraining)

        logits = tf.layers.dense(inputs=dense3, units=len(categories))
        loss   = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.cast(input_labels,dtype=tf.float32))
        loss   = tf.reduce_mean(loss, axis=0)
        loss   = tf.reduce_sum(loss, name="loss")
        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
        
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            training_op = optimizer.minimize(loss, global_step=global_step)
        
        predictions = tf.nn.sigmoid(logits, name="predictions")

    with tf.device("/cpu:0"):
        accuracy, accuracy_op = tf.metrics.accuracy(input_labels, predictions, name="accuracy")

    tf.summary.scalar('loss',loss)
    tf.summary.scalar('accuracy',accuracy)
    summary_op = tf.summary.merge_all()

    writer              = tf.summary.FileWriter('logs', graph=tf.get_default_graph())    
    saver               = tf.train.Saver()    
    init_op_global      = tf.global_variables_initializer()
    init_op_local       = tf.local_variables_initializer()
    batchCount          = 0
    batchReportInterval = 10
    epochLearningRate   = 0.001
    trainTimeStart      = timer()
    with tf.Session() as sess:
        sess.run([init_op_global, init_op_local])

        for epoch in range(sum(PARAMS['numEpochs'])):            
            print("Epoch " + str(epoch))

            tmpCount = 0
            for i in range(len(PARAMS['numEpochs'])):
                tmpCount += PARAMS['numEpochs'][i]
                if epoch < tmpCount:
                    epochLearningRate = PARAMS['learningRates'][i]
                    break
            
            timeStart = timer()
            for batch in inputGenerator(train_data,PARAMS):
                feed_dict = {learning_rate: epochLearningRate,
                             input_vecs:  batch['vecs'],
                             input_labels: batch['labels'],
                             isTraining: 1}
                _ , batch_loss, batch_accuracy, summary, step = sess.run([training_op, loss, accuracy_op, summary_op, global_step], feed_dict=feed_dict)
                batchCount += 1                
                if batchCount % batchReportInterval == 0:
                    timeEnd = timer()
                    trainRate = float(batchReportInterval* PARAMS['batchSize']) / (timeEnd - timeStart)
                    print("Batch {} loss {} accuracy {} rate {}".format(batchCount, batch_loss, batch_accuracy, trainRate))
                    writer.add_summary(summary, step)                    
                    timeStart = timer()

        trainTimeEnd = timer()
        print("Total Training Time {:.2f}m".format((trainTimeEnd-trainTimeStart)/60))

        if not val_data.empty:
            print("Starting validation....")
            numCorrect   = 0
            numTotal     = 0
            checkCorrect = 0
            checkTotal   = 0
            batchCount   = 0
            valTimeStart = timer()            
            timeStart    = valTimeStart
            batch_accurace = 0
            for batch in inputGenerator(val_data, PARAMS):
                feed_dict = {learning_rate: epochLearningRate,
                             input_vecs:  batch['vecs'],
                             input_labels: batch['labels'],
                             isTraining: 0}
                
                batch_accuracy = sess.run([accuracy_op], feed_dict=feed_dict)
                batchCount += 1            
                if batchCount % batchReportInterval == 0:
                    timeEnd = timer()
                    valRate = float(batchReportInterval* PARAMS['batchSize']) / (timeEnd - timeStart)
                    print("Batch {} Accuracy {} Rate {:.2f}".format(batchCount, batch_accuracy, valRate))
                    timeStart = timer()

            valTimeEnd = timer()
            print("Validation Accuracy {} Total Time {:.2f}m".format(batch_accuracy, (valTimeEnd-valTimeStart)/60))
        
