import tensorflow as tf

def conv1d_termscores(input_vecs, input_lengths, input_term_scores, isTraining, PARAMS):

    conv1 = tf.layers.conv1d(input_term_scores, filters=128, kernel_size=2,
                             strides=1, padding='VALID',activation=tf.nn.relu)

    conv1 = tf.expand_dims(conv1, -1)
    
    conv2 = tf.layers.conv2d(conv1, filters=128, kernel_size=[3,3],
                             strides=1, padding='VALID')
    
    conv2 = tf.layers.batch_normalization(conv2, training=isTraining, momentum=0.9)
    conv2 = tf.nn.relu(conv2)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    conv3 = tf.layers.conv2d(conv2, filters=128, kernel_size=[3,3],
                             strides=1, padding='VALID')
    
    conv3 = tf.layers.batch_normalization(conv3, training=isTraining, momentum=0.9)
    conv3 = tf.nn.relu(conv3)
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

    flattened = tf.contrib.layers.flatten(pool3)
    
    fc1  = tf.layers.dense(flattened, 1024, activation=None)
    fc1  = tf.layers.batch_normalization(fc1, training=isTraining, momentum=0.9)
    fc1  = tf.nn.relu(fc1)
    drop1 = tf.layers.dropout(inputs=fc1, rate=0.4, training= isTraining)

    fc2  = tf.layers.dense(drop1, 1024, activation=None)
    fc2  = tf.layers.batch_normalization(fc2, training=isTraining, momentum=0.9)
    fc2  = tf.nn.relu(fc2)
        
    logits = tf.layers.dense(fc2, units=len(PARAMS['categories']))    

    return logits


def avgvec(input_vecs, input_lengths, input_term_scores, isTraining, PARAMS):

    vec_sum   = tf.reduce_sum(input_vecs,axis=1, keepdims=False)
    vec_denom = tf.tile(tf.expand_dims(tf.cast(input_lengths,tf.float32),1), [1,tf.shape(input_vecs)[2]])

    vec_avg = tf.div(vec_sum, vec_denom)

    dense1 = tf.layers.dense(inputs=vec_avg, units = 2048, activation=tf.nn.relu)
    drop1  = tf.layers.dropout(inputs=dense1, rate=0.4, training=isTraining)

    dense2 = tf.layers.dense(inputs=dense1, units = 1024, activation=tf.nn.relu)
    drop2  = tf.layers.dropout(inputs=dense2, rate=0.4, training=isTraining)

    dense3 = tf.layers.dense(inputs=dense2, units = 1024, activation=tf.nn.relu)
    drop3  = tf.layers.dropout(inputs=dense3, rate=0.4, training=isTraining)

    logits = tf.layers.dense(inputs=dense3, units=len(PARAMS['categories']))
    
    return logits



def simple_lstm(input_vecs, input_lengths, input_term_scores, isTraining, PARAMS):

    drop_in = tf.layers.dropout(input_vecs,
                                noise_shape=[tf.shape(input_vecs)[0], tf.shape(input_vecs)[1], 1],
                                rate=0.3, training=isTraining)

    rnn_cells    = tf.contrib.rnn.LSTMCell(num_units=80, use_peepholes=True)    
    outputs, states = tf.nn.dynamic_rnn(cell=rnn_cells,
                                        inputs=drop_in,
                                        dtype=tf.float32)
    
    batch_range = tf.range(tf.shape(input_lengths)[0])
    batch_outs  = tf.stack([batch_range, input_lengths], axis=1)
    rnn_out     = tf.gather_nd(outputs, batch_outs)
    
    logits = tf.layers.dense(rnn_out, units=len(PARAMS['categories']))

    return logits


def stacked_lstm(input_vecs, input_lengths, input_term_scores, isTraining, PARAMS):

    drop_in = tf.layers.dropout(input_vecs,
                                noise_shape=[tf.shape(input_vecs)[0], tf.shape(input_vecs)[1], 1],
                                rate=0.3, training=isTraining)
    
    layers       = [80, 80]
    rnn_cells    = [tf.contrib.rnn.LSTMCell(num_units=n, use_peepholes=True) for n in layers]
    stacked_cell = tf.contrib.rnn.MultiRNNCell(rnn_cells)
    outputs, states = tf.nn.dynamic_rnn(cell=stacked_cell,
                                        inputs=drop_in,
                                        dtype=tf.float32)
    
    batch_range = tf.range(tf.shape(input_lengths)[0])
    batch_outs  = tf.stack([batch_range, input_lengths], axis=1)
    rnn_out     = tf.gather_nd(outputs, batch_outs)
    
    logits = tf.layers.dense(rnn_out, units=len(PARAMS['categories']))

    return logits
    

def stacked_bidir_lstm(input_vecs, input_lengths, input_term_scores, isTraining, PARAMS):

    drop_in = tf.layers.dropout(input_vecs,
                                noise_shape=[tf.shape(input_vecs)[0], tf.shape(input_vecs)[1], 1],
                                rate=PARAMS['modelParams']['droprate'], training=isTraining)
    
    layers       = [PARAMS['modelParams']['fwcells'], PARAMS['modelParams']['bwcells']]
    rnn_cells    = [tf.contrib.rnn.LSTMCell(num_units=n, use_peepholes=True) for n in layers]    
    fw_cells = tf.contrib.rnn.MultiRNNCell(rnn_cells)
    bw_cells = tf.contrib.rnn.MultiRNNCell(rnn_cells)        
    outputs, states = tf.nn.bidirectional_dynamic_rnn(fw_cells, bw_cells, drop_in, dtype=tf.float32)        

    batch_range = tf.range(tf.shape(input_lengths)[0])
    batch_outs  = tf.stack([batch_range, input_lengths], axis=1)

    fw_out     = tf.gather_nd(outputs[0], batch_outs)
    bw_out     = tf.gather_nd(outputs[1], batch_outs)        
    rnn_out    = tf.concat([fw_out, bw_out], 1)

    dense1 = tf.layers.dense(rnn_out, units=1024, activation=tf.nn.relu)
    dense2 = tf.layers.dense(dense1, units=1024, activation=tf.nn.relu)        
    logits = tf.layers.dense(dense2, units=len(PARAMS['categories']))

    return logits


def bidir_gru(input_vecs, input_lengths, input_term_scores, isTraining, PARAMS):
    fw_cells = tf.contrib.rnn.GRUBlockCellV2(PARAMS['modelParams']['fwcells'])    
    bw_cells = tf.contrib.rnn.GRUBlockCellV2(PARAMS['modelParams']['bwcells'])    

    drop_in = tf.layers.dropout(input_vecs,
                                noise_shape=[tf.shape(input_vecs)[0], tf.shape(input_vecs)[1], 1],
                                rate=PARAMS['modelParams']['droprate'], training=isTraining)
    outputs, states = tf.nn.bidirectional_dynamic_rnn(fw_cells, bw_cells, drop_in,
                                                      dtype=tf.float32)        
    
    batch_range = tf.range(tf.shape(input_lengths)[0])
    batch_outs  = tf.stack([batch_range, input_lengths], axis=1)
    
    fw_out     = tf.gather_nd(outputs[0], batch_outs)
    bw_out     = tf.gather_nd(outputs[1], batch_outs)        
    rnn_out    = tf.concat([fw_out, bw_out], 1)
        
    drop1  = tf.layers.dropout(rnn_out, rate=0.5, training=isTraining)
    dense1 = tf.layers.dense(drop1, units=1024, activation=tf.nn.relu)
    logits = tf.layers.dense(dense1, units=len(PARAMS['categories']))

    return logits
    

def bidir_gru_pooled(input_vecs, input_lengths, input_term_scores, isTraining, PARAMS):
    fw_cells = tf.contrib.rnn.GRUBlockCellV2(PARAMS['modelParams']['fwcells'])    
    bw_cells = tf.contrib.rnn.GRUBlockCellV2(PARAMS['modelParams']['bwcells'])    

    drop_in = tf.layers.dropout(input_vecs,
                                noise_shape=[tf.shape(input_vecs)[0], tf.shape(input_vecs)[1], 1],
                                rate=PARAMS['modelParams']['droprate'], training=isTraining)

    outputs, states = tf.nn.bidirectional_dynamic_rnn(fw_cells, bw_cells, drop_in, dtype=tf.float32)        

    stacked_outputs =  tf.concat([outputs[0],outputs[1]], 2)

    max_pool = tf.layers.max_pooling1d(stacked_outputs, PARAMS['maxwords'], 1)
    avg_pool = tf.layers.average_pooling1d(stacked_outputs, PARAMS['maxwords'], 1)

    stacked_pool = tf.concat([max_pool, avg_pool], 2)
    rnn_out      = tf.squeeze(stacked_pool, 1)

    # Mix in the max of the term scores
    combo = tf.concat([rnn_out, tf.reduce_max(input_term_scores,axis=1)], axis=1)
    
    logits = tf.layers.dense(combo, units=len(PARAMS['categories']), activation=None)

    return logits


MODELS = {
    'avgvec': avgvec,
    'conv1d_termscores': conv1d_termscores,
    'simple_lstm': simple_lstm,
    'stacked_bidir_lstm': stacked_bidir_lstm,
    'bidir_gru': bidir_gru,
    'bidir_gru_pooled': bidir_gru_pooled
    }
