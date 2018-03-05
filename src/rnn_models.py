import tensorflow as tf

def stacked_unidir_lstm(input_vecs, input_lengths, isTraining, PARAMS):
    layers       = [80, 80]
    rnn_cells    = [tf.contrib.rnn.LSTMCell(num_units=n, use_peepholes=True) for n in layers]
    stacked_cell = tf.contrib.rnn.MultiRNNCell(rnn_cells)
    outputs, states = tf.nn.dynamic_rnn(cell=stacked_cell,
                                        inputs=input_vecs,
                                        dtype=tf.float32)        
    batch_range = tf.range(tf.shape(input_lengths)[0])
    batch_outs  = tf.stack([batch_range, input_lengths], axis=1)
    rnn_out     = tf.gather_nd(outputs, batch_outs)

    dense1 = tf.layers.dense(rnn_out, units=1024, activation=tf.nn.relu)
    dense2 = tf.layers.dense(dense1, units=1024, activation=tf.nn.relu)        
    logits = tf.layers.dense(dense2, units=len(PARAMS['categories']))

    return logits
    

def stacked_bidir_lstm(input_vecs, input_lengths, isTraining, PARAMS):
    layers       = [80, 80]
    rnn_cells    = [tf.contrib.rnn.LSTMCell(num_units=n, use_peepholes=True) for n in layers]    
    fw_cells = tf.contrib.rnn.MultiRNNCell(rnn_cells)
    bw_cells = tf.contrib.rnn.MultiRNNCell(rnn_cells)        
    outputs, states = tf.nn.bidirectional_dynamic_rnn(fw_cells, bw_cells, input_vecs, dtype=tf.float32)        

    batch_range = tf.range(tf.shape(input_lengths)[0])
    batch_outs  = tf.stack([batch_range, input_lengths], axis=1)

    fw_out     = tf.gather_nd(outputs[0], batch_outs)
    bw_out     = tf.gather_nd(outputs[1], batch_outs)        
    rnn_out    = tf.concat([fw_out, bw_out], 1)

    dense1 = tf.layers.dense(rnn_out, units=1024, activation=tf.nn.relu)
    dense2 = tf.layers.dense(dense1, units=1024, activation=tf.nn.relu)        
    logits = tf.layers.dense(dense2, units=len(PARAMS['categories']))

    return logits


def bidir_gru(input_vecs, input_lengths, isTraining, PARAMS):
    fw_cells = tf.contrib.rnn.GRUBlockCellV2(80)    
    bw_cells = tf.contrib.rnn.GRUBlockCellV2(80)    

    drop_in = tf.layers.dropout(input_vecs,
                                noise_shape=[tf.shape(input_vecs)[0], tf.shape(input_vecs)[1], 1],
                                rate=0.2, training=isTraining)
    outputs, states = tf.nn.bidirectional_dynamic_rnn(fw_cells, bw_cells, drop_in, dtype=tf.float32)        

    #######
    fw_outputs = outputs[0]
    bw_outputs = outputs[1]
    
    fw_outputs = tf.expand_dims(fw_outputs, axis=0)
    bw_outputs = tf.expand_dims(bw_outputs, axis=0)    

    fw_avg_pool = tf.nn.max_pool(fw_outputs,
                                 ksize=[1,1,PARAMS['maxwords'],1],strides=[1,1,1,1],
                                 padding='SAME')
    bw_avg_pool = tf.nn.max_pool(bw_outputs,
                                 ksize=[1,1,PARAMS['maxwords'],1],strides=[1,1,1,1],
                                 padding='SAME')
    
    rnn_out  = tf.squeeze(avg_pool, axis=0)

    #######

    
    batch_range = tf.range(tf.shape(input_lengths)[0])
    batch_outs  = tf.stack([batch_range, input_lengths], axis=1)

    
    fw_out     = tf.gather_nd(outputs[0], batch_outs)
    bw_out     = tf.gather_nd(outputs[1], batch_outs)        
    rnn_out    = tf.concat([fw_out, bw_out], 1)

    rnn_out  = tf.expand_dims(rnn_out, axis=0)
    avg_pool = tf.nn.max_pool(rnn_out,
                              ksize=[1,1,PARAMS['maxwords'],1],strides=[1,1,1,1],
                              padding='SAME')
    rnn_out  = tf.squeeze(avg_pool, axis=0)
        
    drop1  = tf.layers.dropout(rnn_out, rate=0.5, training=isTraining)
    dense1 = tf.layers.dense(drop1, units=1024, activation=tf.nn.relu)
    logits = tf.layers.dense(dense1, units=len(PARAMS['categories']))

    return logits
