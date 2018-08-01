import tensorflow as tf


def multilstm_model(x, y, is_training, batch_size, hidden_size=32, layer_num=2, start_lr=0.1):

    cell = tf.nn.rnn_cell.MultiRNNCell([
        tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(hidden_size), input_keep_prob=0.9)
        for _ in range(layer_num)])
    init_state = cell.zero_state(batch_size, dtype=tf.float32)
    outputs, final_state = tf.nn.dynamic_rnn(cell, x, initial_state=init_state, dtype=tf.float32)
    output = outputs[:, -1, :]

    w = tf.Variable(tf.random_normal([hidden_size, 4]))
    b = tf.Variable(tf.constant(0.1, shape=[4, ]))
    predictions = tf.matmul(output, w) + b

    if not is_training:
        return predictions, None, None

    loss = tf.reduce_mean(tf.square(y-predictions))

    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(start_lr, global_step, 5000, 0.9, staircase=True)

    train_op = tf.contrib.layers.optimize_loss(loss, tf.train.get_global_step(),
                                               optimizer="Adagrad", learning_rate=learning_rate)
    return predictions, loss, train_op


def bilstm_model(x, y, is_training, batch_size, hidden_size=32, layer_num=2, start_lr=0.1):

    fw_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
    bw_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)

    fw_init_state = fw_cell.zero_state(batch_size, dtype=tf.float32)
    bw_init_state = bw_cell.zero_state(batch_size, dtype=tf.float32)

    outputs, final_state = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, x, initial_state_fw=fw_init_state,
                                                           initial_state_bw=bw_init_state, dtype=tf.float32)
    output = tf.concat(outputs, 2)
    output = output[:, -1, :]

    w = tf.Variable(tf.random_normal([2*hidden_size, 1]))
    b = tf.Variable(tf.constant(0.1, shape=[1, ]))
    predictions = tf.matmul(output, w) + b

    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(start_lr, global_step, 5000, 0.9, staircase=True)

    if not is_training:
        return predictions, None, None

    loss = tf.reduce_mean(tf.square(y-predictions))

    train_op = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)

    return predictions, loss, train_op


def multi_bilstm_model(x, y, is_training, batch_size, hidden_size=32, layer_num=2, start_lr=0.1):
    fw_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(hidden_size) for _ in range(layer_num)])
    bw_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(hidden_size) for _ in range(layer_num)])

    fw_init_state = fw_cell.zero_state(batch_size, dtype=tf.float32)
    bw_init_state = bw_cell.zero_state(batch_size, dtype=tf.float32)

    outputs, final_state = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, x, initial_state_fw=fw_init_state,
                                                           initial_state_bw=bw_init_state, dtype=tf.float32)
    output = tf.concat(outputs, 2)
    output = output[:, -1, :]

    w = tf.Variable(tf.random_normal([2 * hidden_size, 4]))
    b = tf.Variable(tf.constant([0.1], shape=[4, ]))
    predictions = tf.matmul(output, w) + b

    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(start_lr, global_step, 5000, 0.9, staircase=True)

    if not is_training:
        return predictions, None, None

    loss = tf.reduce_mean(tf.square(y - predictions))

    train_op = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)

    return predictions, loss, train_op