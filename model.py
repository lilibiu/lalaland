import tensorflow as tf


def lstm_model(x, y, is_training, batch_size, hidden_size=32, layer_num=2, learning_rate=0.1):

    cell = tf.nn.rnn_cell.MultiRNNCell([
        tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
        for _ in range(layer_num)])
    init_state = cell.zero_state(batch_size, dtype=tf.float32)
    outputs, final_state = tf.nn.dynamic_rnn(cell, x, initial_state=init_state, dtype=tf.float32)
    output = outputs[:, -1, :]

    w = tf.Variable(tf.random_normal([hidden_size, 1]))
    b = tf.Variable(tf.constant(0.1, shape=[1, ]))
    predictions = tf.matmul(output, w) + b

    if not is_training:
        return predictions, None, None

    loss = tf.reduce_mean(tf.square(y-predictions))

    train_op = tf.contrib.layers.optimize_loss(loss, tf.train.get_global_step(),
                                               optimizer="Adagrad", learning_rate=learning_rate)
    return predictions, loss, train_op
