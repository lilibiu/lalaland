import os
import numpy as np
import tensorflow as tf
from model import multi_bilstm_model
from preprogress import generate_data
from matplotlib import pyplot as plt

tf.app.flags.DEFINE_integer("time_step", 30, "input size")
tf.app.flags.DEFINE_integer("hidden_size", 32, "hidden layer size")
tf.app.flags.DEFINE_integer("layer_num", 2, "hidden layer num")

tf.app.flags.DEFINE_integer("learning_rate", 0.1, "learning rate")
tf.app.flags.DEFINE_integer("training_step", 1000, "training steps")

tf.app.flags.DEFINE_string("model_path", os.path.abspath("./model"), "save model to this path")
tf.app.flags.DEFINE_string("data_path", os.path.abspath("./data/SH000001_2_test.csv"), "test set path")
FLAGS = tf.flags.FLAGS


def main(_):
    test_x, test_y, data_mean, data_std = generate_data(FLAGS.data_path, FLAGS.time_step)
    ds = tf.contrib.data.Dataset.from_tensor_slices((test_x, test_y))
    ds = ds.batch(1)
    x, y = ds.make_one_shot_iterator().get_next()

    with tf.variable_scope("model"):
        prediction, _, _ = multi_bilstm_model(x, [0.0, 0.0, 0.0, 0.0], False, batch_size=1, hidden_size=FLAGS.hidden_size,
                                              layer_num=FLAGS.layer_num, start_lr=FLAGS.learning_rate)

    predictions = []
    y_ = []
    saver = tf.train.Saver()

    with tf.Session() as sess:
        checkpoint = tf.train.latest_checkpoint(FLAGS.model_path)
        saver.restore(sess, checkpoint)

        for i in range(len(test_x)):
            p, t = sess.run([prediction, y])
            predictions.append(p)
            y_.append(t)

    predictions = np.array(predictions).squeeze()
    y_ = np.array(y_).squeeze()
    predictions_recover = (predictions * data_std) + data_mean
    y_recover = (y_ * data_std) + data_mean

    mse_n = np.average((predictions-y_)**2)
    mse = np.average((predictions_recover-y_recover)**2)
    print("mean square error of normalized data is %f" % mse_n)
    print("mean square error is %f" % mse)

    plt.figure()
    plt.plot(predictions_recover[:, 0], label='close prediction')
    plt.plot(y_recover[:, 0], label='close real_data')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    tf.app.run()