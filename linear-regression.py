import tensorflow as tf
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

def load_data():
    data, targets = load_boston(True)
    data = normalize(data)
    targets = targets.reshape((targets.shape[0], 1))
    x_train, x_test, y_train, y_test = train_test_split(data, targets, test_size=0.3, random_state=42)
    return x_train, x_test, y_train, y_test


if __name__ == '__main__':
    x_train, x_test, y_train, y_test = load_data()
    features = tf.placeholder(tf.float32, shape=[None, 13])
    targets = tf.placeholder(tf.float32, shape=[None, 1])
    weights = tf.Variable(tf.constant(0.1, shape=[13, 1]))
    bias = tf.constant(0.1)
    predicted = tf.matmul(features, weights) + bias
    loss = tf.reduce_mean(tf.square(predicted - targets))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)
    init = tf.global_variables_initializer()
    sess = tf.InteractiveSession()
    sess.run(init)
    for epoch in range(5000):
        sess.run(optimizer, feed_dict={features: x_train, targets: y_train})
        accuracy = loss.eval(feed_dict={features: x_train, targets: y_train})
        if epoch % 100 == 0:
            print('current loss: {}'.format(accuracy))

    testing_predictions = sess.run(predicted, feed_dict={features: x_test})
    testing_predictions = testing_predictions.flatten()
    y_test = y_test.reshape((152))

    for sample in range(20):
        rand = np.random.randint(0, 152)
        prediction = testing_predictions[rand]
        actual = y_test[rand]
        print('prediction: {}, actual: {}'.format(prediction, actual))