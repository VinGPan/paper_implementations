
import tensorflow as tf
from keras.backend.common import epsilon


class WeightedBinaryLoss:
    def __init__(self, w_class0, w_class1):
        self.w_class0 = w_class0
        self.w_class1 = w_class1

    def compute_loss(self, y_true, y_pred):
        _epsilon = tf.convert_to_tensor(epsilon(), dtype=y_pred.dtype.base_dtype)
        y_pred = tf.clip_by_value(y_pred, _epsilon, 1 - _epsilon)
        loss = -self.w_class1 * y_true * tf.log(y_pred) \
               -self.w_class0 * (1 - y_true) * tf.log(1 - y_pred)
        return tf.reduce_mean(loss)


if __name__ == '__main__':
    import numpy as np
    y_true = np.array([[0.], [0.], [1.], [1.]])
    y_pred = np.array([[0.], [1.], [0.], [1.]])

    y_true = tf.convert_to_tensor(y_true)
    y_pred = tf.convert_to_tensor(y_pred)

    wloss = WeightedBinaryLoss(0.75, 0.25)
    wloss = wloss.compute_loss(y_true, y_pred)
    with tf.Session():
        print(wloss.eval())

