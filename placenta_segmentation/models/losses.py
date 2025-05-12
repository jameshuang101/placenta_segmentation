"""
Dice coefficient, dice loss, and custom combined loss.
"""
import tensorflow as tf
from tensorflow.keras import backend as K

def dice_coef(y_true, y_pred, smooth=1e-9):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

def combined_loss(y_true, y_pred):
    """
    Weighted combination for placenta (class=1) and uterus (class=2).
    """
    y_true = tf.reshape(y_true, [-1, 3])
    y_pred = tf.reshape(y_pred, [-1, 3])
    p_true, u_true = y_true[:,1], y_true[:,2]
    p_pred, u_pred = y_pred[:,1], y_pred[:,2]
    dice_p = (2*tf.reduce_sum(p_true * p_pred) + 1e-9) / (tf.reduce_sum(p_true) + tf.reduce_sum(p_pred) + 1e-9)
    dice_u = (2*tf.reduce_sum(u_true * u_pred) + 1e-9) / (tf.reduce_sum(u_true) + tf.reduce_sum(u_pred) + 1e-9)
    # weight classes equally
    return 1.0 - 0.5 * (dice_p + dice_u)
