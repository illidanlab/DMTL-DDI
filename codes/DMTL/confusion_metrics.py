# from https://cloud.google.com/solutions/machine-learning-with-financial-time-series-data
import tensorflow as tf

def tf_confusion_metrics(actuals, predictions):
    '''
    :param actuals:
    :param predictions:
    :return:
    '''
    ones_like_actuals = tf.ones_like(actuals)
    zeros_like_actuals = tf.zeros_like(actuals)
    ones_like_predictions = tf.ones_like(predictions)
    zeros_like_predictions = tf.zeros_like(predictions)

    tp_op = tf.reduce_sum(
    tf.cast(
      tf.logical_and(
        tf.equal(actuals, ones_like_actuals),
        tf.equal(predictions, ones_like_predictions)
      ),
      "float"
    )
    )

    tn_op = tf.reduce_sum(
    tf.cast(
      tf.logical_and(
        tf.equal(actuals, zeros_like_actuals),
        tf.equal(predictions, zeros_like_predictions)
      ),
      "float"
    )
    )

    fp_op = tf.reduce_sum(
    tf.cast(
      tf.logical_and(
        tf.equal(actuals, zeros_like_actuals),
        tf.equal(predictions, ones_like_predictions)
      ),
      "float"
    )
    )

    fn_op = tf.reduce_sum(
    tf.cast(
      tf.logical_and(
        tf.equal(actuals, ones_like_actuals),
        tf.equal(predictions, zeros_like_predictions)
      ),
      "float"
    )
    )
    precision = tp_op / (tp_op + fp_op)
    recall = tp_op / (tp_op + fn_op)

    f1_score = (2 * (precision * recall)) / (precision + recall)

    return tp_op, tn_op, fp_op, fn_op, f1_score
