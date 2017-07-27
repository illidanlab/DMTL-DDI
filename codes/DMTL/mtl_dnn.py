
import tensorflow as tf
import os
import random
import math
from confusion_metrics import tf_confusion_metrics

FLAGS = tf.app.flags.FLAGS


# The DDI dataset has 10 classes, representing the digits 0 through 9.
NUM_CLASSES = 1


def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def activation_function_convd(features_tasks, weights, biases, activation_func):
    '''Set different activation functions'''


    if activation_func == 0:
        hiddens = tf.tanh(tf.matmul(features_tasks, weights) + biases)  # task 1 output of layer 1
    elif activation_func == 1:
        hiddens = tf.sigmoid(tf.matmul(features_tasks, weights) + biases)  # task 1 output of layer 1
    elif activation_func == 2:
        hiddens = tf.nn.relu(tf.matmul(features_tasks, weights) + biases)
    else:
        h_conv1 = tf.nn.relu(conv2d(features_tasks, weights) + biases)
        h_pool1 = max_pool_2x2(h_conv1)
        hiddens = h_pool1

        print "Wrong choice for activation function: use default relu"

    return hiddens


def activation_function(features_tasks, weights, biases, activation_func):
    '''Set different activation functions'''


    if activation_func == 0:
        hiddens = tf.tanh(tf.matmul(features_tasks, weights) + biases)  # task 1 output of layer 1
    elif activation_func == 1:
        hiddens = tf.sigmoid(tf.matmul(features_tasks, weights) + biases)  # task 1 output of layer 1
    elif activation_func == 2:
        hiddens = tf.nn.relu(tf.matmul(features_tasks, weights) + biases)
    else:
        hiddens = tf.nn.relu(tf.matmul(features_tasks, weights) + biases)
        print "Wrong choice for activation function: use default relu"

    return hiddens

def inference(placeholder_tuple, hidden_units, hidden_units_task, FEATURE_SIZE):
    """Build the multi layer neural network model up to where it may be used for inference.

          Args:
            features_tasks: a list of placeholders, features_tasks[0] for first task.
            labels_tasks:
            dropout_tasks:
            hidden1_units: Size of the first hidden layer.
            hidden2_units: Size of the second hidden layer.
          Returns:
            sigmoid: Output tensor with the computed logits.
    """
    regularizers = 0
    regularizers_l1 = 0
    features_tasks, labels_tasks, dropout_tasks = placeholder_tuple
    K = len(labels_tasks) # number of tasks

    L = len(hidden_units)  # number of layers
    hiddens = [None] * L   # hidden layers outputs
    weights = [None] * L
    biases = [None] * L
    activation_func = FLAGS.activation_func

    '''Shared layers: same weights for all tasks'''
    for ll in range(L):

        if ll == 0:
            '''The first input layers'''
            # Hidden 1
            with tf.name_scope('hidden' + str(ll + 1)):
                weights[ll] = tf.Variable(
                    tf.truncated_normal([FEATURE_SIZE/2, hidden_units[ll]],
                                        stddev=1.0 / math.sqrt(float(FEATURE_SIZE))),
                    name='weights')
                biases[ll] = tf.Variable(tf.zeros([hidden_units[ll]]),
                                      name='biases')


                hidden1 = [None] * K  # output value of layer 1 for K tasks
                for kk in range(K):
                    # hidden1[kk] = tf.nn.dropout(activation_function(features_tasks[kk], weights[ll], biases[ll], activation_func), FLAGS.dropout_rate_L1)
                    weight_layer1 = tf.concat(0, [weights[ll], weights[ll]])
                    hidden1[kk] = activation_function(features_tasks[kk], weight_layer1, biases[ll], activation_func)


                hiddens[ll] = hidden1

                regularizers_l1 += tf.reduce_sum(tf.abs(weights[ll]))
                regularizers += tf.nn.l2_loss(weights[ll]) + tf.nn.l2_loss(biases[ll])
                tf.histogram_summary("Layer" + str(ll + 1) + '/weights', weights[ll])

        elif ll >= 1:

            '''Intermediate layers from layer 2 to Layer L '''
            with tf.name_scope('hidden' + str(ll + 1)):
                weights[ll] = tf.Variable(
                    tf.truncated_normal([hidden_units[ll-1], hidden_units[ll]],
                                        stddev=1.0 / math.sqrt(hidden_units[ll-1])), name='weights')

                biases[ll] = tf.Variable(tf.zeros([hidden_units[ll]]), name='biases')

                hidden_l = [None] * K  #output value of layer 1 for K tasks input is different although the parameters shared
                for kk in range(K):
                    hidden_l[kk] = activation_function(hiddens[ll - 1][kk], weights[ll], biases[ll], activation_func)

                hiddens[ll] = hidden_l
                regularizers_l1 += tf.reduce_sum(tf.abs(weights[ll]))
                regularizers += tf.nn.l2_loss(weights[ll]) + tf.nn.l2_loss(biases[ll])
                tf.histogram_summary("Layer" + str(ll + 1) + '/weights', weights[ll])

    L_task = len(hidden_units_task)
    weights_t_l = [None] * L_task
    biases_t_l = [None] * L_task
    hidden_tasks = [None] * L_task

    for ll in range(L_task):
        if ll == 0:
            hidden_tasks_temp = [None] * K
            weights_t_l_temp = [None] * K
            biases_t_l_temp = [None] * K
            for kk in range(K):
                with tf.name_scope('tasks_hidden_layers' + str(ll) + "_" + str(kk)):
                    weights_t_l_temp[kk] = tf.Variable(
                        tf.truncated_normal([hidden_units[L-1], hidden_units_task[ll][kk]],
                                            stddev=1.0 / math.sqrt(float(hidden_units[L-1]))),
                        name='weights')
                    biases_t_l_temp[kk] = tf.Variable(tf.zeros([hidden_units_task[ll][kk]]),
                                         name='biases')

                    hidden_tasks_temp[kk] = activation_function(hiddens[L-1][kk], weights_t_l_temp[kk], biases_t_l_temp[kk], activation_func)

                    tf.histogram_summary("taskLayer" + str(ll+1) + '/weights' + str(kk), weights_t_l_temp[kk])
            hidden_tasks[ll] = hidden_tasks_temp
            weights_t_l[ll] = weights_t_l_temp
            biases_t_l[ll] = biases_t_l_temp
        else:
            hidden_tasks_temp = [None] * K
            weights_t_l_temp = [None] * K
            biases_t_l_temp = [None] * K
            for kk in range(K):
                with tf.name_scope('tasks_hidden_layers' + str(ll) + "_" + str(kk)):
                    weights_t_l_temp[kk] = tf.Variable(
                        tf.truncated_normal([hidden_units_task[ll-1][kk], hidden_units_task[ll][kk]],
                                            stddev=1.0 / math.sqrt(float(hidden_units_task[ll-1][kk]))),
                        name='weights')
                    biases_t_l_temp[kk] = tf.Variable(tf.zeros([hidden_units_task[ll][kk]]),
                                                 name='biases')

                    hidden_tasks_temp[kk] = activation_function(hidden_tasks[ll-1][kk], weights_t_l_temp[kk],
                                                                biases_t_l_temp[kk],
                                                                activation_func)

                    tf.histogram_summary("taskLayer" + str(ll + 1) + '/weights' + str(kk), weights_t_l_temp[kk])

            hidden_tasks[ll] = hidden_tasks_temp
            weights_t_l[ll] = weights_t_l_temp
            biases_t_l[ll] = biases_t_l_temp


    # hidden_last = hiddens[L-1]

    '''Task specific parameters: '''
    # Linear
    logits = [None] * K
    weights_t = [None] * K
    biases_t = [None] * K
    linear_pred_t = [None] * K
    hidden_drop_t = [None] * K
    regularizers_t = [None] * K
    for kk in range(K):
        with tf.name_scope('softmax_linear_task' + str(kk)):
            weights_t[kk] = tf.Variable(
                tf.truncated_normal([hidden_units_task[L_task-1][kk], NUM_CLASSES],
                                    stddev=1.0 / math.sqrt(float(hidden_units_task[L_task-1][kk]))),
                                    name='weights')
            biases_t[kk] = tf.Variable(tf.zeros([NUM_CLASSES]),
                                 name='biases')
            tf.histogram_summary("Layer4" + '/weights' + str(kk), weights_t[kk])

            # add dropout
            hidden_drop_t[kk] = tf.nn.dropout(hidden_tasks[L_task - 1][kk], dropout_tasks[kk])

            # Calculate log(1 + e^{- (x'w + bias)})
            linear_pred_t[kk] = tf.matmul(hidden_drop_t[kk], weights_t[kk]) + biases_t[kk]
            aa = -tf.mul(labels_tasks[kk], linear_pred_t[kk])  # labels : (100,) ytemp: 100x 100   y_i*(x_i*w + bias)
            bb = tf.nn.relu(aa)
            logits[kk] = tf.log(tf.exp(-bb) + tf.exp(aa - bb)) + bb  # = log(1 + e^{- (x'w + bias)})

            regularizers_t[kk] = tf.nn.l2_loss(weights_t[kk]) + tf.nn.l2_loss(biases_t[kk]) + regularizers
            # regularizers_l1 += tf.reduce_sum(tf.abs(weights_t[kk]))

    # var_list1 = []
    # for ii in FLAGS.Layer_last_restore:
    #     var_list1.append(weights[ii])
    #     var_list1.append(biases[ii])

    # task shared parameters
    L_hidden_vars = [weights[ll] for ll in range(L)] + [biases[ll] for ll in range(L)]

    task_hidden_vars = [None] * K

    for kk in range(K):
        task_hidden_vars[kk] = [weights_t[kk], biases_t[kk]] # softmax layer
        for ll in range(L_task):
            task_hidden_vars[kk] += [weights_t_l[ll][kk], biases_t_l[ll][kk]] # task specific layer


    return logits, regularizers_t, linear_pred_t, L_hidden_vars, task_hidden_vars, regularizers_l1

def loss(logits, regularizers_t, regularizers_l1, l2_reg_para_t, l1_reg_para):
    """Calculates the loss from the logits and the labels.
    Args:
        logits: Logits tensor, float - [batch_size, NUM_CLASSES].
        labels: Labels tensor, int32 - [batch_size].
        Returns:
        loss: Loss tensor of type float.
    """
    K = len(logits)
    reg_l1_val = regularizers_l1 * l1_reg_para
    loss = 0
    # joint loss
    for kk in range(K):
        loss += tf.reduce_mean(logits[kk])
    loss += reg_l1_val
    tf.scalar_summary("logistic loss", loss)

    regularization = sum([regularizers_t[kk] * l2_reg_para_t[kk] for kk in range(K)]) # regularization term for K tasks
    tf.scalar_summary("regularization value ", regularization)

    obj_value = loss + regularization + reg_l1_val
    tf.scalar_summary("objective function value ", obj_value)

    # sperate objective value
    obj_value_t = [tf.reduce_mean(logits[kk]) + regularizers_t[kk] * l2_reg_para_t[kk] for kk in range(K)]

    return loss, regularization, obj_value, obj_value_t, reg_l1_val

def training(obj_value, learning_rate1, learning_rate_t, L_hidden_vars, task_hidden_vars):
    """Sets up the training Ops.
    Creates a summarizer to track the loss over time in TensorBoard.
    Creates an optimizer and applies the gradients to all trainable variables.
    The Op returned by this function is what must be passed to the
    `sess.run()` call to cause the model to train.
    Args:
    loss: Loss tensor, from loss().
    learning_rate1: The learning rate for first hidden layer, comes from auto encoder.

    Returns:
    train_op: The Op for training.
    """

    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # optimizer1 = tf.train.GradientDescentOptimizer(learning_rate1)

    # learning_rate1_exp = tf.train.exponential_decay(learning_rate1, global_step,
    #                                            1e+4, 0.96, staircase=True)

    optimizer1 = tf.train.AdamOptimizer(learning_rate1)
    train_op1 = optimizer1.minimize(obj_value, global_step=global_step, var_list=L_hidden_vars)

    K = len(learning_rate_t)

    # learning_rate_t_exp = [None] * K
    '''Optimizer for k tasks'''
    optimizer_task = [None] * K
    for kk in range(K):
        # learning_rate_t_exp[kk] = tf.train.exponential_decay(learning_rate_t[kk], global_step,
        #                            1e+4, 0.96, staircase=True)
        optimizer_task[kk] = tf.train.AdamOptimizer(learning_rate_t[kk])
        # optimizer_task[kk] = tf.train.GradientDescentOptimizer(learning_rate_t[kk])
    train_op_tasks = [None] * K

    for kk in range(K):
        train_op_tasks[kk] = optimizer_task[kk].minimize(obj_value, global_step=global_step, var_list=task_hidden_vars[kk])

    train_op = train_op1
    for kk in range(K):
        train_op = tf.group(train_op, train_op_tasks[kk])
    # train_op = tf.group(train_op1, train_op_tasks)   # TODO
    return train_op, optimizer_task

def evaluation(linear_pred, y_true):
    """Evaluate the quality of the logits at predicting the label.
    Args:
    y_pred: Logits tensor, float - [batch_size, NUM_CLASSES].
    y_true: Labels tensor, int32 - [batch_size], with values in the
      range [0, 1].
    Returns:
    A float the represents the prediction accuracy
    """
    K = len(linear_pred)
    eval_correct_t = [0] * K
    y_pred = [None] * K
    label_pred_t = [None] * K
    label_true_t = [None] * K
    correct_prediction = [None] * K
    tp = [0] * K  # true positive
    tn = [0] * K  # true negative
    fp = [0] * K  # false positive
    fn = [0] * K  # false negative
    f1_score = [0] * K   # f1 score.

    for kk in range(K):
        y_pred[kk] = tf.sigmoid(linear_pred[kk]) # linear_pred (100, 1)    # 1/{1 + exp{-(x'w + bias)}}
        label_pred_t[kk] = tf.cast(tf.round(y_pred[kk]), tf.int64)   # checked
        label_true_t[kk] = tf.cast(tf.nn.relu(y_true[kk]), tf.int64)
        correct_prediction[kk] = tf.equal(label_pred_t[kk], label_true_t[kk])

        eval_correct_t[kk] = tf.reduce_sum(tf.cast(correct_prediction[kk], tf.float32))

        tp[kk], tn[kk], fp[kk], fn[kk], f1_score[kk] = tf_confusion_metrics(label_pred_t[kk], label_true_t[kk])

    # Return the number of true entries.
    return eval_correct_t, y_pred, label_true_t, tuple([tp, tn, fp, fn])         # TODO debug





