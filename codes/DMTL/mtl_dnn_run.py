import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import time
import sys
from scipy.signal import savgol_filter
from scipy import stats
from shutil import copyfile
import mtl_dnn as models
from mtl_dnn_configure import FLAGS
import scipy.io as sio
from sklearn import metrics
import random
current_time = FLAGS.current_time




def batch_data(x, y, batch_size):
    '''Input numpy array, and batch size'''
    data_size = len(x)
    index = np.random.permutation(data_size)
    # index = range(data_size)
    batch_index = index[:batch_size]

    batch_x = x[batch_index]
    batch_y = y[batch_index]
    return batch_x, batch_y

def placeholder_inputs(batch_size):
    """Generate placeholder variables to represent the input tensors.

    These placeholders are used as inputs by the rest of the model building
    code and will be fed from the downloaded data in the .run() loop, below.

    Args:
    batch_size: The batch size will be baked into both placeholders.

    Returns:
    features_placeholder: features placeholder.
    labels_placeholder: Labels placeholder.
    """

    features_placeholder_t = [None] * FLAGS.K
    labels_placeholder_t = [None] * FLAGS.K
    dropout_placeholder_t = [None] * FLAGS.K
    # learning_rate1 = tf.placeholder(tf.float32)
    for kk in range(FLAGS.K):
        features_placeholder_t[kk] = tf.placeholder(tf.float32, shape=(batch_size[kk],
                                                                    FLAGS.FEATURE_SIZE))
        labels_placeholder_t[kk] = tf.placeholder(tf.float32, shape=(batch_size[kk], 1))
        dropout_placeholder_t[kk] = tf.placeholder(tf.float32)
    return features_placeholder_t, labels_placeholder_t, dropout_placeholder_t


def fill_feed_dict(train_x_t, train_y_t, batch_size_t):
    """Fills the feed_dict for training the given step.

    A feed_dict takes the form of:
    feed_dict = {
      <placeholder>: <tensor of values to be passed for placeholder>,
      ....
    }

    Args:
    data_set: The set of images and labels, from input_data.read_data_sets()
    images_pl: The images placeholder, from placeholder_inputs().
    labels_pl: The labels placeholder, from placeholder_inputs().

    Returns:
    feed_dict: The feed dictionary mapping from placeholders to values.
    """
    # Create the feed_dict for the placeholders filled with the next
    # `batch size` examples.
    K = FLAGS.K
    batch_x_t = [None] * K
    batch_y_t = [None] * K

    for kk in range(K):
        data_set_size = len(train_y_t[kk])
        if batch_size_t[kk] < len(train_y_t[kk]):
            batch_x_t[kk], batch_y_t[kk] = batch_data(train_x_t[kk], train_y_t[kk], batch_size_t[kk])
        elif batch_size_t[kk] == len(train_y_t[kk]):
            batch_x_t[kk], batch_y_t[kk] = train_x_t[kk], train_y_t[kk]
        else:
            num_repeat = batch_size_t[kk] // data_set_size  # + 1
            xtemp = train_x_t[kk]
            ytemp = train_y_t[kk]
            for i in range(num_repeat):
                xtemp = np.concatenate((xtemp, train_x_t[kk]), axis=0)
                ytemp = np.concatenate((ytemp, train_y_t[kk]), axis=0)

            batch_x_t[kk] = xtemp[:batch_size_t[kk]]
            batch_y_t[kk] = ytemp[:batch_size_t[kk]]

        batch_x_t[kk] = np.reshape(batch_x_t[kk], (batch_size_t[kk], FLAGS.FEATURE_SIZE))
        batch_y_t[kk] = np.reshape(batch_y_t[kk], (batch_size_t[kk], 1))

    return batch_x_t, batch_y_t


def fill_feed_dict_run(train_x_t, train_y_t, dropout_rate_t,
                       features_placeholder_t, labels_placeholder_t, dropout_placeholder_t):
    """Fills the feed_dict for training the given step.

    A feed_dict takes the form of:
    feed_dict = {
      <placeholder>: <tensor of values to be passed for placeholder>,
      ....
    }

    Args:
    data_set: The set of images and labels, from input_data.read_data_sets()
    images_pl: The images placeholder, from placeholder_inputs().
    labels_pl: The labels placeholder, from placeholder_inputs().

    Returns:
    feed_dict: The feed dictionary mapping from placeholders to values.
    """
    # Create the feed_dict for the placeholders filled with the next
    # `batch size` examples.
    batch_x_t, batch_y_t = fill_feed_dict(train_x_t, train_y_t, FLAGS.batch_size_t)

    feed_dict = {}

    K = FLAGS.K
    for kk in range(K):
        feed_dict[features_placeholder_t[kk]] = batch_x_t[kk]
        feed_dict[labels_placeholder_t[kk]] = batch_y_t[kk]
        feed_dict[dropout_placeholder_t[kk]] = dropout_rate_t[kk]

    return feed_dict

def do_eval(sess, f1_tuple, lable_pred_t, label_true_t,
            features_placeholder_t, labels_placeholder_t, dropout_placeholder_t,
            train_x_t, train_y_t, dropout_rate_t):
    """Runs one evaluation against the full epoch of data.

    Args:
    sess: The session in which the model has been trained.
    eval_correct: The Tensor that returns the number of correct predictions.
    features_placeholder: The images placeholder.
    labels_placeholder: The labels placeholder.
    data_set: The set of images and labels to evaluate, from
      input_data.read_data_sets().
    """
    # And run one epoch of eval.
    K = FLAGS.K
    true_count = [0] * K   # Counts the number of correct predictions.
    num_examples_t = [0] * K
    steps_per_epoch_t = [0] * K
    precision_t = [0] * K

    f1_score = [0] * K
    precision = [0] * K
    recall = [0] * K
    AUCs  = [0] * K
    tp = [0] * K  # true positive
    tn = [0] * K  # true negative
    fp = [0] * K  # false positive
    fn = [0] * K  # false negative

    data_set_size_t = [len(train_y_t[kk]) for kk in range(K)]
    batch_size_t = FLAGS.batch_size_t

    label_pred_t_all = [None] * K
    label_true_t_all = [None] * K

    for kk in range(K):
        batch_size = batch_size_t[kk]
        data_set_size = data_set_size_t[kk]
        if batch_size <= data_set_size:
            steps_per_epoch_t[kk] = data_set_size // batch_size
            num_examples_t[kk] = steps_per_epoch_t[kk] * batch_size
        else:
            num_repeat = batch_size // data_set_size  # + 1
            xtemp = train_x_t[kk]
            ytemp = train_y_t[kk]
            for i in range(num_repeat):
                xtemp = np.concatenate((xtemp, train_x_t[kk]), axis=0)
                ytemp = np.concatenate((ytemp, train_y_t[kk]), axis=0)

            train_x_t[kk] = xtemp[:batch_size]
            train_y_t[kk] = ytemp[:batch_size]
            steps_per_epoch_t[kk] = 1
            num_examples_t[kk] = batch_size

    steps_per_epoch = max(steps_per_epoch_t)

    for step in xrange(steps_per_epoch):  # TODO low efficient here: need to repeat unnecessary time for most of tasks
        feed_dict = fill_feed_dict_run(train_x_t, train_y_t, dropout_rate_t,
                                       features_placeholder_t, labels_placeholder_t, dropout_placeholder_t)

        # true_count_temp = sess.run(eval_correct_t, feed_dict=feed_dict)
        f1_tuple_val, \
        label_pred_temp, label_true_temp = sess.run([f1_tuple,
                                                    lable_pred_t,
                                                    label_true_t],
                                                    feed_dict=feed_dict)
        tp_temp, tn_temp, fp_temp, fn_temp = f1_tuple_val
        for kk in range(K):
            tp[kk] += tp_temp[kk]  # true positive
            tn[kk] += tn_temp[kk]# true negative
            fp[kk] += fp_temp[kk]  # false positive
            fn[kk] += fn_temp[kk] # false negative

            # concatenate all predicted vectors and true vectors
            if step == 0:
                label_pred_t_all[kk] = label_pred_temp[kk]
                label_true_t_all[kk] = label_true_temp[kk]

            else:
                # print label_pred_temp[kk].shape, label_pred_t_all[kk].shape
                label_pred_t_all[kk] = np.concatenate((label_pred_temp[kk], label_pred_t_all[kk]), axis=0)
                label_true_t_all[kk] = np.concatenate((label_true_temp[kk], label_true_t_all[kk]), axis=0)



    for kk in range(K):
        precision[kk] = float(tp[kk]) / (tp[kk] + fp[kk])

        recall[kk] = float(tp[kk]) / (tp[kk] + fn[kk])

        f1_score[kk] = (2 * (precision[kk] * recall[kk])) / float(precision[kk] + recall[kk])
        # calculate AUC
        fpr, tpr, thresholds = metrics.roc_curve(label_true_t_all[kk], label_pred_t_all[kk], pos_label=1)
        AUCs[kk] = metrics.auc(fpr, tpr)

    return precision, recall, f1_score, AUCs


def valid_curvesmooth(valid_list, number):

    if len(valid_list) < number:
        valid_list_smooth = valid_list
    else:
        valid_list_smooth = savgol_filter(valid_list, number, 3)  # window size number, polynomial order 3

    return valid_list_smooth

def early_stop_by_valid(valid_auc, k):
    '''Early stop"
    '''

    if k > len(valid_auc):
        return False
    print k, valid_auc

    x = range(k)
    y = valid_auc[-k:]

    # print x, y
    slop, bias = np.polyfit(x, y, 1)
    print "slop: {0:f}".format(slop)

    stopflag = False
    if slop <= FLAGS.slope:
        stopflag = True
    return stopflag



def plot_figures(data_tuple, taskid, fig_id, figname):
    '''plot figure '''
    train_accs, test_accs, valid_accs, obj_value_values = data_tuple
    fig = plt.figure(1 + fig_id)
    plt.plot(train_accs, 'g--', label='f1')
    plt.savefig(FLAGS.train_dir + current_time + '_' + taskid + '_train_' + figname + '.png')
    plt.close(fig)

    fig = plt.figure(2 + fig_id)
    plt.plot(test_accs)
    plt.savefig(FLAGS.train_dir + current_time + '_' + taskid + '_test_' + figname + '.png')
    plt.close(fig)

    fig = plt.figure(3 + fig_id)
    plt.plot(valid_accs, 'g--', label='valid_' + figname)
    plt.plot(test_accs, 'r-o', label='test_' + figname)
    plt.plot(train_accs, 'g^', label='train_' + figname)
    plt.legend(bbox_to_anchor=(0., 1.01, 1., .101), loc=2,
               ncol=3, mode="expand", borderaxespad=0.)
    plt.savefig(FLAGS.train_dir + current_time + '_' + taskid + '_valid_all_' + figname + '.png')
    plt.close(fig)

    fig = plt.figure(4 + fig_id)
    plt.plot(obj_value_values)
    plt.savefig(FLAGS.train_dir + current_time + '_' + taskid + '_obj_value_values.png')
    plt.close(fig)

    valid_accs_smooth = []
    results_saved = tuple([train_accs, valid_accs, test_accs, obj_value_values, valid_accs_smooth])
    np.save(FLAGS.train_dir + current_time + '_task' + str(taskid) + '_' + figname + '_results.npy', results_saved)

    #print stats
    max_train_acc = max(train_accs)
    print('Task: ' + taskid + ' max' + figname + ' train %f: at step = %d' % (max_train_acc, np.argmax(train_accs)))  # Only the first occurrence is returned.
    max_valid_acc = max(valid_accs)
    print('Task: ' + taskid + ' max ' + figname + ' valid %f: at step = %d' % (max_valid_acc,  np.argmax(valid_accs)))
    # max_valid_smooth_acc = max(valid_accs_smooth)
    # print('max valid smooth %f: at step = %d' % (max_valid_smooth_acc, valid_accs.index(max_valid_smooth_acc)))
    max_test_acc = max(test_accs)
    print('Task: ' + taskid + ' max ' + figname + ' test %f: at step = %d \n' % (max_test_acc, np.argmax(test_accs)))

    max_test_byvalid = test_accs[np.argmax(valid_accs)]
    print('Task: ' + taskid + ' max valid test' + figname + ' test %f: at step = %d \n' % (max_test_byvalid, np.argmax(valid_accs)))

    return max_test_byvalid, np.argmax(valid_accs)

def split_train_valid_test_byinds(x, y, train_inds, test_inds, valid_inds):
    '''

    :param x:  fetures
    :param y:  labels
    :param train_inds: training indices
    :param test_inds:
    :param valid_inds:
    :return:
    '''
    train_x = x[train_inds, :]
    train_y = y[train_inds]

    valid_x = x[valid_inds, :]
    valid_y = y[valid_inds]

    test_x = x[test_inds, :]
    test_y = y[test_inds]

    return train_x, train_y, valid_x, valid_y, test_x, test_y


def down_sampling(train_x_t_k, train_y_t_k):
    '''Sampleing the positive negative sample to balance the data, ratio 1: 1'''

    pos_num = len(train_y_t_k[train_y_t_k == 1])
    neg_num = len(train_y_t_k[train_y_t_k == -1])
    data_size = len(train_y_t_k)

    neg_inds = [i for i in range(data_size) if train_y_t_k[i] == -1]
    pos_inds = [i for i in range(data_size) if train_y_t_k[i] == 1]
    if neg_num  > pos_num:
        train_x_t_k = np.concatenate((train_x_t_k[pos_inds], train_x_t_k[neg_inds[:pos_num]]), axis=0)
        train_y_t_k = np.concatenate((train_y_t_k[pos_inds], train_y_t_k[neg_inds[:pos_num]]), axis=0)
    else:
        train_x_t_k = np.concatenate((train_x_t_k[neg_inds], train_x_t_k[pos_inds[:pos_num]]), axis=0)
        train_y_t_k = np.concatenate((train_y_t_k[neg_inds], train_y_t_k[pos_inds[:pos_num]]), axis=0)
    return train_x_t_k, train_y_t_k

def print_train_stat(train_y, test_y, valid_y):
    '''Print training testing validation set'''

    print "positive in testing: ", len(test_y[test_y==1]),\
          "test_size: ", len(test_y), 'all negative acc: ', 1 - len(test_y[test_y==1])/float(len(test_y))
    print "positive in validation: ", len(valid_y[valid_y==1]),\
          "valid_size: ", len(valid_y), 'all negative acc: ', 1 - len(valid_y[valid_y == 1]) / float(len(valid_y))
    print "positive in train: ", len(train_y[train_y == 1]),\
          "train_size: ", len(train_y), 'all negative acc: ', 1 - len(train_y[train_y == 1]) / float(len(train_y))


def select_tasks_load_inds(features_normalized, labels):
    '''select specific tasks to run from all tasks file
        spliting the data by indices loaded from mat files'''

    K = FLAGS.K
    train_x_t = [None] * K
    train_y_t = [None] * K
    valid_x_t = [None] * K
    valid_y_t = [None] * K
    test_x_t = [None] * K
    test_y_t = [None] * K

    tasks_ids = FLAGS.tasks_ids
    train_inds, test_inds, valid_inds = get_train_test_valid_inds()
    for kk in range(K):
        id = tasks_ids[kk]
        train_x_t[kk], train_y_t[kk], valid_x_t[kk], valid_y_t[kk], test_x_t[kk], test_y_t[kk]\
        = split_train_valid_test_byinds(features_normalized, labels[:, id], train_inds[kk], test_inds[kk], valid_inds[kk])

        #   TODO print
        print_train_stat(train_y_t[kk], test_y_t[kk], valid_y_t[kk])


    return tuple([train_x_t, train_y_t, valid_x_t, valid_y_t, test_x_t, test_y_t])

def check_data_valid(dataset_tuple):
    train_x, train_y, valid_x, valid_y, test_x, test_y = dataset_tuple

    if np.isnan(np.sum(train_x)) or np.isnan(np.sum(train_y)) or np.isnan(np.sum(valid_x)):
        print('\n data is no valid 1\n')
    if np.isnan(np.sum(valid_y)) or np.isnan(np.sum(test_x)) or np.isnan(np.sum(test_y)):
        print('\n data is no valid 2\n')

def record_configure_files(argv):
    file_name = argv[0].split("/")[-1]
    configure_name = "mtl_dnn_sym_earlystop_configure.py"
    copyfile(configure_name, FLAGS.train_dir + configure_name)
    copyfile(file_name, FLAGS.train_dir + file_name)
    copyfile("mtl_dnn_sym_earlystop.py", FLAGS.train_dir + "mtl_dnn_sym_earlystop.py")

def get_train_test_valid_inds():
    '''get training testing validation indices from mat files
    '''
    fname = FLAGS.input_dir + 'matlab_results/' + FLAGS.timeflag_matlab + FLAGS.input_filename_matfile
    if not os.path.isfile(fname):
        print "Wrong: mising index file"
    mat_contents = sio.loadmat(fname)
    K = FLAGS.K
    train_inds = [None] * K
    test_inds = [None] * K
    valid_inds = [None] * K

    for kk in range(K):
        index_task = FLAGS.tasks_ids_index
        indices = mat_contents['results'][index_task[kk]][0][0][-2]
        train_inds_temp, test_inds_temp, valid_inds_temp = indices[0][0]
        train_inds[kk] = np.reshape(train_inds_temp, [len(train_inds_temp)]) - 1
        test_inds[kk] = np.reshape(test_inds_temp, [len(test_inds_temp)]) - 1
        valid_inds[kk] = np.reshape(valid_inds_temp, [len(valid_inds_temp)]) - 1

    return train_inds, test_inds, valid_inds

def run_training(dataset_tuple, current_taskid):
    """Train DDI for a number of steps."""
    # Get the sets of images and labels for training, validation, and
    # test on DDI.
    K = FLAGS.K

    train_x_t, train_y_t, valid_x_t, valid_y_t, test_x_t, test_y_t = dataset_tuple

    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():
        # Generate placeholders for the images and labels.
        features_placeholder_t, labels_placeholder_t, dropout_placeholder_t = placeholder_inputs(
            FLAGS.batch_size_t)

        # Build a Graph that computes predictions from the inference model.
        placeholder_tuple = tuple([features_placeholder_t, labels_placeholder_t, dropout_placeholder_t])

        logits, regularizers_t, linear_pred_t, L_hidden_vars, task_hidden_vars, regularizers_l1 \
                                                = models.inference(placeholder_tuple,
                                                                   FLAGS.hidden_units,
                                                                   FLAGS.hidden_units_task,
                                                                   FLAGS.FEATURE_SIZE)

        # Add to the Graph the Ops for loss calculation.
        loss, regularization, obj_value, obj_value_t, reg_l1 = models.loss(logits, regularizers_t,
                                                                   regularizers_l1, FLAGS.l2_reg_para_t, FLAGS.l1_reg_para)

        # Add to the Graph the Ops that calculate and apply gradients.
        train_op , optimizer_task = models.training(obj_value, FLAGS.learning_rate1, FLAGS.learning_rate_t,
                                                    L_hidden_vars, task_hidden_vars)

        # Add the Op to compare the logits to the labels during evaluation.
        eval_correct_t, lable_pred_t, label_true_t, f1_tuple = models.evaluation(linear_pred_t, labels_placeholder_t)

        # TODO debug

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.merge_all_summaries()

        # Add the variable initializer Op.
        init = tf.initialize_all_variables()

        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver()
        # Create a saver for first hidden layer variables.
        # weights1, biases1 = var_list1
        # saver_l1 = tf.train.Saver({"weights1": weights1, "biases1": biases1})
        save_dict = {}
        for ll in range(FLAGS.L):
            save_dict["weights" + str(ll + 1)] = L_hidden_vars[2*ll]
            save_dict["biases" + str(ll + 1)] = L_hidden_vars[2*ll+1]
        saver_l_half = tf.train.Saver(save_dict)

        # Create a session for running Ops on the Graph.
        sess = tf.Session()

        # Instantiate a SummaryWriter to output summaries and the Graph.
        summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)

        # And then after everything is built:

        # Run the Op to initialize the variables.
        sess.run(init)

        test_f1_t = [[] for ii in range(K)]
        valid_f1_t = [[] for ii in range(K)]
        train_f1_t = [[] for ii in range(K)]

        test_aucs_t = [[] for ii in range(K)]
        valid_aucs_t = [[] for ii in range(K)]
        train_aucs_t = [[] for ii in range(K)]

        obj_value_values = []
        obj_value_values_t = [[] for ii in range(K)]
        lable_pred_t_val = 0 # TODO debug
        lable_pred_t_val = 0
        if FLAGS.restore_flag == 0:
            # restore all variables from previous training
            ckpt_file = FLAGS.load_dir + FLAGS.loadflag + '/' + FLAGS.checkpointfilename
            if os.path.exists(ckpt_file):
                saver.restore(sess, ckpt_file)
                for kk in range(K):
                    results_saved = np.load(FLAGS.load_dir + FLAGS.loadflag + '/' + FLAGS.loadflag + '_tasktask' + str(FLAGS.tasks_ids[kk]) + '_AUC_results.npy')
                    train_aucs_t[kk], valid_aucs_t[kk], test_aucs_t[kk], obj_value_values_t[kk], valid_accs_smooth = results_saved
                    results_saved = np.load(FLAGS.load_dir + FLAGS.loadflag + '/' + FLAGS.loadflag + '_tasktask' + str(
                        FLAGS.tasks_ids[kk]) + '_F1_results.npy')
                    train_f1_t[kk], valid_f1_t[kk], test_f1_t[kk], obj_value_values_t[kk], valid_accs_smooth = results_saved
                print("Model loaded")
            else:
                print('No checkpoint file found')
        else:
            # restore first hidden layer variables from autoencoder.
            ckpt_file = FLAGS.load_dir + FLAGS.loadflag + '/' + FLAGS.checkpointfilename_l_half
            if os.path.exists(ckpt_file):
                saver_l_half.restore(sess, ckpt_file)
                print("Auto encoder Model loaded " + ckpt_file)
            else:
                print('Didn t load auto encoder checkpoint found')

        # Start the training loop.
        for step in xrange(FLAGS.max_steps):
            start_time = time.time()

            feed_dict = fill_feed_dict_run(train_x_t, train_y_t, FLAGS.dropout_rate_t,
                                           features_placeholder_t, labels_placeholder_t, dropout_placeholder_t)
            # '''TODO debug'''
            # feed_dict[learning_rate1] = FLAGS.learning_rate1



            _, \
            loss_value, \
            regularization_value, \
            reg_l1_val,\
            obj_value_val, \
            obj_value_t_val, \
            labels_placeholder_t_val, \
            lable_pred_t_val \
                = sess.run([train_op,
                            loss,
                            regularization,
                            reg_l1,
                            obj_value,
                            obj_value_t,
                            labels_placeholder_t,
                            lable_pred_t],
                            feed_dict=feed_dict)




            obj_value_values.append(obj_value_val)
            for kk in range(K):
                obj_value_values_t[kk].append(obj_value_t_val[kk])

            duration = time.time() - start_time

            # Write the summaries and print an overview fairly often.
            if step % 1000 == 0 or step == 1:
                # Print status to stdout.
                print('Step %d: loss = %.2f reg l2 = %.2f reg_l1 = %.3f obj = %.2f (%.3f sec)' % (step,
                                                                                 loss_value,
                                                                                 regularization_value,
                                                                                 reg_l1_val,
                                                                                 obj_value_val,
                                                                                 duration))
                # print(logits_value)
                # Update the events file.
            # if step % 2000 == 0:
            #     summary_str = sess.run(summary_op, feed_dict=feed_dict)
            #     summary_writer.add_summary(summary_str, step)
            #     summary_writer.flush()

            # Save a checkpoint and evaluate the model periodically.
            if (step + 1) == FLAGS.max_steps: # (step + 1) % 4000 == 0 or
                checkpoint_file = os.path.join(FLAGS.train_dir, 'checkpoint'+ str(current_taskid))
                saver.save(sess, checkpoint_file, global_step=step)
                # checkpoint_file_l1 = os.path.join(FLAGS.train_dir, FLAGS.checkpointfilename_l_half)
                # saver_l_half.save(sess, checkpoint_file_l1)

            if (step + 1) % 4000 == 0 or (step + 1) == FLAGS.max_steps:
                # Evaluate against the training set.
                print('Training Data Eval:')
                precision_train, recall_train, f1_score_train, aucs_train = do_eval(sess,
                                                                      f1_tuple,
                                                                      lable_pred_t,
                                                                      label_true_t,
                                                                      features_placeholder_t,
                                                                      labels_placeholder_t,
                                                                      dropout_placeholder_t,
                                                                      train_x_t, train_y_t, FLAGS.dropout_rate_t_evaluation)


                for kk in range(K):
                    print('task %d:  Precision: %0.04f  Recall: %0.04f  F1 score: %0.04f, AUC: %0.04f' %
                         (FLAGS.tasks_ids[kk], precision_train[kk], recall_train[kk], f1_score_train[kk], aucs_train[kk]))

                for kk in range(FLAGS.K):
                    train_f1_t[kk].append(f1_score_train[kk])
                    train_aucs_t[kk].append(aucs_train[kk])

                # print train_acc, true_count, data_set_size, num_examples

                # Evaluate against the validation set.
                print('Validation Data Eval:')
                precision_valid, recall_valid, f1_score_valid, aucs_valid = do_eval(sess,
                                                                        f1_tuple,
                                                                        lable_pred_t,
                                                                        label_true_t,
                                                                         features_placeholder_t,
                                                                         labels_placeholder_t,
                                                                         dropout_placeholder_t,
                                                                         valid_x_t, valid_y_t, FLAGS.dropout_rate_t_evaluation)
                for kk in range(K):
                    print('task %d:  Precision: %0.04f  Recall: %0.04f  F1 score: %0.04f, AUC: %0.04f' %
                          (FLAGS.tasks_ids[kk], precision_valid[kk], recall_valid[kk], f1_score_valid[kk], aucs_valid[kk]))


                for kk in range(FLAGS.K):
                    valid_f1_t[kk].append(f1_score_valid[kk])
                    valid_aucs_t[kk].append(aucs_valid[kk])
                #
                #     if len(valid_aucs_t[kk]) >= FLAGS.stop_num:    # only consider early stop condition when there step more than threshold
                #         stop_flag = early_stop_by_valid(valid_aucs_t[kk], FLAGS.stop_num)
                #         if stop_flag == True:
                #             FLAGS.learning_rate_t[kk] = 0
                #             # print FLAGS.learning_rate_t[kk]


                # Evaluate against the test set.
                print('Test Data Eval:')
                precision_test, recall_test, f1_score_test, aucs_test = do_eval(sess,
                                                                     f1_tuple,
                                                                     lable_pred_t,
                                                                     label_true_t,
                                                                     features_placeholder_t,
                                                                     labels_placeholder_t,
                                                                     dropout_placeholder_t,
                                                                     test_x_t, test_y_t, FLAGS.dropout_rate_t_evaluation)
                for kk in range(K):
                    print('task %d:  Precision: %0.04f  Recall: %0.04f  F1 score: %0.04f, AUC: %0.04f' %
                          (FLAGS.tasks_ids[kk], precision_test[kk], recall_test[kk], f1_score_test[kk], aucs_test[kk]))

                for kk in range(FLAGS.K):
                    test_f1_t[kk].append(f1_score_test[kk])
                    test_aucs_t[kk].append(aucs_test[kk])

                #
                # for kk in range(FLAGS.K):
                #     print "learning rate task "+str(kk), sess.run(optimizer_task[kk]._lr_t, feed_dict=feed_dict)

            # count_tasks = 0  # count how many tasks learning rate set to zero
            # for kk in range(FLAGS.K):
            #     if FLAGS.learning_rate_t[kk] == 0:
            #         count_tasks += 1
            # if count_tasks == FLAGS.K:
            #     print "early stop by slop"
            #     break

        max_test_performance = [0]*K
        max_test_steps = [0] * K
        # plot figures
        for kk in range(K):
            data_tuple = tuple([train_f1_t[kk], test_f1_t[kk], valid_f1_t[kk],
                                obj_value_values_t[kk]])
            plot_figures(data_tuple, 'task' + str(FLAGS.tasks_ids[kk]), K, 'F1')
            data_tuple = tuple([train_aucs_t[kk], test_aucs_t[kk], valid_aucs_t[kk],
                                obj_value_values_t[kk]])
            max_test_performance[kk], max_test_steps[kk] = plot_figures(data_tuple, 'task' + str(FLAGS.tasks_ids[kk]), K, 'AUC')


        plt.figure()
        plt.plot(obj_value_values)
        plt.savefig(FLAGS.train_dir + current_time + '_alltasks_obj_values.png')
        plt.close("all")
        for kk in range(K):
            print("Testing AUC of task %4d  at step %d: %f ") %(FLAGS.tasks_ids[kk], max_test_steps[kk], max_test_performance[kk])

    return max_test_performance # return the main task's performance



def main(argv):
    ''' multi task dnn for arbitrary number of tasks
    '''
    # record print out info
    print('\n' + FLAGS.train_dir + '\n')


    orig_stdout = sys.stdout
    if not os.path.exists(FLAGS.train_dir):
        os.makedirs(FLAGS.train_dir)
    f = file(FLAGS.train_dir + 'out.txt', 'w', 0)
    sys.stdout = f

    '''record configure files'''
    record_configure_files(argv)

    '''Load data'''
    fname = FLAGS.input_dir + FLAGS.input_filename_all  # with normalization
    dataset_tuple = np.load(fname)
    features, labels, column_map = dataset_tuple

    '''Normalize and split data'''
    features_normalized = stats.zscore(features, axis=0, ddof=1) # specified axis, using n-1 degrees of freedom (ddof=1) to calculate the standard deviation:

    dataset_tuple = select_tasks_load_inds(features_normalized, labels)

    print('\n The current time is : %s \n' % current_time)

    task_ids_store = [] # store nonlinear tasks
    tasks_run = []
    tasks_skiped = []
    task_auc_random = []
    task_auc_selected = []
    f1 = file(FLAGS.train_dir + 'results.txt', 'w', 0)
    num_auxiary_tasks = 4
    auxiary_strategy = 1   # 1: cosine model similarity 0: random

    # compute the model similarity from lr model vectors.
    modelsimi = np.load(FLAGS.input_dir + FLAGS.timeflag_matlab + "tasks_lrmodel_similarity.npy")
    sorted_index_all, sorted_simis_all = modelsimi

    for ii in list(range(1072, 1318)):
        # random sample tasks
        if auxiary_strategy == 0:
            taskids = list(range(1318))
            del taskids[ii]
            list_of_random_tasks = random.sample(taskids, num_auxiary_tasks)
            task_ids = [ii] + list_of_random_tasks
        else:
            # similar task strategy
            temp = sorted_index_all[ii][:num_auxiary_tasks+1]
            task_ids = [int(temp[i]) for i in range(num_auxiary_tasks+1)]

        FLAGS.tasks_ids = task_ids
        FLAGS.tasks_ids_index = task_ids

        task_ids_store.append(task_ids)  # multi tasks run together

        current_label = labels[:, ii]
        all_pos_num = len(current_label[current_label==1])
        if all_pos_num <= 100:
            tasks_skiped.append(ii)
            print "skip task ", ii
            continue
        tasks_run.append(ii)

        nn_test_auc = run_training(dataset_tuple, ii)
        for kk in range(num_auxiary_tasks + 1):
            f1.write("taskid: " + str(task_ids[kk]))
            f1.write(" AUC: " + str(nn_test_auc[kk]))
        f1.write("\n")
        task_auc_random.append(nn_test_auc)

    results_final_saved = tuple([task_auc_random, tasks_run, task_ids_store])
    np.save(FLAGS.train_dir + current_time + 'final_results.npy', results_final_saved)
    sys.stdout = orig_stdout
    f1.close()
    f.close()
    print('\n' + FLAGS.train_dir + '\n')



if __name__ == "__main__":
    '''multi task - multi layer fully connected neural network
       for arbitrary number of tasks, arbitrary number of layers
       Use F1 and AUC as measuremets, restore the splitting indices from mat file
       add l1 norm to the network
       early stop of network
       add more tasks specific layers
       CUDA_VISIBLE_DEVICES= nohup python -u mtl_dnn_sym_earlystop_runall.py > nohup.simiall_fold2 2>&1 &
       CUDA_VISIBLE_DEVICES= nohup python -u mtl_dnn_sym_earlystop_runall.py > nohup.simiall_fold2_2 2>&1 &
       '''
    tf.app.run()





