import time
import tensorflow as tf


current_time = time.strftime("%Y%m%d_%H-%M")
tf.app.flags.DEFINE_string('current_time', current_time, '')
timeflag_matlab = '05-Mar-201719-58/'
tf.app.flags.DEFINE_string('timeflag_matlab', timeflag_matlab, 'The timeflag for matlab resutls for this fold')
tf.app.flags.DEFINE_string('train_dir', '../../../../AllData/Fei-DrugData/ddi_prep/tensorboard_tmp/' + current_time + '/', 'Directory to put the check points data')
tf.app.flags.DEFINE_string('load_dir', '../../../../AllData/Fei-DrugData/ddi_prep/tensorboard_tmp/', 'Directory to put the check points data')
tf.app.flags.DEFINE_string('input_dir', '../../../../AllData/Fei-DrugData/ddi_prep/', 'Directory to download data files')
tf.app.flags.DEFINE_string('input_filename_matfile', 'eventall_featlabel_clean_sym05-Mar-201719-58.mat', 'store spliting indices in matlab')
tf.app.flags.DEFINE_string('input_filename_all', 'eventall_featlabel_clean_sym.npy', 'input file for all tasks, features and labels')


tf.app.flags.DEFINE_float('filewrite', 0, 'Do we write results into file 1 yes, 0 no')
tf.app.flags.DEFINE_string('loadflag', 'NA', '20161012_13-02  Do we load results from previous round, load: a time flag')
tf.app.flags.DEFINE_integer('restore_flag', 0, '0: restore variable from the checkpoint file;'
                                               '1: restore from auto encoder')
tf.app.flags.DEFINE_string('checkpointfilename', 'checkpoint-99999', 'Directory to put the check points data')
tf.app.flags.DEFINE_string('checkpointfilename_l_half', 'checkpointfilename_l_half', 'Directory to put the check points data')

tf.app.flags.DEFINE_string('FEATURE_SIZE', 1232, 'num of features')
tf.app.flags.DEFINE_integer('max_steps', int(2e+4), 'Number of steps to run trainer.')
tf.app.flags.DEFINE_float('test_rate', 0.1, 'rate for testing dataset')
tf.app.flags.DEFINE_float('valid_rate', 0.1, 'rate for training dataset')
tf.app.flags.DEFINE_integer('activation_func', 0,  '0: tanh function'
                                                   '1: sigmoid function'
                                                   '2: relu function')
# tf.app.flags.DEFINE_integer('Layer_last_store', 1, 'store form 0 to which layer')
# tf.app.flags.DEFINE_integer('Layer_last_restore', [0, 1], 'the layers you want to restore, 0: w0 the input layer')

FLAGS = tf.app.flags.FLAGS


'''Tune those parameters if you want to change the num of layers/tasks'''

task_ids = [400, 769, 625, 614, 479] # similar [400, 1137, 567, 49, 1023]# dissimilar [400, 769, 625, 614, 479]  [760, 306,400, 558, 306, 786, 1250, 1225]  # 771, 367, 1225, 908, 173
tf.app.flags.DEFINE_integer('tasks_ids', task_ids, 'MUST IN this order [445, 353, 804, 1246], so to correspond to indicies'
                                                                'tasks ids selected to run together [1246, 804, 809] ')
tf.app.flags.DEFINE_integer('tasks_ids_index', task_ids, ' [445, 353, 804, 1246] corresponding index in mat file:,'
                                                                 ' task 1246 correspoinds to 1246')
tf.app.flags.DEFINE_integer('K', len(task_ids), 'Number of tasks')
tf.app.flags.DEFINE_integer('stop_num', 10, 'Number of points used for early stop rule')
tf.app.flags.DEFINE_integer('slope', 0.001, 'Number of points used for early stop rule')


hidden_layers = [128]
hidden_units_task = [[ii] * FLAGS.K for ii in [32]]
tf.app.flags.DEFINE_integer("hidden_units_task", hidden_units_task, "task specific hidden layers")
tf.app.flags.DEFINE_integer('L', len(hidden_layers), 'Number of layers, shared by all tasks')
tf.app.flags.DEFINE_integer('hidden_units', hidden_layers, '[600, 50, 50, 50] Number of units in hidden layer 1.')
tf.app.flags.DEFINE_float('learning_rate1', 1e-4, '1e-20learning rate for first hidden layers')
'''Task specific parameters'''
tf.app.flags.DEFINE_float('learning_rate_t', [1e-4] * FLAGS.K, 'learning rate task for all tasks')
tf.app.flags.DEFINE_float('dropout_rate_t', [1] * FLAGS.K, 'For training: Initial the probability of keep the value')
tf.app.flags.DEFINE_float('dropout_rate_L1', 0.5, 'first layer dropout rate')
tf.app.flags.DEFINE_float('dropout_rate_t_evaluation', [1] * FLAGS.K, 'For evaluation: Initial the probability of keep the value')
tf.app.flags.DEFINE_float('l2_reg_para_t', [0] * FLAGS.K, 'regularization parameter ')
tf.app.flags.DEFINE_float('l1_reg_para', 0, 'regularization parameter ')
tf.app.flags.DEFINE_integer('batch_size_t', [100] * FLAGS.K, 'Batch size.Must divide evenly into the dataset sizes.')

