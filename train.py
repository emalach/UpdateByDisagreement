from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from data import distorted_inputs
from model import select_model
import json
import re


LAMBDA = 0.01
MOM = 0.9
tf.app.flags.DEFINE_string('pre_checkpoint_path', '',
                           """If specified, restore this pretrained model """
                           """before beginning any training.""")

tf.app.flags.DEFINE_string('train_dir', '/home/dpressel/dev/work/AgeGenderDeepLearning/Folds/tf/test_fold_is_0',
                           'Training directory')

tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

tf.app.flags.DEFINE_integer('num_preprocess_threads', 4,
                            'Number of preprocessing threads')

tf.app.flags.DEFINE_string('optim', 'Momentum',
                           'Optimizer')

tf.app.flags.DEFINE_integer('image_size', 227,
                            'Image size')

tf.app.flags.DEFINE_float('eta', 0.002,
                          'Learning rate')

tf.app.flags.DEFINE_float('pdrop', 0.,
                          'Dropout probability')

tf.app.flags.DEFINE_integer('max_steps', 15000,
                          'Number of iterations')

tf.app.flags.DEFINE_integer('epochs', -1,
                            'Number of epochs')

tf.app.flags.DEFINE_integer('batch_size', 128,
                            'Batch size')

tf.app.flags.DEFINE_string('checkpoint', 'checkpoint',
                          'Checkpoint name')

tf.app.flags.DEFINE_string('model_type', 'default',
                           'Type of convnet')

tf.app.flags.DEFINE_string('pre_model',
                            '',#'./inception_v3.ckpt',
                           'checkpoint file')

tf.app.flags.DEFINE_boolean('dual', False,
                            """Use dual training.""")

tf.app.flags.DEFINE_boolean('Qloss', False,
                            """Use Qloss [arXiv:1406.2080].""")

tf.app.flags.DEFINE_string('reed', 'none',
                            """Use Reed's mehtod (soft/hard/None).""")

tf.app.flags.DEFINE_float('Qloss_weight', 0.01,
                            """Weight decay for Qloss [arXiv:1406.2080].""")

tf.app.flags.DEFINE_boolean('s_model', False,
                            """Implement Smodel loss suggested by Goldberg 2016.""")

tf.app.flags.DEFINE_integer('init_iter', 0,
                            """Init dual training after iteration k (Default 0).""")

tf.app.flags.DEFINE_integer('min_batch_size', 0.1,
                            """Ratio of the original batch size that is still trainable. If -1, use full batches with buffering (Default 0.1)""")

FLAGS = tf.app.flags.FLAGS

print("reed = %s" % FLAGS.reed)
if FLAGS.Qloss:
    print("Qloss_weight = %f" % FLAGS.Qloss_weight)
if FLAGS.s_model:
    print("s_model")
print("init_iter = %d" % FLAGS.init_iter)
if FLAGS.dual:
    print("dual")
Q_GLOBAL = None

# Every 5k steps cut learning rate in half
def exponential_staircase_decay(at_step=5000, decay_rate=0.5):

    def _decay(lr, global_step):
        return tf.train.exponential_decay(lr, global_step,
                                          at_step, decay_rate, staircase=True)
    return _decay

def optimizer(optim, eta, loss_fn, variables=None, name="optimizer"):
    global_step = tf.Variable(0, trainable=False)
    optz = optim
    if optim == 'Adadelta':
        optz = lambda lr: tf.train.AdadeltaOptimizer(lr, 0.95, 1e-6)
    elif optim == 'Adam':
        optz = 'Adam'
        lr_decay_fn = None
        lr = eta
    elif optim == 'Momentum':
        lr_decay_fn = exponential_staircase_decay()
        optz = lambda lr: tf.train.MomentumOptimizer(lr, MOM)
        lr = tf.train.exponential_decay(eta, global_step, 5000, 0.5, staircase=True)
        return tf.train.MomentumOptimizer(lr, MOM).minimize(loss_fn, global_step=global_step)

    return tf.contrib.layers.optimize_loss(loss_fn, global_step, lr, optz, clip_gradients=4., variables=variables, name=name)

def loss(logits, labels, global_step, weights=None, scope=None):
    Qloss = FLAGS.Qloss
    labels = tf.cast(labels, tf.int32)
    NUM_OF_CLASSES = 2
    Q_weight = 0
    if FLAGS.reed == 'soft':
        BETA = 0.95
        log_softmax = tf.nn.log_softmax(logits)
        softmax = tf.nn.softmax(logits)
        cross_entropy = -tf.reduce_sum(BETA*tf.one_hot(labels, NUM_OF_CLASSES)*log_softmax + (1-BETA)*softmax*log_softmax, reduction_indices=[1])
        print(cross_entropy)
    elif FLAGS.reed == 'hard':
        BETA = 0.8
        log_softmax = tf.nn.log_softmax(logits)
        z = tf.one_hot(tf.argmax(logits, 1), NUM_OF_CLASSES)
        cross_entropy = -tf.reduce_sum(BETA*tf.one_hot(labels, NUM_OF_CLASSES)*log_softmax + (1-BETA)*z*log_softmax, reduction_indices=[1])
    elif FLAGS.s_model:
        APRIOR_NOISE=0.46
        bias_weights = (np.array([np.array([np.log(1. - APRIOR_NOISE)
                            if i == j else
                            np.log(APRIOR_NOISE / (NUM_OF_CLASSES - 1.))
                            for j in range(NUM_OF_CLASSES)]) for i in
                            range(NUM_OF_CLASSES)])
                            + 0.01 * np.random.random((NUM_OF_CLASSES, NUM_OF_CLASSES)))

        Q_mat = tf.Variable(initial_value=bias_weights, dtype=tf.float32, name='Q_mat')
        softmax1 = tf.nn.softmax(logits)
        log_softmax2 = tf.nn.log_softmax(tf.matmul(softmax1, Q_mat))
        cross_entropy = -tf.reduce_sum(tf.one_hot(labels, NUM_OF_CLASSES) * log_softmax2, reduction_indices=[1])
    elif not Qloss:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits, labels, name='cross_entropy_per_example')
    else:
        Q_mat = tf.Variable(initial_value=np.eye(NUM_OF_CLASSES)*5, dtype=tf.float32, name='Q_mat')
        Q = tf.transpose(tf.nn.softmax(Q_mat))
        softmax = tf.cond(global_step > FLAGS.init_iter, lambda: tf.matmul(tf.nn.softmax(logits, name=None), Q), 
                                                 lambda: (tf.nn.softmax(logits, name=None)))
        global Q_GLOBAL
        Q_GLOBAL = Q
        cross_entropy = -tf.reduce_sum(tf.one_hot(labels, NUM_OF_CLASSES) * tf.log(softmax), reduction_indices=[1])
        Q_weight = tf.cond(global_step > FLAGS.init_iter, lambda: tf.nn.l2_loss(Q), lambda: tf.constant(0, dtype=tf.float32))

    # weight by classes
    neg_weight = FLAGS.batch_size/(2*tf.maximum(1.0, tf.reduce_sum(tf.to_float(tf.equal(labels, 0)))))
    pos_weight = FLAGS.batch_size/(2*tf.maximum(1.0, tf.reduce_sum(tf.to_float(tf.equal(labels, 1)))))
    pn_weights = tf.to_float(labels)*pos_weight + tf.to_float(1-labels)*neg_weight
    print(pn_weights)

    if weights is None:
      cross_entropy_mean = tf.reduce_mean(pn_weights*cross_entropy, name='cross_entropy')
    else:
      # avoid division by zero
      cross_entropy_mean = tf.reduce_sum(pn_weights*weights*cross_entropy, name='cross_entropy')/ \
                                tf.maximum(FLAGS.batch_size*FLAGS.min_batch_size,tf.reduce_sum(weights))

    tf.add_to_collection('losses', cross_entropy_mean)
    losses = tf.get_collection('losses', scope=scope)
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=scope)
    print(regularization_losses)
    total_loss = cross_entropy_mean + LAMBDA * sum(regularization_losses) + FLAGS.Qloss_weight*Q_weight
    if scope is not None:
        tf.scalar_summary(scope + '/tl (raw)', total_loss)
    else:
        tf.scalar_summary('/tl (raw)', total_loss)
    #total_loss = tf.add_n(losses + regularization_losses, name='total_loss')
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    loss_averages_op = loss_averages.apply(losses + [total_loss])
    for l in losses + [total_loss]:
        if scope is not None:
            tf.scalar_summary(scope + '/' + l.op.name + ' (raw)', l)
            tf.scalar_summary(scope + '/' + l.op.name, loss_averages.average(l))
        else:
            tf.scalar_summary('/' + l.op.name + ' (raw)', l)
            tf.scalar_summary('/' + l.op.name, loss_averages.average(l))
    with tf.control_dependencies([loss_averages_op]):
        total_loss = tf.identity(total_loss)
    
    pred_true = tf.to_float(tf.equal(tf.argmax(logits, 1), tf.to_int64(labels)))
    acc = tf.reduce_mean(pred_true)
    acc_pos = tf.reduce_sum(tf.to_float(pred_true*tf.to_float(labels)))/tf.to_float(tf.maximum(1,tf.reduce_sum(labels)))
    acc_neg = tf.reduce_sum(tf.to_float(pred_true*tf.to_float(1-labels)))/tf.to_float(tf.maximum(1,tf.reduce_sum(1-labels)))
    if scope is not None:
        tf.scalar_summary(scope + '/acc (raw)', acc)
        tf.scalar_summary(scope + '/acc pos (raw)', acc_pos)
        tf.scalar_summary(scope + '/acc neg (raw)', acc_neg)
    else:
        tf.scalar_summary('/acc (raw)', acc)
        tf.scalar_summary('/acc pos (raw)', acc_pos)
        tf.scalar_summary('/acc neg (raw)', acc_neg)
    
    if Qloss:
        return total_loss, Q[0,0]
    return total_loss, acc

def main(argv=None):
    with tf.Graph().as_default():

        global_step = tf.Variable(0, trainable=False)

        model_fn = select_model(FLAGS.model_type)
        # Open the metadata file and figure out nlabels, and size of epoch
        input_file = os.path.join(FLAGS.train_dir, 'md.json')
        print(input_file)
        with open(input_file, 'r') as f:
            md = json.load(f)

        images, labels, _ = distorted_inputs(FLAGS.train_dir, FLAGS.batch_size, FLAGS.image_size, FLAGS.num_preprocess_threads)
        if not FLAGS.dual:
            logits = model_fn(md['nlabels'], images, 1-FLAGS.pdrop, True)
            total_loss, accuracy = loss(logits, labels, global_step)
        else:
            with tf.variable_scope("net1") as scope:
                logits1 = model_fn(md['nlabels'], images, 1-FLAGS.pdrop, True)
            with tf.variable_scope("net2") as scope:
                logits2 = model_fn(md['nlabels'], images, 1-FLAGS.pdrop, True)

            pred1 = tf.argmax(logits1, 1)
            pred2 = tf.argmax(logits2, 1)

            update_step = tf.stop_gradient(tf.to_float(tf.logical_or(tf.not_equal(pred1, pred2), global_step < FLAGS.init_iter)))

            with tf.variable_scope("net1") as scope:
                if FLAGS.min_batch_size == -1:
                    total_loss1, accuracy1 = loss(logits1, labels, global_step, None, scope.name)
                else:
                    total_loss1, accuracy1 = loss(logits1, labels, global_step, update_step, scope.name)
            with tf.variable_scope("net2") as scope:
                if FLAGS.min_batch_size == -1:
                    total_loss2, accuracy2 = loss(logits2, labels, global_step, None, scope.name)
                else:
                    total_loss2, accuracy2 = loss(logits2, labels, global_step, update_step, scope.name)
            
            disagree_rate = tf.reduce_mean(tf.to_float(tf.not_equal(pred1, pred2)))

        if not FLAGS.dual:
            train_op = optimizer(FLAGS.optim, FLAGS.eta, total_loss)
        else:
            with tf.variable_scope("net1") as scope:
                var_net1 = [var for var in tf.all_variables() if var.name.startswith("net1")]
                train_op1 = optimizer(FLAGS.optim, FLAGS.eta, total_loss1, variables=var_net1, name=scope.name)
            with tf.variable_scope("net2") as scope:
                var_net2 = [var for var in tf.all_variables() if var.name.startswith("net2")]
                train_op2 = optimizer(FLAGS.optim, FLAGS.eta, total_loss2, variables=var_net2, name=scope.name)
        saver = tf.train.Saver(tf.all_variables(), max_to_keep=151)
        summary_op = tf.merge_all_summaries()
        init = tf.initialize_all_variables()
        sess = tf.Session(config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement))

        sess.run(init)

        # This is total hackland, it only works to fine-tune iv3
        if FLAGS.pre_model:
            inception_variables = tf.get_collection(
                tf.GraphKeys.VARIABLES, scope="InceptionV3")
            restorer = tf.train.Saver(inception_variables)
            restorer.restore(sess, FLAGS.pre_model)

        if FLAGS.pre_checkpoint_path:
            if tf.gfile.Exists(FLAGS.pre_checkpoint_path) is True:
                print('Trying to restore checkpoint from %s' % FLAGS.pre_checkpoint_point)
                restorer = tf.train.Saver()
                tf.train.latest_checkpoint(FLAGS.pre_checkpoint_path)
                print('%s: Pre-trained model restored from %s' %
                      (datetime.now(), FLAGS.pre_checkpoint_path))


        run_dir = '%s/run-%d' % (FLAGS.train_dir, os.getpid())

        checkpoint_path = '%s/%s' % (run_dir, FLAGS.checkpoint)
        if tf.gfile.Exists(run_dir) is False:
            print('Creating %s' % run_dir)
            tf.gfile.MakeDirs(run_dir)

        tf.train.write_graph(sess.graph_def, run_dir, 'model.pb', as_text=True)

        tf.train.start_queue_runners(sess=sess)


        summary_writer = tf.train.SummaryWriter(run_dir, sess.graph)
        steps_per_train_epoch = int(md['train_counts'] / FLAGS.batch_size)
        num_steps = FLAGS.max_steps if FLAGS.epochs < 1 else FLAGS.epochs * steps_per_train_epoch
        print('Requested number of steps [%d]' % num_steps)

        trainable_buffer_img = None
        trainable_buffer_lbl = None
        for step in range(num_steps):
            start_time = time.time()
            if FLAGS.Qloss:
                _, loss_value, acc_value, q_val = sess.run([train_op, total_loss, accuracy, Q_GLOBAL], feed_dict={global_step: step})
                print(q_val)
            elif not FLAGS.dual:
                _, loss_value, acc_value = sess.run([train_op, total_loss, accuracy], feed_dict={global_step: step})
            elif FLAGS.dual and (step < FLAGS.init_iter or FLAGS.min_batch_size != -1):
                _, _, loss_value, acc_value1, acc_value2, drate = sess.run([train_op1, train_op2, total_loss1, accuracy1, accuracy2, disagree_rate], feed_dict={global_step: step})
            else:
                #loss_value, acc_value1, acc_value2, drate = (0,0,0,0)
                img, lbl, us, loss_value, acc_value1, acc_value2, drate = sess.run([images, labels, update_step, total_loss1, accuracy1, accuracy2, disagree_rate], feed_dict={global_step: step})
                rel_img = img[us == 1]
                rel_lbl = lbl[us == 1]
                if trainable_buffer_img is None:
                    trainable_buffer_img = rel_img
                    trainable_buffer_lbl = rel_lbl
                else:
                    print(np.shape(trainable_buffer_lbl), np.shape(rel_lbl))
                    trainable_buffer_img = np.vstack((trainable_buffer_img, rel_img))
                    trainable_buffer_lbl = np.hstack((trainable_buffer_lbl, rel_lbl))

                if trainable_buffer_img.shape[0] >= FLAGS.batch_size:
                    batch_img = trainable_buffer_img[:FLAGS.batch_size]
                    batch_lbl = trainable_buffer_lbl[:FLAGS.batch_size]
                    _, _, loss_value, acc_value1, acc_value2, drate = sess.run([train_op1, train_op2, total_loss1, accuracy1, accuracy2, disagree_rate], feed_dict={global_step: step, images: batch_img, labels: batch_lbl})
                    trainable_buffer_img = trainable_buffer_img[FLAGS.batch_size:]
                    trainable_buffer_lbl = trainable_buffer_lbl[FLAGS.batch_size:]	
                #_, loss_value, acc_value2, drate = sess.run([train_op2, total_loss2, accuracy2, disagree_rate], feed_dict={global_step: step})
            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 1 == 0:
                num_examples_per_step = FLAGS.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)
                
                if not FLAGS.dual:
                    format_str = ('%s: step %d, loss = %.3f, acc = %.3f (%.1f examples/sec; %.3f ' 'sec/batch)')
                    print(format_str % (datetime.now(), step, loss_value, acc_value,
                                    examples_per_sec, sec_per_batch))
                else:
                    format_str = ('%s: step %d, loss = %.3f, acc1 = %.3f, acc2 = %.3f, disagree_rate = %.3f (%.1f examples/sec; %.3f ' 'sec/batch)')
                    print(format_str % (datetime.now(), step, loss_value, acc_value1, acc_value2, drate,
                                    examples_per_sec, sec_per_batch))

            # Loss only actually evaluated every 100 steps?
            if step % 100 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)
                
            if step % 200 == 0 or (step + 1) == num_steps:
                saver.save(sess, checkpoint_path, global_step=step)

if __name__ == '__main__':
    tf.app.run()
