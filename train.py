# epoch 단위로 안하기?
# final 그림 그리기

from inputs import inputs
from model import Model
import tensorflow as tf
import os
import argparse


class OverfitError(Exception):
    pass

parser = argparse.ArgumentParser()
parser.add_argument('-trn_site', type=str)
parser.add_argument('-test_site', type=str)
parser.add_argument('-rnn_cell', type=str)
parser.add_argument('-rnn_layer', type=int, nargs='+')
parser.add_argument('-output_layer', type=int, nargs='+')
parser.add_argument('-keep_prob', type=float, default=1.0)
parser.add_argument('-share_encoder', action='store_true')
parser.add_argument('-no_mode', action='store_true')
parser.add_argument('-int_mode', action='store_true')

args = parser.parse_args()

trn_site = args.trn_site
test_site = args.test_site
rnn_cell = args.rnn_cell
rnn_layer = args.rnn_layer
output_layer = args.output_layer
keep_prob = args.keep_prob
share_encoder = args.share_encoder
no_mode = args.no_mode
int_mode = args.int_mode

batch_size = 32

with tf.variable_scope('Model'):
    with tf.name_scope('Train'):
        trn_inputs, trn_init_op = inputs(trn_site, int_mode, batch_size, 5)
        trn_model = Model(trn_inputs, rnn_cell, rnn_layer, output_layer, keep_prob,
                          share_encoder, no_mode)

with tf.variable_scope('Model', reuse=True):
    with tf.name_scope('Test'):
        test_inputs, test_init_op = inputs(test_site, int_mode, batch_size, 5)
        test_model = Model(test_inputs, rnn_cell, rnn_layer, output_layer, keep_prob,
                           share_encoder, no_mode)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

saver = tf.train.Saver()

if not os.path.exists('model'):
    os.mkdir('model')

model_config = {'trn_site': trn_site,
                'test_site': test_site,
                'rnn_cell': rnn_cell,
                'rnn_layer': '_'.join([str(x) for x in rnn_layer]),
                'output_layer': '_'.join([str(x) for x in output_layer]),
                'keep_prob': keep_prob,
                'share_encoder': str(share_encoder),
                'no_mode' : str(no_mode),
                'int_mode': str(int_mode)
                }

model_dir = ['{}-{}'.format(key, model_config[key]) for key in sorted(model_config.keys())]
model_dir = '_'.join(model_dir)

with tf.Session(config=config) as sess:
    summary_writer = tf.summary.FileWriter('summary/{}'.format(model_dir), sess.graph,
                                           flush_secs=120)

    count = 0
    epoch = 0
    best_test_loss = 1e4
    sess.run(tf.global_variables_initializer())
    while True:
        sess.run(trn_init_op)
        sess.run(tf.local_variables_initializer())

        while True:
            try:
                sess.run(trn_model.train_op)
                count += 1

            except tf.errors.OutOfRangeError:
                epoch += 1
                print('train_epoch', epoch)

                summary = sess.run(trn_model.summary)
                summary_writer.add_summary(summary, epoch)

                sess.run(test_init_op)
                while True:
                    try:
                        test_loss = sess.run(test_model.update_op_list)
                    except tf.errors.OutOfRangeError:
                        break
                # summary = sess.run(test_model.summary)
                # summary_writer.add_summary(summary, epoch)

                train_loss = sess.run(trn_model.summary_list)

                if test_loss['loss'] < best_test_loss:
                    saver.save(sess, 'model/{}/model.ckpt'.format(model_dir),
                               global_step=epoch)
                    best_test_loss = test_loss['loss']

                    summary = sess.run(test_model.summary)
                    summary_writer.add_summary(summary, epoch)

                print('test_loss: {} train_loss: {}'.format(test_loss, train_loss))
                if epoch > 50 and test_loss['loss'] > train_loss['loss'] * 1.5:
                    raise OverfitError('overfit')
                break