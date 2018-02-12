# epoch 단위로 안하기?
# final 그림 그리기
import time
import collections
from inputs_single import inputs
from model import SimpleModel as Model
import tensorflow as tf
import os
import argparse


class OverfitError(Exception):
    pass

class MaxEpoch(Exception):
    pass

parser = argparse.ArgumentParser()
parser.add_argument('-trn_site', type=str)
parser.add_argument('-test_site', type=str)
parser.add_argument('-mlp_layer', type=int, nargs='+')
parser.add_argument('-mode_size', type=int, default=6)
parser.add_argument('-embedding_size', type=int)
parser.add_argument('-batch_size', type=int, default=128)
parser.add_argument('-max_epoch', type=int, default=50)
parser.add_argument('-weights', type=int)
parser.add_argument('-option', type=str)

args = parser.parse_args()

trn_site = args.trn_site
test_site = args.test_site
mlp_layer = args.mlp_layer
mode_size = args.mode_size
embedding_size = args.embedding_size
batch_size = args.batch_size
max_epoch = args.max_epoch
weights = args.weights
option = args.option


with tf.variable_scope('Model'):
    with tf.name_scope('Train'):
        trn_inputs, trn_init_op = inputs(trn_site, batch_size, 5)
        trn_model = Model(trn_inputs, mlp_layer, mode_size, embedding_size, weights)

with tf.variable_scope('Model', reuse=True):
    with tf.name_scope('Test'):
        test_inputs, test_init_op = inputs(test_site, batch_size, 5)
        test_model = Model(test_inputs, mlp_layer, mode_size, embedding_size, weights)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

saver = tf.train.Saver()

if not os.path.exists('model'):
    os.mkdir('model')

model_config = collections.OrderedDict({'trn_site': trn_site,
                'test_site': test_site,
                'mlp_layer' : '-'.join([str(x) for x in mlp_layer]),
                'embedding_size' : str(embedding_size),
                'weights' : str(weights),
                'option' : str(option)
                })

model_dir = ['{}-{}'.format(key, model_config[key]) for key in model_config.keys()]
model_dir = '_'.join(model_dir)

with tf.Session(config=config) as sess:
    summary_writer = tf.summary.FileWriter('summary/{}'.format(model_dir), sess.graph,
                                           flush_secs=10)

    count = 0
    epoch = 0
    best_test_loss_s = 1e4
    best_test_loss_l = 1e4
    sess.run(tf.global_variables_initializer())
    while True:
        sess.run(trn_init_op)
        sess.run(tf.local_variables_initializer())
        start = time.time()
        while True:
            try:
                if epoch == max_epoch:
                    raise MaxEpoch('max epoch')
                sess.run([trn_model.train_op_s, trn_model.train_op_l])

                count += 1

            except tf.errors.OutOfRangeError:
                epoch += 1
                print('train_epoch', epoch, 'took {:2f} mintues'.format((time.time() -
                                                                      start)/ 60) )


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

                if test_loss['cost_s'] < best_test_loss_s:
                    saver.save(sess, 'model/{}/model_s.ckpt'.format(model_dir),
                               global_step=epoch)
                    best_test_loss_s = test_loss['cost_s']

                    summary = sess.run(test_model.summary)
                    summary_writer.add_summary(summary, epoch)



                if epoch > 20 and test_loss['cost_s'] > train_loss['cost_s'] * 1.5:
                    raise OverfitError('overfit')

                if test_loss['cost_l'] < best_test_loss_l:
                    saver.save(sess, 'model/{}/model_l.ckpt'.format(model_dir),
                               global_step=epoch)
                    best_test_loss_l = test_loss['cost_l']

                    summary = sess.run(test_model.summary)
                    summary_writer.add_summary(summary, epoch)

                print('test_loss_l: {} train_loss_l: {}'.format(test_loss, train_loss))
                if epoch > 20 and test_loss['cost_l'] > train_loss['cost_l'] * 1.5:
                    raise OverfitError('overfit')
                break
