# patience

import itertools
import numpy as np
import os
import logging
import tensorflow as tf


def get_logger(data_dir, model_config):
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    keys_to_remove = [x for x in model_config.keys() if not x]
    for key in keys_to_remove:
        model_config.pop(key)

    # create a file handler

    model_dir = '_'.join(str(key) + '-' + str(val) for key, val in model_config.items())

    if not os.path.exists(data_dir + '/log/' + model_dir):
        os.makedirs(data_dir + '/log/' + model_dir)

    if not os.path.exists(data_dir + '/summary/' + model_dir):
        os.makedirs(data_dir + '/summary/' + model_dir)

    handler = logging.FileHandler(data_dir + '/log/' + model_dir + '/log.txt')
    handler.setLevel(logging.INFO)

    # create a logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(handler)
    return logger, model_dir


class Model():
    def __init__(self, data_inputs, rnn_cell, rnn_layer, output_layer, keep_prob,
                 share_encoder, no_mode):
        inputs = data_inputs['inputs']  # [batch_size, time_step, dim]

        if no_mode:
            inputs_1 = inputs[:, :, :3]
            inputs_2 = inputs[:, :, 4:]
            inputs = tf.concat([inputs_1, inputs_2], axis=2)

        targets = data_inputs['targets']  # [batch_size, 1, classes]


        def lstm_cell(rnn_hidden):
            return tf.contrib.rnn.LSTMCell(rnn_hidden)

        def gru_cell(rnn_hidden):
            return tf.contrib.rnn.GRUCell(rnn_hidden)

        def layernorm_lstm_cell(rnn_hidden):
            return tf.contrib.rnn.LayerNormBasicLSTMCell(rnn_hidden, forget_bias=1.0,
                                                         norm_gain=1.0, norm_shift=0.0,
                                                         dropout_keep_prob=keep_prob)

        def stacked_rnn_cell():
            if rnn_cell == 'lstm':
                return tf.contrib.rnn.MultiRNNCell([lstm_cell(i) for i in rnn_layer])
            elif rnn_cell == 'gru':
                return tf.contrib.rnn.MultiRNNCell([gru_cell(i) for i in rnn_layer])
            else:
                return tf.contrib.rnn.MultiRNNCell(
                    [layernorm_lstm_cell(i) for i in rnn_layer])

        def rnn(inputs, stacked_rnn_cell):
            outputs, output_state_fw, output_state_bw = \
                tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                    [stacked_rnn_cell()],
                    [stacked_rnn_cell()],
                    inputs,
                    dtype=tf.float32
                )

            rnn_output = outputs[:, -1,
                         :]  # outputs shape : [batch_size, time_step, rnn_hidden*2] -> [batch_size, rnn_hidden*2]
            return rnn_output

        def mlp_scalar(inputs, layer_num):
            layer_list = [inputs]
            for hidden_dim in layer_num:
                layer_inputs = layer_list[-1]
                layer_list.append(
                    tf.contrib.layers.fully_connected(layer_inputs, hidden_dim))
            output = tf.contrib.layers.fully_connected(layer_list[-1], 1, None)
            return output

        if share_encoder:
            rnn_output = rnn(inputs, stacked_rnn_cell)
            pred_s = mlp_scalar(rnn_output, output_layer)
            pred_l = mlp_scalar(rnn_output, output_layer)
        else:
            with tf.variable_scope('load_s'):
                rnn_output_s = rnn(inputs, stacked_rnn_cell)
                pred_s = mlp_scalar(rnn_output_s, output_layer)
            with tf.variable_scope('load_l'):
                rnn_output_l = rnn(inputs, stacked_rnn_cell)
                pred_l = mlp_scalar(rnn_output_l, output_layer)


        pred = tf.concat([pred_s, pred_l], 1)
        self.pred = tf.expand_dims(pred, 1)

        def mape(pred, target, threshold):
            mask = tf.abs(target) > threshold
            pred = tf.boolean_mask(pred, mask)
            target = tf.boolean_mask(target, mask)
            result = tf.abs((pred - target) / target)
            return result

        self.loss = tf.losses.mean_squared_error(targets, self.pred)
        self.mape_1 = mape(pred_s, targets[:, :, 0], 0.5)
        self.mape_2 = mape(pred_l, targets[:, :, 1], 0.5)

        self.summary_list, self.update_op_list = self.make_summary({'loss': self.loss,
                                                                    'mape_s': self.mape_1,
                                                                    'mape_l': self.mape_2})

        with tf.control_dependencies(self.update_op_list.values()):
            self.train_op = tf.train.AdamOptimizer().minimize(self.loss)

        self.summary = tf.summary.merge([tf.summary.scalar(key, val) for key, val in \
                                         self.summary_list.items()])

    def make_summary(self, variable_list):
        summary_list = dict()
        update_op_list = dict()
        for key, value in variable_list.items():
            mean, update_op = tf.contrib.metrics.streaming_mean(value,
                                                                metrics_collections='loss',
                                                                updates_collections='update')
            update_op_list[key] = update_op
            summary_list[key] = mean
        return summary_list, update_op_list


# class CNNModel():
#     def __init__(self, data_inputs, cnn_layer, output_layer, keep_prob):
#         inputs = data_inputs['inputs']  # [batch_size, time_step, dim]
#         targets = data_inputs['targets']  # [batch_size, 1, classes]
#
#         feature_dim = inputs.shape.as_list()[2]
#
#         def cnn(inputs, cnn_layer):
#             layer_list = [inputs]
#             inputs = tf.expand_dims(inputs, 3)
#             for kernel, stride, output_dim in cnn_layer:
#                 kernel = (kernel, feature_dim)
#                 layer_inputs = layer_list[-1]
#                 tf.contrib.layers.conv2d(layer_inputs, kernel)
#
#
#         def mlp_scalar(inputs, layer_num):
#             layer_list = [inputs]
#             for hidden_dim in layer_num:
#                 layer_inputs = layer_list[-1]
#                 layer_list.append(
#                     tf.contrib.layers.fully_connected(layer_inputs, hidden_dim))
#             output = tf.contrib.layers.fully_connected(layer_list[-1], 1, None)
#             return output
#
#         with tf.variable_scope('load_s'):
#             rnn_output_s = rnn(inputs, stacked_rnn_cell)
#             pred_s = mlp_scalar(rnn_output_s, output_layer)
#         with tf.variable_scope('load_l'):
#             rnn_output_l = rnn(inputs, stacked_rnn_cell)
#             pred_l = mlp_scalar(rnn_output_l, output_layer)
#
#         pred = tf.concat([pred_s, pred_l], 1)
#         self.pred = tf.expand_dims(pred, 1)
#
#         def mape(pred, target, threshold):
#             mask = tf.abs(target) > threshold
#             pred = tf.boolean_mask(pred, mask)
#             target = tf.boolean_mask(target, mask)
#             result = tf.abs((pred - target) / target)
#             return result
#
#         self.loss = tf.losses.mean_squared_error(targets, self.pred)
#         self.mape_1 = mape(pred_s, targets[:, :, 0], 0.5)
#         self.mape_2 = mape(pred_l, targets[:, :, 1], 0.5)
#
#         self.summary_list, self.update_op_list = self.make_summary({'loss': self.loss,
#                                                                     'mape_s': self.mape_1,
#                                                                     'mape_l': self.mape_2})
#
#         with tf.control_dependencies(self.update_op_list.values()):
#             self.train_op = tf.train.AdamOptimizer().minimize(self.loss)
#
#         self.summary = tf.summary.merge([tf.summary.scalar(key, val) for key, val in \
#                                          self.summary_list.items()])
#
#     def make_summary(self, variable_list):
#         summary_list = dict()
#         update_op_list = dict()
#         for key, value in variable_list.items():
#             mean, update_op = tf.contrib.metrics.streaming_mean(value,
#                                                                 metrics_collections='loss',
#                                                                 updates_collections='update')
#             update_op_list[key] = update_op
#             summary_list[key] = mean
#         return summary_list, update_op_list





class SimpleModel():
    def __init__(self, data_inputs, mlp_layer,
                 vocab_size, embedding_size, weights):
        self.float_inputs = data_inputs['float_vars']
        self.int_inputs = data_inputs['int_vars']
        self.targets = data_inputs['target_vars']

        self.weights = tf.cast(tf.not_equal(self.int_inputs, 0), tf.float32) * weights + 1
        # self.weights = tf.ones_like(self.int_inputs)

        self.target_s = self.targets[:,0]
        self.target_l = self.targets[:,1]

        self.embedding_vector = tf.get_variable('embed_var',
                                           shape=(vocab_size, embedding_size))

        self.mode_var_embedding = tf.nn.embedding_lookup(self.embedding_vector,
                                                         self.int_inputs)

        inputs = tf.concat([self.float_inputs, self.mode_var_embedding], axis=1)

        def mlp_scalar(inputs, layer_num):
            layer_list = [inputs]
            for hidden_dim in layer_num:
                layer_inputs = layer_list[-1]
                layer_list.append(
                    tf.contrib.layers.fully_connected(layer_inputs, hidden_dim))
            output = tf.contrib.layers.fully_connected(layer_list[-1], 1, None)
            output = tf.squeeze(output)
            return output

        def mape(pred, target, threshold):
            mask = tf.abs(target) > threshold
            pred = tf.boolean_mask(pred, mask)
            target = tf.boolean_mask(target, mask)
            result = tf.abs((pred - target) / target)
            return result

        self.pred_s = mlp_scalar(inputs, mlp_layer)
        self.pred_l = mlp_scalar(inputs, mlp_layer)

        self.mape_s_3 = mape(self.pred_s, self.target_s, 1e-3)
        self.mape_l_3 = mape(self.pred_l, self.target_s, 1e-3)

        self.mape_s_2 = mape(self.pred_s, self.target_s, 1e-2)
        self.mape_l_2 = mape(self.pred_s, self.target_l, 1e-2)

        # self.mse_l = tf.losses.mean_squared_error(self.target_s, self.pred_s)
        # self.mse_s = tf.losses.mean_squared_error(self.target_l, self.pred_l)

        self.cost_l = tf.losses.mean_squared_error(self.target_s, self.pred_s,
                                                   self.weights)
        self.cost_s = tf.losses.mean_squared_error(self.target_l, self.pred_l,
                                                   self.weights)

        self.mae_s = tf.losses.absolute_difference(self.target_s, self.pred_s)
        self.mae_l = tf.losses.absolute_difference(self.target_l, self.pred_l)

        self.summary_list, self.update_op_list = self.make_summary({'mae_s':
                                                                        self.mae_s,
                                                                    'mae_l':
                                                                        self.mae_l,
                                                                    'mape_s_3' :
                                                                    self.mape_s_3,
                                                                    'mape_l_3':
                                                                    self.mape_l_3,
                                                                    'mape_s_2':
                                                                    self.mape_s_2,
                                                                    'mape_l_2':
                                                                    self.mape_l_2,
                                                                    'cost_l':
                                                                    self.cost_l,
                                                                    'cost_s':
                                                                    self.cost_s
                                                                    },
                                                                   )

        with tf.control_dependencies(self.update_op_list.values()):
            self.train_op_s = tf.train.AdamOptimizer().minimize(self.cost_s)
            self.train_op_l = tf.train.AdamOptimizer().minimize(self.cost_l)

        self.summary = tf.summary.merge([tf.summary.scalar(key, val) for key, val in \
                                         self.summary_list.items()])

    def make_summary(self, variable_list):
        summary_list = dict()
        update_op_list = dict()
        for key, value in variable_list.items():
            mean, update_op = tf.contrib.metrics.streaming_mean(value,
                                                                metrics_collections='loss',
                                                                updates_collections='update')
            update_op_list[key] = update_op
            summary_list[key] = mean
        return summary_list, update_op_list
