
import re
import tensorflow as tf
import os
import numpy as np
import data_preprocess
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def make_example(float_vars, int_vars, target_vars):

    int_vars = int_vars.astype(np.int64)
    float_vars = float_vars.astype(np.float32)
    target_vars = target_vars.astype(np.float32)

    float_features = _float_feature(float_vars)
    int_features = _int64_feature(int_vars)
    target_features = _float_feature(target_vars)

    feature = {
        'float_vars': float_features,
        'int_vars': int_features,
        'target_vars': target_features
                    }
    features = tf.train.Features(feature=feature)
    ex = tf.train.Example(features=features)
    return ex


def make_tfrecords(data_type, filename):
    numpy_data = 'data/{}/data.npz'.format(data_type)

    if not os.path.exists(numpy_data):
        print('making numpy data', data_type)
        data_preprocess.make_data_single('data/{}'.format(data_type))

    print('loading numpy data', data_type)
    data = np.load(numpy_data)

    float_inputs = data['float_vars']
    int_inputs = data['int_vars']
    targets = data['target_vars']

    print('making tfrecord data')
    writer = tf.python_io.TFRecordWriter(filename)

    for float_vars, int_vars, target_vars in zip(float_inputs, int_inputs, targets):
        ex = make_example(float_vars, int_vars, target_vars)
        writer.write(ex.SerializeToString())
    writer.close()

    print('tfrecord data made')

def read_and_decode(example_proto):
    print('Reading and Decoding')
    features =  {'float_vars' : tf.FixedLenFeature([8], tf.float32),
                 'int_vars' : tf.FixedLenFeature([], tf.int64),
                 'target_vars' : tf.FixedLenFeature([2], tf.float32)}

    feature_data_parsed = tf.parse_single_example(
        example_proto,
        features=features
        )

    return feature_data_parsed

def inputs(data_type, batch_size, num_parallel_calls):
    filename = 'data/{}/data.tfrecords'.format(data_type)

    if not os.path.exists(filename):
        make_tfrecords(data_type, filename)

    dataset = tf.data.TFRecordDataset(filename)
    dataset = dataset.map(read_and_decode, num_parallel_calls)
    # dataset = dataset.shuffle(buffer_size =10000)
    dataset = dataset.batch(batch_size)
    dataset.repeat()
    iterator = tf.contrib.data.Iterator.from_structure(dataset.output_types,
                                       dataset.output_shapes)
    next_element = iterator.get_next()
    init_op = iterator.make_initializer(dataset)

    return next_element, init_op


def test():
    # make_tfrecords('train')
    input_data_stream, init_op = inputs('seed0_test', 128, 10)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    sess.run(init_op)
    try:
        while True:
            data = sess.run(input_data_stream)
            print(data.keys())
            break

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')

    sess.close()

if __name__ == '__main__':
    test()



