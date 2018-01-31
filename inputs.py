
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


def make_sequence_example(input_vars, target_vars):

    input_features = [_float_feature(value=input_) for input_ in input_vars]
    target_features = [_float_feature(target_) for target_ in [target_vars]]
    feature_list = {
        'inputs': tf.train.FeatureList(feature=input_features),
        'targets': tf.train.FeatureList(feature=target_features)
                    }
    feature_lists = tf.train.FeatureLists(feature_list=feature_list)
    ex = tf.train.SequenceExample(feature_lists=feature_lists)
    return ex


def make_tfrecords(data_type, int_mode, filename):
    numpy_data = 'data/{}/data.npz'.format(data_type)
    if not os.path.exists(numpy_data):
        print('making numpy data', data_type)
        data_preprocess.make_data('data/{}'.format(data_type))
    print('loading numpy data', data_type)
    data = np.load(numpy_data)

    def isinteger(x):
        return np.equal(np.mod(x, 1), 0)

    if int_mode:
        mask = isinteger(data['X'][:,:,3]).all(axis=1)
        inputs = data['X'][mask]
        targets = data['y'][mask]
    else:
        inputs = data['X']
        targets = data['y']


    writer = tf.python_io.TFRecordWriter(filename)

    for input_vars, target_vars in zip(inputs, targets):
        ex = make_sequence_example(input_vars, target_vars)
        writer.write(ex.SerializeToString())
    writer.close()

    print('tfrecord made')

def read_and_decode(example_proto):
    print('Reading and Decoding')
    features =  {'inputs' : tf.FixedLenSequenceFeature([9], tf.float32),
                 'targets' : tf.FixedLenSequenceFeature([2], tf.float32)}


    context_data_parsed, feature_data_parsed = tf.parse_single_sequence_example(
        example_proto,
        sequence_features=features
        )

    return feature_data_parsed

def inputs(data_type, int_mode, batch_size, num_threads):
    if int_mode:
        filename = 'data/{}/data_int.tfrecords'.format(data_type)
    else:
        filename = 'data/{}/data.tfrecords'.format(data_type)

    if not os.path.exists(filename):
        make_tfrecords(data_type, int_mode, filename)

    dataset = tf.contrib.data.TFRecordDataset(filename)
    dataset = dataset.map(read_and_decode, num_threads)
    dataset = dataset.shuffle(buffer_size =10000)
    dataset = dataset.batch(batch_size)
    dataset.repeat()
    iterator = tf.contrib.data.Iterator.from_structure(dataset.output_types,
                                       dataset.output_shapes)
    next_element = iterator.get_next()
    init_op = iterator.make_initializer(dataset)

    return next_element, init_op


def test():
    # make_tfrecords('train')
    input_data_stream, init_op = inputs('test', 128, 10, )

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


