import tensorflow as tf


# The following functions can be used to convert a value to a type compatible with tf.train.Example.
# stolen from https://www.tensorflow.org/tutorials/load_data/tfrecord
def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


@tf.py_function(Tout=tf.string)
def serialize_image_patch(hls_atl08_arr, patch_size, num_bands):
    arr_ser = tf.io.serialize_tensor(hls_atl08_arr)
    feature = {
        # 'height': _int64_feature(patch_size),
        # 'width': _int64_feature(patch_size),
        # 'depth': _int64_feature(num_bands),
        'arr': _bytes_feature(arr_ser)
    }
    ex = tf.train.Example(features=tf.train.Features(feature=feature))
    return ex.SerializeToString()
