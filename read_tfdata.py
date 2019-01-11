import os
import tensorflow as tf

def read_and_decode(filename, batch_size):
	file_queue = tf.train.string_input_producer([filename])

	reader = tf.TFRecordReader()
	_, serialized_example = reader.read(file_queue)
	features = tf.parse_single_example(serialized_example,
		                               features={
		                               'label': tf.FixedLenFeature([], tf.int64),
		                               'img_raw': tf.FixedLenFeature([], tf.string)
		                               # 'height': tf.FixedLenFeature([], tf.int64),
		                               # 'width': tf.FixedLenFeature([], tf.int64),
		                               # 'channel': tf.FixedLenFeature([], tf.int64)
		                               })

	img = tf.decode_raw(features['img_raw'], tf.uint8)
	# h = tf.cast(features['height'], tf.int32)
	# w = tf.cast(features['width'], tf.int32)
	# c = tf.cast(features['channel'], tf.int32)
	img = tf.reshape(img, [150,150,3])
	label = tf.cast(features['label'], tf.int32)

	img_batch, label_batch = tf.train.shuffle_batch([img,label],
                                                   batch_size = batch_size,
                                                   num_threads=64,#why??
                                                   capacity = 2000,
                                                   min_after_dequeue = 1500
		                                           )

	return img_batch, tf.reshape(label_batch,[batch_size])


if __name__ == '__main__':
    filename = 'kodak.tfrecords'
    batch_size = 4
    img_batch, label_batch = read_and_decode(filename, batch_size)
