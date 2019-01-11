import os
import tensorflow as tf 
from PIL import Image
import matplotlib.pyplot as plt 
import numpy as np 

in_path = "./picture/150*150imagenet1/"
classes = {'150*150imagenet1'}
writer = tf.python_io.TFRecordWriter('150*150imagenet1.tfrecords')

def _int64_feature(value):
	return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))

def _bytes_feature(value):
	return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))

for img_name in os.listdir(in_path):
	img_path = in_path + img_name

	img = Image.open(img_path)
	#img = img.resize((150,150))
	#img = img.astype(np.uint8)
	#print(img.dtype)

	img_raw = img.tobytes()
	example = tf.train.Example(features = tf.train.Features(feature = {
		"label": _int64_feature(0),
		"img_raw": _bytes_feature(img_raw)
		# "height": _int64_feature(h),
		# "width": _int64_feature(w),
		# "channel": _int64_feature(c)
		}))

	writer.write(example.SerializeToString())

writer.close()
