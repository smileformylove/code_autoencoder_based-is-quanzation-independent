import os
import tensorflow as tf 
import matplotlib.pyplot as plt 
import numpy as np 
from PIL import Image
from read_tfdata import read_and_decode

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
filename = 'kodak.tfrecords'
BATCH_SIZE = 3

image_batch, label_batch = read_and_decode(filename,BATCH_SIZE)

with tf.Session() as sess:
	i = 0
	sess.run(tf.global_variables_initializer())
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)
	#image_batch = tf.squeeze(image_batch)

	try:
		while not coord.should_stop() and i<1:
			#print(image_batch)
			image, label = sess.run([image_batch, label_batch])

			for j in np.arange(BATCH_SIZE):
				print('label: %d' % label[j])
				plt.subplot(1,BATCH_SIZE,j+1)
				plt.imshow(image[j,:,:,:], cmap='gray')
				plt.axis('off')
				#image[j,:,:] = Image.fromarray(image[j,:,:])
				#image[j,:,:].save("1.jpeg") #""

			i += 1
		plt.show()

	except tf.errors.OutOfRangeError:
	    print('done!')

	finally:
	    coord.request_stop()

	coord.join(threads)
