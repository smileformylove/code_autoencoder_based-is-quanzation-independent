import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt
import time
import os
import math
#import codec.Codec as cc

from PIL import Image
#from oppo import Net
from oppo_reader import Net
from msssim import MultiScaleSSIM
from code_beta import Codec

os.environ['CUDA_VISIBLE_DEVICES'] = '1' 
model_path = './log_beta_1_192_ow/'
bit_path = './result/out.bin'

pic1 = './1.png'
pic2 = './4.jpg'
pic3 = './3.png'
pic = pic1
code = Codec(pic, bit_path)

with tf.Graph().as_default() as g:
	image = tf.gfile.FastGFile(pic, 'rb').read()
	X_in = tf.placeholder(tf.string, shape=None)
	X = tf.image.decode_jpeg(X_in, channels=3)
	#input_x = code.pre_pic()
	X = tf.cast(tf.expand_dims(X, axis = 0), tf.float32)
	X = tf.divide(tf.cast(X, tf.float32), 255)
	net = Net(X) #initializer the network
	feature = net.encoder()
	qbar = tf.round(feature)
	r_image = net.decoder(qbar)
	mean, std = net.hyper_net(qbar)
	pro = (qbar - mean) / std
	img_1, img_2 = tf.split(qbar[:,:,:,:6], [3, 3], axis=-1)
	img_3, img_4 = tf.split(pro[:,:,:,:6], [3, 3], axis=-1)

	with tf.Session() as sess:
		ckpt = tf.train.get_checkpoint_state(model_path)
		saver = tf.train.Saver()
		if ckpt and ckpt.model_checkpoint_path:
			print(ckpt.model_checkpoint_path)
			saver.restore(sess, ckpt.model_checkpoint_path)
		img1, img2, img3, img4, image_r = sess.run([img_1, img_2, img_3, img_4, r_image], feed_dict={X_in:image})
		y_hat, mu_y, sigma_y = sess.run([qbar, mean, std], feed_dict={X_in:image})
		print('max:', np.max(y_hat))
		print('min:', np.min(y_hat))
		print('mean:', mu_y)
		print('sigma:', sigma_y)

		# statistics for encoding data
		ss = np.ravel(y_hat)
		ss = sorted(ss)
		unique_data = np.unique(ss)
		resdata = []
		for i in unique_data:
			resdata.append(ss.count(i))
		fig = plt.figure(1)
		plt.plot(unique_data, resdata)
		plt.show()
		plt.savefig('statistics.png')

		time1 = time.time()
		code.encoder(y_hat, mu_y, sigma_y)
		time2 = time.time()
		print('coding time:', time2 - time1)
		rr_image = code.decoder(mu_y, sigma_y)
		time3 = time.time()
		print('decoding time:', time3 - time2)
		print('code loss:', np.sum(y_hat - rr_image))
		image_r = (image_r*255).astype(int)
		img1 = (img1*255).astype(int)
		img2 = (img2*255).astype(int)
		img3 = (img3*255).astype(int) ##layer_0 show
		img4 = (img4*255).astype(int)

img_o = Image.open(pic)
_, h, w, _ = image_r.shape
def psnr(img1, img2):
	mse = np.mean((img1 - img2[0,:,:,:]) ** 2)
	if mse == 0:
		return 100
	# PIXEL_MAX_H =  h
	# PIXEL_MAX_W = w
	return 10 * math.log10(255*255 / mse)

def ms_ssim(img1, img2):
    img1 = np.expand_dims(img1, 0)
    #img2 = np.expand_dims(img2, 0)
    return -10 * math.log10(1-MultiScaleSSIM(img1, img2))

print(pic)
print('psnr:', psnr(img_o, image_r))
print('msssim:', ms_ssim(img_o, image_r))
#print('quantization:', points)

plt.figure(2)
plt.subplot(2, 3, 1)
plt.imshow(img_o)
plt.axis('off')
plt.subplot(2, 3, 4)
plt.imshow(image_r[0,:,:,:])
plt.axis('off')
###
plt.subplot(2, 3, 2)
plt.imshow(img1[0,:,:,:])
plt.axis('off')
plt.subplot(2, 3, 5)
plt.imshow(img2[0,:,:,:])
plt.axis('off')
plt.subplot(2, 3, 3)
plt.imshow(img3[0,:,:,:])
plt.axis('off')
plt.subplot(2, 3, 6)
plt.imshow(img4[0,:,:,0])
plt.axis('off')
plt.savefig('show2.png')
plt.show()