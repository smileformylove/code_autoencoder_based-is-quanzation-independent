import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt
import os
import math
import scipy.special

#from PIL import Image
from fjcommon import config_parser
from data.read_tfdata import read_and_decode
from oppo_reader import Net, mask_pro
from data import probclass

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())
logs_train_dir = './log_beta_1_192_ow'
batch_size = 50
MAX_step = 10000
learn_rate = 1e-4
pi = math.pi
lamda = 5
beta = 10
base_rate = 0
delt_range = 1e-2

# @tf.RegisterGradient("QuantizerGrad")
# def round_grad(op, grad):
# 	input = op.inputs[0]
# 	cond = False
# 	#cond = (input >= -1) & (input <= 1)
# 	ones = tf.ones_like(grad)
# 	#zeros = tf.zeros_like(grad)
# 	return tf.where(cond, grad, grad)

# def round(input):
# 	x = input
# 	with tf.get_default_graph().gradient_override_map({"Round":'QuantizerGrad'}):
# 		x = tf.round(x)

# 	return x

# def sign(input):
# 	x = input
# 	with tf.get_default_graph().gradient_override_map({"Sign":'QuantizerGrad'}):
# 		x = tf.sign(x)

# 	return x

def init_probclass():
	config_path = './data/config/base'
	pc_config, pc_config_real_path = config_parser.parse(config_path)
	pc_cls = probclass.get_network_cls(pc_config)
	pc = pc_cls(pc_config, pc_config.num_centers)

	return  pc

def get_batch(batch_size):
	file_name1 = './data/150*150imagenet1.tfrecords'
	file_name2 = './data/150*150img.tfrecords'
	file_name3 = './data/kodak.tfrecords'
	file_name4 = './data/imagenet256.tfrecords'
	img_batch1, label_batch1 = read_and_decode(file_name4, batch_size)
	#img_batch2, label_batch2 = read_and_decode(file_name2, 3)
	#uint8->float32
	img_batch1 = tf.divide(tf.cast(img_batch1,tf.float32), 255)
	#img_batch2 = tf.divide(tf.cast(img_batch2,tf.float32), 255)

	return img_batch1#, img_batch2

def train():
	image = get_batch(batch_size)
	#X_in = tf.placeholder(dtype=tf.float32, shape=None, name='X_in')
	#X = tf.reshape(X_in, shape=[-1, 256, 256, 3], name='X_reshape')
        
	with tf.Session() as sess:
		pc=init_probclass()

		net = Net(image)
		feature = net.encoder()
		with tf.variable_scope("data"):
			qbar, _, symbol, centers = net.quantization(feature)
			#qbar = round(feature)
		r_image = net.decoder(qbar)
		#entropy estimate
		enc_entropy = net.hyper_enc(qbar)
		with tf.variable_scope("hyper"):
			q_entropy, _, q_symbol, q_centers = net.quantization(enc_entropy)
			#q_entropy = round(enc_entropy)
		dec_entropy_mu = net.hyper_dec(q_entropy)
		dec_entropy_std = net.generate_std(dec_entropy_mu)
		
		#decrease the number of distribution
		# dec_entropy_mu  = tf.reduce_mean(dec_entropy_mu, axis=0)
		# dec_entropy_std = tf.reduce_mean(dec_entropy_std, axis=0)

		#entropy for code(symbol)		
		#std = (qbar - dec_entropy_mu)/dec_entropy_std
		std = (qbar - dec_entropy_mu)/dec_entropy_std
		#std = tf.reduce_mean(tf.reduce_mean(tf.reduce_mean(std, axis=0), axis=0), axis=0)

		pro = 1/(tf.sqrt(2*pi)) * tf.exp(-(tf.square(std)/2)) * delt_range 
		#pro = get_freq(qbar, dec_entropy_mu, dec_entropy_std)
		#pro = tf.reduce_mean(tf.reduce_mean(tf.reduce_mean(pro, axis=0), axis=0), axis=0)
		rate_entropy = tf.reduce_mean(tf.log(pro)/tf.log(2.0))

		#rate_entropy for hyper(symbol)
		#q_pro = net.density_model(q_entropy)
		q_std = net.non_para(q_entropy)
		q_pro = 1/(tf.sqrt(2*pi)) * tf.exp(-(tf.square(q_std)/2)) * delt_range
		q_rate_entropy = tf.reduce_mean(tf.log(q_pro)/tf.log(2.0)) 

		q_in = tf.stop_gradient(q_entropy)
		q_centers_loss = pc.bitcost(q_in, q_symbol, is_training=True, pad_value=q_centers[0])
		q_centers_loss = tf.reduce_mean(q_centers_loss)

		img_loss = tf.reduce_sum(tf.squared_difference(image, r_image))
		pc_in = tf.stop_gradient(qbar)
		centers_loss = pc.bitcost(pc_in, symbol, is_training=True, pad_value=centers[0])
		centers_loss = tf.reduce_mean(centers_loss) + q_centers_loss
		centers_loss = tf.abs(centers_loss - base_rate)  ##to estimate the bit rate
		loss = img_loss / batch_size * lamda + centers_loss * beta - (rate_entropy  + q_rate_entropy)
		#loss = loss / batch_size

		global_step = tf.Variable(0, name="global_step", trainable=False)
		decayed_learning_rate = tf.train.exponential_decay(learn_rate,
					                           global_step,
					                           decay_steps = 600,
					                           decay_rate = 0.5,
					                           staircase = True)
		#optimizer = tf.GradientDescent(decayed_learning_rate)
		optimizer = tf.train.AdamOptimizer(learning_rate = decayed_learning_rate)
		optimizer = tf.train.AdamOptimizer(decayed_learning_rate).minimize(loss, global_step=global_step)
		
		#save the model
		train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
		saver = tf.train.Saver()

		#sess = tf.Session()
		sess.run(tf.global_variables_initializer())
		#sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=sess, coord=coord)
		#batch = sess.run(image)
		#batch_test = sess.run(test_image)
		#batch = image.eval()
		#batch_test = test_image.eval()
		#continue 
		# ckpt = tf.train.get_checkpoint_state(logs_train_dir)
		# if ckpt and ckpt.model_checkpoint_path:
		# 	saver.restore(sess, ckpt.model_checkpoint_path)
		# 	global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

		try:
			i = 0
			for step in range(MAX_step):
				if coord.should_stop():
					break;

				sess.run(optimizer)
				#mask = mask_pro(std)

				if step % 50 == 0 and step > 0:
					ls, lr, bit_ls, p, ff= sess.run([loss, decayed_learning_rate, centers_loss, pro, feature])
					print("step:%d lr:%.7f loss=%.5f bit:%.5f" % (step, lr, ls, bit_ls+base_rate))
					#print("centers:", cen)
					#print("p:", p[0][0][0])
					#print('feature:', np.max(ff), np.min(ff))

				if (step % 4000 == 0 and step > 0) or step == 10000-1:
					#batch_test = sess.run(image)
					checkpoint_path = os.path.join(logs_train_dir,'oppo.ckpt')
					saver.save(sess, checkpoint_path, global_step = step)

			# while not coord.should_stop() and i<1:
			# 	r_img = sess.run(tf.cast(tf.multiply(r_image, 255), tf.uint8), feed_dict = {X_in: batch})
			# 	img = (batch*255).astype(int)

			# 	for j in np.arange(batch_size):
			# 		plt.subplot(2,batch_size,j+1)
			# 		plt.imshow(img[j,:,:,:])
			# 		plt.axis('off')

			# 		plt.subplot(2,batch_size,j+1+batch_size)
			# 		plt.imshow(r_img[j,:,:,:])
			# 		plt.axis('off')
			# 	i += 1
			# plt.show()

		except tf.errors.OutOfRangeError:
			print('done!')
		finally:
			coord.request_stop()
		coord.join(threads)
		sess.close()

def main():
	train()

if __name__ == '__main__':
	main()
