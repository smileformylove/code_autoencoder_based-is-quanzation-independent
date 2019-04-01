import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt
import os
import math
import copy
import scipy.special

#from PIL import Image
#from fjcommon import config_parser
from data.read_tfdata import read_and_decode
from oppo_reader import Net, mask_pro
from data import probclass

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())
logs_train_dir = './log_beta_1_192_ow'
batch_size = 60
MAX_step = 10000
learn_rate = 1e-4
pi = math.pi
lamda = 1e3
beta = 1e1
base_rate = 1
delt_range = 1e-7
step = 8
TINY = 1e-10

# @tf.RegisterGradient("QuantizerGrad")
# def round_grad(op, grad):
# 	input = op.inputs[0]
# 	return grad 

# def round(input):
# 	x = input
# 	y = tf.identity(x)
# 	with tf.get_default_graph().gradient_override_map({"Round":'QuantizerGrad'}):
# 		#x = tf.round(x, name='Round')
# 		a, b, c, d = x.get_shape().as_list()
# 		prob = np.random.rand(a, b, c, d) 
# 		ones = tf.ones_like(x)
# 		n_ones = tf.zeros_like(x) 
# 		zeros = tf.zeros_like(x)
# 		cond1 = (1 - x) / 4 <= prob # -1
# 		cond2 = (1 - x) / 4 * 3 > prob # 1
# 		cond3 = ((1 - x) / 4 > prob) & ((1 - x) / 4 * 3 <= prob) # 0
# 		# y = tf.where(cond1, x, n_ones)
# 		# y = tf.where(cond2, x, ones)
# 		# y = tf.where(cond3, x, zeros)
# 		y_hat = tf.stop_gradient(tf.round(x))

# 	return x + prob, y_hat

def noise_(x):
	a, b, c, d = x.get_shape().as_list()
	prob = np.random.rand(a, b, c, d) - 0.5
	return prob

def init_probclass():
	config_path = './data/config/base'
	pc_config, pc_config_real_path = config_parser.parse(config_path)
	pc_cls = probclass.get_network_cls(pc_config)
	pc = pc_cls(pc_config, pc_config.num_centers)

	return  pc

def get_batch(batch_size):
	file_name1 = './data/150*150imagenet1.tfrecords'
	file_name2 = './data/professional_train.tfrecords'
	file_name3 = './data/kodak.tfrecords'
	file_name4 = './data/imagenet256.tfrecords'
	img_batch1, label_batch1 = read_and_decode(file_name2, batch_size)
	img_batch1 = tf.divide(tf.cast(img_batch1,tf.float32), 255)

	return img_batch1

def train():
	with tf.Session() as sess:		
		image = get_batch(batch_size)
		net = Net(image)
		feature = net.encoder()
		with tf.variable_scope("data"):
			# qbar, symbol = round(feature)
			noise1 = noise_(feature)
			qbar = feature + noise1
			# qbar = feature 
		r_image = net.decoder(qbar)
		#entropy estimate
		# enc_entropy = net.hyper_enc(qbar)
		# with tf.variable_scope("hyper"):
		# 	# q_entropy, q_symbol = round(enc_entropy)
		# 	noise2 = noise_(enc_entropy)
		# 	q_entropy = enc_entropy + noise2
		# 	# q_entropy = enc_entropy
		# dec_entropy_mu = net.hyper_dec(q_entropy)
		# dec_entropy_std = net.generate_std(dec_entropy_mu)
		# dec_entropy_mu = tf.reduce_mean(tf.reduce_mean(tf.reduce_mean(dec_entropy_mu, 0), 0), 0)
		# dec_entropy_mu = tf.reduce_mean(tf.reduce_mean(tf.reduce_mean(dec_entropy_std, 0), 0), 0)

		# std = (qbar - dec_entropy_mu)/dec_entropy_std
		# pro = 1/(tf.sqrt(2*pi)) * tf.exp(-(tf.square(std)/2)) * delt_range
		# rate_entropy = tf.reduce_mean(tf.log(pro)/tf.log(2.0))
		# q_std = net.non_para(q_entropy)
		# q_std = tf.reduce_mean(tf.reduce_mean(tf.reduce_mean(q_std, 0), 0), 0)
		# q_pro = 1/(tf.sqrt(2*pi)) * tf.exp(-(tf.square(q_std)/2)) * delt_range
		# q_rate_entropy = tf.reduce_mean(tf.log(q_pro)/tf.log(2.0)) 

		mean, std = net.hyper_net(qbar)
		pro = (qbar - mean) / std
		# pro = 1/(tf.sqrt(2*pi)) * tf.exp(-(tf.square(pro)/2)) * delt_range
		pro = tf.sigmoid(pro) + TINY 
		rate_entropy = tf.reduce_mean(tf.log(pro)/tf.log(2.0))

		rate = (1 + tf.log(np.pi)) / 2 + tf.log(std)
		
		img_loss = tf.reduce_sum(tf.squared_difference(image, r_image))
		loss = img_loss / batch_size * lamda + tf.abs(rate_entropy + base_rate) * beta

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
		try:
			i = 0
			for step in range(MAX_step):
				if coord.should_stop():
					break;

				sess.run(optimizer)
				#mask = mask_pro(std)

				if step % 50 == 0 and step > 0:
					ls, lr, ff, qq, bit = sess.run([loss, decayed_learning_rate, feature, qbar, rate_entropy])
					# bit_ls, p = sess.run([rate_entropy, pro])
					print("step:%d lr:%.7f loss=%.5f bit:%.5f" % (step, lr, ls, -(bit)))
					#print("centers:", cen)
					# print("ff:", ff[:,0,0,0])		
					# print("qq:", np.max(qq), np.min(qq))		

				if (step % 4000 == 0 and step > 0) or step == 10000-1:
					#batch_test = sess.run(image)
					checkpoint_path = os.path.join(logs_train_dir,'oppo.ckpt')
					saver.save(sess, checkpoint_path, global_step = step)

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