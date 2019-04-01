#code for oppo contest
import tensorflow as tf
import tensorflow_compression as tfc
import numpy as np
import os
import argparse

from data import quantizer

num_centers = 5
minval = 0
maxval = 1
beta = 1/3
class Net(object):
	"""docstring for Net"""
	def __init__(self, img):
		#super(Net, self).__init__()
		self.img = img

	def encoder(self):
		with tf.variable_scope("encoder"):
			with tf.variable_scope("layer_0"):
				layer = tfc.SignalConv2D(
					128, (5, 5), corr=True, strides_down=2, padding="same_zeros",
					use_bias=True, activation=tfc.GDN(), name='ly_0')
				tensor = layer(self.img)

			with tf.variable_scope("layer_1"):
				layer = tfc.SignalConv2D(
					256, (5, 5), corr=True, strides_down=2, padding="same_zeros",
					use_bias=True, activation=tfc.GDN(), name='ly_1')
				tensor = layer(tensor)

			with tf.variable_scope("layer_2"):
				layer = tfc.SignalConv2D(
					256, (5, 5), corr=True, strides_down=2, padding="same_zeros",
					use_bias=True, activation=tfc.GDN(), name='ly_2')
				tensor = layer(tensor)

			with tf.variable_scope("layer_3"):
				layer = tfc.SignalConv2D(
					192 * beta, (5, 5), corr=True, strides_down=2, padding="same_zeros",
					use_bias=True, activation=None, name='ly_3')
				tensor = layer(tensor)
				# tensor = tf.tanh(tensor)

			return tensor

	def decoder(self, tensor):
		with tf.variable_scope("decoder"):
			with tf.variable_scope("layer_0"):
				layer = tfc.SignalConv2D(
					256, (5, 5), corr=False, strides_up=2, padding="same_zeros",
					use_bias=True, activation=tfc.GDN(inverse=True), name='ly_0')
				tensor = layer(tensor)

			with tf.variable_scope("layer_1"):
				layer = tfc.SignalConv2D(
					256, (5, 5), corr=False, strides_up=2, padding="same_zeros",
					use_bias=True, activation=tfc.GDN(inverse=True), name='ly_1')
				tensor = layer(tensor)

			with tf.variable_scope("layer_2"):
				layer = tfc.SignalConv2D(
					128, (5, 5), corr=False, strides_up=2, padding="same_zeros",
					use_bias=True, activation=tfc.GDN(inverse=True), name='ly_2')
				tensor = layer(tensor)

			with tf.variable_scope("layer_3"):
				layer = tfc.SignalConv2D(
					3, (5, 5), corr=False, strides_up=2, padding="same_zeros",
					use_bias=True, activation=None, name='ly_3')
				tensor = tf.sigmoid(layer(tensor))

			return tensor

	def quantization(self, sampled_z):
		regularization_factor_centers = 1
		centers = quantizer.create_centers_variable(num_centers, minval, maxval)
		quantizer.create_centers_regularization_term(regularization_factor_centers, centers)
		qsoft, qhard, symbol = quantizer.quantize(sampled_z, centers, sigma=1)
		qbar = tf.add(qsoft, tf.stop_gradient(qhard - qsoft), name='qbar')
		#symbol = tf.add(symbol, 0, name='symbol')

		return qbar, qhard, symbol, centers

	def hyper_enc(self, tensor):
		with tf.variable_scope("hyper_enc"):
			with tf.variable_scope("abs"):
				tensor = tf.abs(tensor, name='abs')

			with tf.variable_scope("layer_0"):
				tensor = tf.layers.conv2d(tensor, filters=128, kernel_size=3,
					strides=1, padding='same', activation=tf.nn.relu, name='ly_0')

			with tf.variable_scope("layer_1"):
				tensor = tf.layers.conv2d(tensor, filters=128, kernel_size=3,
					strides=2, padding='same', activation=tf.nn.relu, name='ly_1')

			with tf.variable_scope("layer_2"):
				tensor = tf.layers.conv2d(tensor, filters=128, kernel_size=3,
					strides=2, padding='same', activation=None, name='ly_2')

			# with tf.variable_scope("dense"):
			# 	#Cin = tensor.get_shape().as_list()[3]
			# 	tensor = tf.redunce_mean(tf.reduce_mean(tf.reduce_mean(tensor, axis=-1), axis=-1), axis=0)
			# 	tensor = tf.expand_dims(tf.expand_dims(tensor, axis=-1), axis=-1)
			# tensor = tf.tanh(tensor)

			return tensor
	
	def hyper_dec(self, tensor):
		with tf.variable_scope("hyper_dec"):
			with tf.variable_scope("layer_0"):
				tensor = tf.layers.conv2d_transpose(tensor, filters=128, kernel_size=3,
					strides=2, padding='same', activation=tf.nn.relu, name='ly_0')

			with tf.variable_scope("layer_1"):
				tensor = tf.layers.conv2d_transpose(tensor, filters=128, kernel_size=3,
					strides=2, padding='same', activation=tf.nn.relu, name='ly_1')

			with tf.variable_scope("layer_2"):
				tensor = tf.layers.conv2d_transpose(tensor, filters=192 * beta, kernel_size=3,
					strides=1, padding='same', activation=tf.sigmoid, name='mu_y')

			return tensor

	def generate_std(self, tensor):
		with tf.variable_scope("generate_std"):
			tensor = tf.layers.conv2d(tensor, filters=192 * beta, kernel_size=7, strides=1,
				padding='same', activation=tf.exp, name='sigma_y')

		return tensor

	def density_model(self, tensor):
		K = 4
		f_num = 10
		Cin = tensor.get_shape().as_list()[3] 
		filters = [0 for _ in range(K)]
		x = [0 for _ in range(K)]
		b = [0 for _ in range(K)]
		a = [0 for _ in range(K-1)]
		p = [0 for _ in range(K)]
		for i in np.arange(K):
			x[i] = tensor

		with tf.variable_scope("density_model"):
			for i in np.arange(K):
				with tf.variable_scope("density_function_{}".format(i)):
					filters[i] = tf.Variable(tf.random_normal(shape=[3, 3, Cin, f_num]), name="filters_{}".format(i))
					filters[i] = tf.nn.softplus(filters[i])
					b[i] = tf.Variable(tf.random_normal(shape=[f_num]), name="b_{}".format(i))
					x[i] = tf.nn.conv2d(x[i], filters[i], strides=[1, 2, 2, 1],padding="SAME")
					x[i] = x[i] + b[i]
					if(i < K-1):
						a[i] = tf.Variable(tf.random_normal(shape=[f_num]), name="a_{}".format(i))
						x[i] = x[i] + tf.multiply(tf.tanh(a[i]), tf.tanh(x[i]))

					else:
						x[i] = tf.sigmoid(x[i])

				with tf.variable_scope("density_{}".format(i)):
					p[i] = tf.gradients(x[i], tensor)

		return p

	def non_para(self, tensor):
		Cin = tensor.get_shape().as_list()[3]
		mean = tf.Variable(tf.random_normal(shape=[Cin]), name='non-para_mean')
		std = tf.Variable(tf.random_normal(shape=[Cin]),  name='non-para_std')
		std = tf.exp(std)

		temp = tf.divide((tensor-mean), std, name='sigma_z')
		return temp


	def conv2D(self, tensor, flag, f_out, name, fn=None, s_a=2, f_a=3):
		assert name is not None, 'Need name'
		with tf.variable_scope(name):
			#create the mask for conv2d
			k = f_a
			shape = [k, k]
			mask = np.ones(shape, dtype=np.float32)
			mask[k // 2, (k // 2 + 1):] = 0
			mask[k // 2 + 1:, :]  = 0
			mask = np.expand_dims(np.expand_dims(mask, -1), -1)

			#conv2d--filter
			#f_a = 3
			f_in = tensor.get_shape().as_list()[-1]
			#f_out = 32
			f_shape = [f_a, f_a, f_in, f_out]
			filters = tf.get_variable('weights', shape=f_shape, dtype=tf.float32,
				                      initializer=tf.contrib.layers.xavier_initializer())
			filters = filters * mask
			bias = tf.get_variable('bias', shape=(f_out,), dtype=tf.float32,
				                      initializer=tf.zeros_initializer())

			#conv2d--stride
			#s_a = 2
			strides_a = [1, s_a, s_a, 1]
			#strides_b = [1, s_a/2, s_a/2, 1]

			#conv2d
			if flag == 'down':
				tensor = tf.nn.conv2d(tensor, filters, strides_a, padding='SAME')

			if flag == 'up':
				tensor == tf.nn.conv2d_transpose(tensor, filters_a, strides, padding='SAME')
			tensor = tf.nn.bias_add(tensor, bias, name = 'bias2d')

			if fn != None:
				tensor = fn(tensor)
			else:
				tensor = tf.nn.relu(tensor)

		return tensor

	def hyper_net(self, tensor):
		tensor = self.conv2D(tensor, flag='down', f_out=192 * beta, name='hyper1', f_a=5)
		tensor = self.conv2D(tensor, flag='down', f_out=192 * beta, name='hyper2', f_a=5)
		mean = self.conv2D(tensor, flag='down', f_out=192 * beta, name='hyper3', f_a=5)
		std = self.conv2D(tensor, flag='down', f_out=192 * beta, name='hyper4', fn=tf.exp, s_a=1, f_a=5)
		mean = tf.reduce_mean(tf.reduce_mean(tf.reduce_mean(mean, 0), 0), 0)
		std = tf.reduce_mean(tf.reduce_mean(tf.reduce_mean(std, 0), 0), 0)
		# mean = tf.reduce_mean(mean)
		# std = tf.reduce_mean(std)

		return mean, std

def mask_pro(tensor):
	b, h, w, c = tensor.get_shape().as_list()
	mask = np.random.randint(0, 2, [h, w])
	# while(np.mean(mask) < 15/256.0 or np.mean(mask) > 17/256.0):
	# 	mask = np.random.randint(0, 2, [h, w])

	mask = np.expand_dims(mask, -1)
	#tensor = tensor * mask

	return mask

# if __name__ == '__main__':
# 	parser = argparse.ArgumentParser(
#       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# 	parser.add_argument(
# 		"--info", default="info",
# 		help="infomation about this code")
# 	args = parser.parse_args()

# 	print(args.info)
