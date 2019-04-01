import os
import math
import numpy as np
import tensorflow as tf

from codec_base import arithmeticcoding
from PIL import Image

shape = 60.0
class Codec(object):
	"""docstring for Codec"""
	def __init__(self, input_path, out_path):
		super(Codec, self).__init__()
		self.input  = input_path
		self.output = out_path
		# self.sess   = sess

	def pre_pic(self, image):
		if not os.path.exists(self.input):
			return "No picture in this folder!"
		if not os.path.exists(self.output):
			os.mkdir(self.output)
		fileobj = open(self.output, mode='wb')

		# img = Image.open(self.input)
		# w, h = img.size
		h, w, c = image.get_shape().as_list()
		arr = np.array([w, h], dtype=np.uint16)
		arr.tofile(fileobj)
		fileobj.close()
		new_w = int(math.ceil(w / shape) * shape)
		new_h = int(math.ceil(h / shape) * shape)                
		pad_w = new_w - w
		pad_h = new_h - h

		#input_x = np.asarray(img)
		input_x = tf.pad(image, ((pad_h,0), (pad_w,0), (0,0)), mode='reflect')
		#input_x = input_x.reshape(1, new_h, new_w, 3)
		#input_x = input_x.transpose([0, 3, 1, 2])

		return input_x

	def encoder(self, y_hat, mu_y, sigma_y):
		# image = self.pre_pic()
		fileobj = open(self.output, 'wb')
		# image, w, h = self.pre_pic()
		# arr = np.array([w, h], dtype=np.uint8)
		# # arr.tofile(fileobj)
		# # arr = np.array(h, dtype=np.uint8)
		# arr.tofile(fileobj)
		# shape_z = z_hat.shape
		shape_y = y_hat.shape
		# print(shape_z)
		# print(shape_y)
		# arr = np.array(shape_z, dtype=np.uint16)
		# arr.tofile(fileobj)
		arr = np.array(shape_y, dtype=np.uint16)
		arr.tofile(fileobj)
		fileobj.close()

		bitout = arithmeticcoding.BitOutputStream(open(self.output, "ab+"))
		enc = arithmeticcoding.ArithmeticEncoder(bitout)

		############### encode zhat ####################################
		# for ch_idx in range(z_hat.shape[1]):
		#     #printProgressBar(ch_idx + 1, z_hat.shape[1], prefix='Encoding z_hat:', suffix='Complete', length=50)
		#     mu_val = 0
		#     # sigma_val = sigma_z[0, ch_idx, 0, 0]

		#     # freq = arithmeticcoding.ModelFrequencyTable(mu_val, sigma_val)

		#     for h_idx in range(z_hat.shape[2]):
		#         for w_idx in range(z_hat.shape[3]):
		#             #print(z_hat[0, ch_idx, h_idx, w_idx])
		#             symbol = np.int(z_hat[0, ch_idx, h_idx, w_idx] + 255)
		#             sigma_val = sigma_z[0, ch_idx, h_idx, w_idx]
		#             #print('sigma_z', sigma_val)
		#             freq = arithmeticcoding.ModelFrequencyTable(mu_val, sigma_val)
		#             #print(symbol)
		#             if symbol < 0 or symbol > 511:
		#                 print("symbol range error: " + str(symbol))
		#             enc.write(freq, symbol)

        ############### encode yhat ####################################
        # padded_y_hat = np.pad(y_hat, ((0, 0), (0, 0), (3, 0), (2, 1)), 'constant',
        #                       constant_values=((0, 0), (0, 0), (0, 0), (0, 0)))

		for ch_idx in range(y_hat.shape[3]):
		    #printProgressBar(h_idx + 1, y_hat.shape[2], prefix='Encoding y_hat:', suffix='Complete', length=50)
		    for w_idx in range(y_hat.shape[2]):
		        for h_idx in range(y_hat.shape[1]):
		            mu_val = mu_y[ch_idx] + 0
		            sigma_val = sigma_y[ch_idx]

		            freq = arithmeticcoding.ModelFrequencyTable(mu_val, sigma_val)

		            symbol = np.int(y_hat[0, h_idx, w_idx, ch_idx] + 31)
		            if symbol < 0 or symbol > 63:
		                print("symbol range error: " + str(symbol))

		            enc.write(freq, symbol)

		enc.write(freq, 64)
		enc.finish()
		bitout.close()

	def decoder(self, mu_y, sigma_y):
	    fileobj = open(self.output, mode='rb')
	    # buf = fileobj.read(4)
	    # arr = np.frombuffer(buf, dtype=np.uint16)
	    # w = int(arr[0])
	    # h = int(arr[1])

	    # padded_w = int(math.ceil(w / shape) * shape)
	    # padded_h = int(math.ceil(h / shape) * shape)
	    shape_y = [0, 0, 0, 0]
	    shape_z = [0, 0, 0, 0]
	    #construct the z_hat and y_hat
	    # buf = fileobj.read(8)
	    # arr = np.frombuffer(buf, dtype=np.uint16)
	    # for i in range(4):
	    #     shape_z[i] = int(arr[i])
	    buf = fileobj.read(8)
	    arr = np.frombuffer(buf, dtype=np.uint16)
	    for i in range(4):
	    	shape_y[i] = int(arr[i])
	    #shape_y = int(arr)
	    # print(shape_z)
	    # print(shape_y)
	    z_hat = np.zeros(shape_z)
	    y_hat = np.zeros(shape_y)

	    ############### decode zhat ####################################
	    bitin = arithmeticcoding.BitInputStream(fileobj)
	    dec = arithmeticcoding.ArithmeticDecoder(bitin)
	    #z_hat is to be constucted
	    #z_hat[:, :, :, :] = 0.0

	    # for ch_idx in range(z_hat.shape[1]):
	    #     #printProgressBar(ch_idx + 1, z_hat.shape[1], prefix='Decoding z_hat:', suffix='Complete', length=50)
	    #     mu_val = 0
	    #     # sigma_val = sigma_z[0, ch_idx, 0, 0]

	    #     # freq = arithmeticcoding.ModelFrequencyTable(mu_val, sigma_val)

	    #     for h_idx in range(z_hat.shape[2]):
	    #         for w_idx in range(z_hat.shape[3]):
	    #             sigma_val = sigma_z[0, ch_idx, h_idx, w_idx]
	    #             freq = arithmeticcoding.ModelFrequencyTable(mu_val, sigma_val)
	    #             symbol = dec.read(freq)
	    #             if symbol == 512:  # EOF symbol
	    #                 print("EOF symbol")
	    #                 break
	    #             z_hat[:, ch_idx, h_idx, w_idx] = symbol - 255

	    ############### decode yhat ####################################
	    # padded_y_hat = np.pad(y_hat, ((0, 0), (0, 0), (3, 0), (2, 1)), 'constant',
	    #                       constant_values=((0, 0), (0, 0), (0, 0), (0, 0)))
	    # padded_y_hat[:, :, :, :] = 0.0
	    for ch_idx in range(y_hat.shape[3]):
	        #printProgressBar(h_idx + 1, y_hat.shape[2], prefix='Decoding y_hat:', suffix='Complete', length=50)
	        for w_idx in range(y_hat.shape[2]):
	            # c_prime_i = self.extractor_prime(padded_c_prime, h_idx, w_idx)
	            # c_doubleprime_i = self.extractor_doubleprime(padded_y_hat, h_idx, w_idx)
	            # concatenated_c_i = np.concatenate([c_doubleprime_i, c_prime_i], axis=1)

	            # pred_mean, pred_sigma = self.sess.run(
	            #     [self.pred_mean, self.pred_sigma],
	            #     feed_dict={self.concatenated_c_i: concatenated_c_i})

	            for h_idx in range(y_hat.shape[1]):
	                mu_val = mu_y[ch_idx] + 0
	                sigma_val = sigma_y[ch_idx]

	                freq = arithmeticcoding.ModelFrequencyTable(mu_val, sigma_val)

	                symbol = dec.read(freq)
	                if symbol == 64:  # EOF symbol
	                    print("EOF symbol")
	                    break
	                y_hat[:, h_idx, w_idx, ch_idx] = symbol - 31

	    bitin.close()
	    #y_hat = padded_y_hat[:, :, 3:, 2:-1]        
	    #recon = self.sess.run(y_hat[0, -h:, -w:, :])
	    return y_hat
