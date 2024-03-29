#!/usr/bin/env python
# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Trains MNIST using the model described here:
https://www.tensorflow.org/versions/r0.11/tutorials/mnist/pros/index.html#deep-mnist-for-experts
"""

import tensorflow as tf


class MnistConvModel(object):

	def __init__(self):
		self.init_model()

	def init_model(self):

		self.x = tf.placeholder(tf.float32, shape=[None, 784])
		self.y_ = tf.placeholder(tf.float32, shape=[None, 10])

		# first convolutional layer

		W_conv1 = self.weight_variable([5, 5, 1, 32])
		b_conv1 = self.bias_variable([32])

		x_image = tf.reshape(self.x, [-1, 28, 28, 1])

		h_conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1) + b_conv1)
		h_pool1 = self.max_pool_2x2(h_conv1)

		# second convolutional layer

		W_conv2 = self.weight_variable([5, 5, 32, 64])
		b_conv2 = self.bias_variable([64])

		h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)
		h_pool2 = self.max_pool_2x2(h_conv2)

		# densely connected layer

		W_fc1 = self.weight_variable([7 * 7 * 64, 1024])
		b_fc1 = self.bias_variable([1024])

		h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
		h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

		# dropout

		self.keep_prob = tf.placeholder(tf.float32)
		h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

		# readout layer

		W_fc2 = self.weight_variable([1024, 10])
		b_fc2 = self.bias_variable([10])

		y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
		self.cross_entropy = tf.reduce_mean(
			tf.nn.softmax_cross_entropy_with_logits(y_conv, self.y_))
		self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy)
		correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(self.y_, 1))
		self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	def get_entropy_train_step_accuracy(self):
		return self.cross_entropy, self.train_step, self.accuracy

	def get_placeholders(self):
		return self.x, self.y_, self.keep_prob

	def weight_variable(self, shape):
		initial = tf.truncated_normal(shape, stddev=0.1)
		return tf.Variable(initial)

	def bias_variable(self, shape):
		initial = tf.constant(0.1, shape=shape)
		return tf.Variable(initial)

	def conv2d(self, x, W):
		return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

	def max_pool_2x2(self, x):
		return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
							  strides=[1, 2, 2, 1], padding='SAME')
