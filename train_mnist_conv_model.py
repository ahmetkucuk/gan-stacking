
import tensorflow as tf
import time
import os

# MODEL_DIR = os.path.join("/tmp/tfmodels/mnist_cnn", str(int(time.time())))


class MnistConvModelTrain(object):

	def __init__(self, model, dataset, test_dataset, name="Not Specified"):

		self.x, self.y_, self.keep_prob = model.get_placeholders()
		self.cross_entropy, self.train_step, self.accuracy = model.get_entropy_train_step_accuracy()
		self.dataset = dataset
		self.test_dataset = test_dataset
		self.name = name

	def log_print(self, val_str):
		print("Log in " + self.name + ":\t" + val_str)

	def train(self, n_of_epochs, batch_size=50):
		sess = tf.InteractiveSession()

		# Define info to be used by the SummaryWriter. This will let TensorBoard
		# plot loss values during the training process.
		# loss_summary = tf.scalar_summary("loss", self.cross_entropy)
		# train_summary_op = tf.merge_summary([loss_summary])

		sess.run(tf.initialize_all_variables())

		# Create a saver for writing training checkpoints.
		# saver = tf.train.Saver()

		# Create a summary writer.
		# train_summary_writer = tf.train.SummaryWriter(MODEL_DIR)

		# training
		n_of_iter = int(((self.dataset.size()*n_of_epochs)/batch_size))
		self.log_print("Number of iteration %d" % n_of_iter)
		for step in xrange(n_of_iter):
			batch = self.dataset.next_batch(batch_size)

			_, loss = sess.run(
				[self.train_step, self.cross_entropy],
				feed_dict={self.x: batch[0], self.y_: batch[1], self.keep_prob: 0.5})
			if step % 100 == 0:
				self.log_print("adding summary for step: %s" % step)
				# train_summary_writer.add_summary(tsummary, step)
				train_accuracy = self.accuracy.eval(feed_dict={
					self.x: batch[0], self.y_: batch[1], self.keep_prob: 1.0})
				self.log_print("step %d, training accuracy %g" % (step, train_accuracy))
				self.log_print("loss: %s" % loss)
			# if step % 5000 == 0:
			# 	# Write a checkpoint.
			# 	print("Writing checkpoint file.")
			# 	checkpoint_file = os.path.join(MODEL_DIR, 'checkpoint')
			# 	saver.save(sess, checkpoint_file, global_step=step)

		self.log_print("test accuracy %g" % self.accuracy.eval(
			feed_dict={self.x: self.test_dataset.images,
					   self.y_: self.test_dataset.labels, self.keep_prob: 1.0}))