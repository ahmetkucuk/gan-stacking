

class EvaluationDataset(object):

	def __init__(self, data, labels):
		self.data = data
		self.labels = labels
		self.batch_index = 0

	def next_batch(self, batch_size):
		if self.batch_index*batch_size + batch_size > len(self.data):
			self.batch_index = 0
		batched_data, batched_labels = self.data[self.batch_index*batch_size: self.batch_index*batch_size + batch_size], self.labels[self.batch_index*batch_size: self.batch_index*batch_size + batch_size]
		self.batch_index += 1
		return batched_data, batched_labels

	def size(self):
		return len(self.data)