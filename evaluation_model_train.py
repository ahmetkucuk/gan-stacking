from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import os

from mnist_gan import MnistGanModel
from mnist_conv import MnistConvModel
from train_mnist_gan_model import MnistGanModelTrain
from train_mnist_conv_model import MnistConvModelTrain
from dataset import EvaluationDataset

small_n_mnist_samples = 1000

layer0_model_epochs = 2000
layer0_model_sample_size = 16*1000


layer1_model1_epochs = 2000
layer1_model2_epochs = 2000

layer1_model1_sample_size = 16*2500
layer1_model2_sample_size = 16*2500


layer2_model1_epochs = 4000
layer2_model1_sample_size = 16*2500

mnist_trained_with_small_n_samples_epochs = 4000
mnist_trained_with_gen_samples_epochs = 4000

data_dir = '../../MNIST_data'

#GET MNIST DATA
mnist = input_data.read_data_sets(data_dir, one_hot=True)
data, labels = mnist.train.images, mnist.train.labels

#GET SUB DATASETS FROM MNIST
small_n_mnist_for_gen_samples = EvaluationDataset(data=data[:small_n_mnist_samples], labels=labels[:small_n_mnist_samples])
small_n_mnist_train_samples = EvaluationDataset(data=data[:small_n_mnist_samples], labels=labels[:small_n_mnist_samples])

#LAYER 0
layer0_model = MnistGanModelTrain(MnistGanModel(), small_n_mnist_for_gen_samples, name="Layer0_Model1")
layer0_samples, layer0_labels = layer0_model.train(layer0_model_epochs, layer0_model_sample_size, should_plot=True)

layer0_data_output = EvaluationDataset(data=layer0_samples, labels=layer0_labels)

print("layer0 finished")

layer1_model1 = MnistGanModelTrain(MnistGanModel(), layer0_data_output, name="Layer1_Model1")
layer1_model1_samples, layer1_model1_labels = layer1_model1.train(layer1_model1_epochs, layer1_model1_sample_size, should_plot=False)

print("layer 1 model 1 finished")

layer1_model2 = MnistGanModelTrain(MnistGanModel(), layer0_data_output, name="Layer1_Model2")
layer1_model2_samples, layer1_model2_labels = layer1_model2.train(layer1_model2_epochs, layer1_model2_sample_size, should_plot=False)

print("layer 1 model 2 finished")

layer1_data_output = EvaluationDataset(data=np.concatenate([layer1_model1_samples, layer1_model2_samples]), labels=np.concatenate([layer1_model1_labels, layer1_model2_labels]))

layer2_model1 = MnistGanModelTrain(MnistGanModel(), layer1_data_output, name="Layer2_Model1")
gen_samples, gen_labels = layer2_model1.train(layer2_model1_epochs, layer2_model1_sample_size, should_plot=True)

print("layer 2 finished")

gen_mnist_train = EvaluationDataset(data=gen_samples, labels=gen_labels)
mnist_trained_with_small_n_samples = MnistConvModelTrain(MnistConvModel(), small_n_mnist_train_samples, mnist.test, name="SMALL_NUMBER_OF_MNIST_SAMPLES").train(mnist_trained_with_gen_samples_epochs)
mnist_trained_with_gen_samples = MnistConvModelTrain(MnistConvModel(), gen_mnist_train, mnist.test, name="MNIST_WITH_GENERATED_SAMPLES").train(mnist_trained_with_gen_samples_epochs)

os.rmdir(data_dir)
