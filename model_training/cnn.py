from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Dropout
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch import flatten

class CNN(Module):
    
	def __init__(self, num_filter_list: list[int], kernal_size_list: list[int], stride_list: list[int], padding_list: list[int], drop_list: list[bool], pooling_list: list[int],
			  linear_layer_outputs: list[int], classes: list[int], input_size: int=48):
		"""
		customizable class to create a convolutional neural network using PyTorch. Activation function for each layer is ReLU

		Args:
			num_filter_list (list[int]): list of the number of filters for the coressponding layer
			kernal_size_list (list[int]): list of the size of each filter for the coressponding convloutional layer
			stride_list (list[int]): list of the stride length of each filter for the coressponding convolutional layer 
			padding_list (list[int]): list of the amount of padding the coressponding convolutional layer
			pooling_list (list[int]): list of the pooling layers applied to each convolutional layer
			drop_list (list[bool]): list of if we want to drop random nodes on the coressponding convolutional layer
			linear_layer_outputs (list[int]): the output nodes on each linear layer (after flattenning)
			classes (list[int]): number of classes
			intput_size (int): input size of image
		"""

		# assert all elements are of the proper sizes
		assert len(num_filter_list) == len(kernal_size_list)
		assert len(kernal_size_list) == len(stride_list)
		assert len(stride_list) == len(padding_list)
		assert len(padding_list) == len(drop_list)
		assert len(drop_list) == len(pooling_list)

		assert linear_layer_outputs[-1] == classes

		super(CNN, self).__init__()

		self.NUM_CONV_LAYERS = len(num_filter_list)
		self.NUM_LINEAR_LAYERS = len(linear_layer_outputs)

		self.drop_list = drop_list
		self.pooling_list = pooling_list

		self.conv_layers= []

		prev_in = input_size
		for i in range(self.NUM_CONV_LAYERS):
			curr_layer = Conv2d(in_channels=prev_in, out_channels=num_filter_list[i], kernel_size=kernal_size_list[i],
						stride=stride_list[i], padding=padding_list[i])
			prev_in = ((prev_in + 2 * padding_list[i] - kernal_size_list[i]) / (stride_list[i])) + 1
			prev_in = prev_in / (max(1, ))
			
			self.conv_layers.append(curr_layer)
		
		prev_in = (prev_in ** 2) * num_filter_list[-1]

		self.linear_layers = []
		for i in range(self.NUM_LINEAR_LAYERS):
			curr_layer = Linear(in_features=prev_in, out_features=linear_layer_outputs[i])
			self.linear_layers.append(curr_layer)
			prev_in = linear_layer_outputs[i]




		self.classes = classes

	def forward(self, x):
		
		for i in range(self.NUM_CONV_LAYERS):
			conv_layer, dropout_true, pooling_value= self.conv_layers, self.drop_list[i], self.pooling_list[i]

			x = conv_layer(x)
			x = ReLU()(x)
			if dropout_true:
				x = Dropout(p=0.5)(x)
			x = MaxPool2d(pooling_value)
		
		# flatten after all convolution layers 
		x = flatten(x)

		for i in range(self.NUM_LINEAR_LAYERS - 1):
			linear_layer = self.linear_layers[i]
			x = linear_layer(x)
			x = ReLU()(x)

		# perform log softmax on the last layer
		x = self.linear_layers[-1](x)
		output = LogSoftmax()(x)
		
		return output


				








		


