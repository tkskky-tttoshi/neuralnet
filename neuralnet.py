#coding: UTF-8
import numpy as np
import math 
import random
	
class NeuralNet:
    #コンストラクタ

	def __init__(self, number_of_input, number_of_hidden):
		self.hidden_weight=np.random.random_sample((number_of_input,number_of_hidden))
		self.output_weight=np.random.random_sample(number_of_hidden)

	# def __sigmoid(self, array):
	# 	return np.vectorize(lambda x: 1.0/(1.0+math.exp(-x)))(array)

	#シグモイド関数
	def __sigmoid(self,x):
		return 1/(1+np.exp(-x))
	#シグモイド関数の微分
	def __sigmoid_derivative(self, x):
		sigmoid=self.__sigmoid(x)
		return sigmoid*(1.0-sigmoid)

	#隠れ層での活性化関数
	def activation_on_hidden(self,X):
		vec=np.vectorize(np.dot)(X, self.hidden_weight)
		return np.vectorize(self.__sigmoid)(vec)

	#出力層での活性化関数
	def activation_on_output(self,V):
		return np.dot(V, self.output_weight)

	#損失関数
	def __loss(self, training_data, V):
		return self.__activation_on_output(V)-training_data

	def __update_w2_weight(self, training_data, V, eta):
		self.error=self.__loss(learning_data, V)
		f_dot=1
		dj_dw=error*f_dot*V
		self.output_weight=self.output_weight+eta*dj_dw

	def update_w1_weight(self, training_data, X, V, eta):
		# error=self.error
		error=0.1
		f_dot1=np.vectorize(self.__sigmoid_derivative)(X,self.hidden_weight)
		print(f_dot1.size())
		f_dot2=1
		dj_dw=error*f_dot1*self.output_weight*f_dot2*X

		return W1+eta*dj_dw

	#def train(self,x,training_data):



if __name__=="__main__":
	neuralnet=NeuralNet(2,4)
	print("hidden")
	print(neuralnet.hidden_weight)
	print("output")
	print(neuralnet.output_weight)
	V=np.arange(8).reshape(2,4)
	print(V)
	print("hidden_updated")
	print(neuralnet.activation_on_hidden(V))
	print("output_updated")
	X=np.arange(4)
	print(neuralnet.activation_on_output(X))
	training_data=1
	print(neuralnet.update_w1_weight(training_data,X,V,eta=0.1))


  





	
