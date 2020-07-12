#coding: UTF-8
import numpy as np
import math 
import random
from matplotlib import pyplot as plt

	
class NeuralNet:

	#コンストラクタ
	#input_data_set...学習データ
	#training_data_set...教師データ
	#number_of_hidden...隠れ層の要素数
	def __init__(self, number_of_input, number_of_hidden, input_data_set, training_data_set):
		self.__hidden_weights=np.random.random_sample((number_of_hidden,number_of_input))
		self.__output_weights=np.random.random_sample(number_of_hidden)
		self.__input_data_set=input_data_set
		self.__training_data_set=training_data_set


	# def __sigmoid(self, array):
	# 	return np.vectorize(lambda x: 1.0/(1.0+math.exp(-x)))(array)

	#シグモイド関数
	def __sigmoid(self, x):
		return 1/(1+np.exp(-x))
	#シグモイド関数の微分
	def __sigmoid_derivative(self, x):
		sigmoid=self.__sigmoid(x)
		return sigmoid*(1.0-sigmoid)

	#隠れ層での活性化関数
	def __activation_on_hidden(self, input_data):
		vector=np.dot(self.__hidden_weights, input_data)
		return np.vectorize(self.__sigmoid)(vector)

	#出力層での活性化関数
	def __activation_on_output(self, v):
		return np.dot(v, self.__output_weights)

	#損失関数
	def __loss(self, training_data, y_nn):
		return training_data-y_nn


	#隠れ層の重みの更新
	def __update_output_weight(self, training_data, v, eta, error):
		f_dot_output=1
		dj_dw=error*f_dot_output*v
		self.__output_weights=self.__output_weights+eta*dj_dw

		#print("")
		#print("Updating Weight between Output Layer and Hidden Layer")
		# print("---Output Weight---")
		# print(self.output_weight)

	#出力層の重みの更新
	def __update_hidden_weight(self, training_data, input_data, v, eta, error):
		#print("")
		#print("Updating Weight between Input and Hidden Layer")

		f_dot_hidden=np.dot(self.__hidden_weights, input_data)
		f_dot_hidden=np.vectorize(self.__sigmoid_derivative)(f_dot_hidden)
		f_dot_output=1

		for i in range(self.__output_weights.size):
			w=self.__output_weights[i]
			f_dot_hidden_vector=f_dot_hidden[i]
			
			dj_dw=error*f_dot_hidden_vector*w*f_dot_output*input_data
			updated_weight=eta*dj_dw
		
			#各重みの書き換え
			self.__hidden_weights=np.delete(self.__hidden_weights,i,0)
			self.__hidden_weights=np.insert(self.__hidden_weights,i,updated_weight,axis=0)
		
			

	def __back_propagation(self,training_data, X, V, eta, error):
		self.__update_output_weight(training_data, V, eta, error)
		self.__update_hidden_weight(training_data, X, V, eta, error)

	def __forward(self, input_data):
		self.__V=self.__activation_on_hidden(input_data)
		return self.__activation_on_output(self.__V)

	#training_styleが0なら順方向の学習，training_styleがそれ以外なら逆方向の学習
	def train(self, eta=0.2):
		i=0
		data_size=self.__training_data_set.size
		self.__y_nn_data_set=np.zeros(data_size)
		for (input_data, training_data, i) in zip(self.__input_data_set, self.__training_data_set, range(data_size)):
			print("")
			print("")
			print("**********")
			print(str(i+1)+"回目")

			y_nn=self.__forward(input_data)
			self.__y_nn_data_set[i]=y_nn
			print("Output y_nn")
			print(y_nn)
			print("Output y")
			print(training_data)

			error=self.__loss(training_data,y_nn)
			
			print("")
			print("===ERROR===")
			print(error)
			self.__back_propagation(training_data, input_data, self.__V, eta, error)


	def make_graph(self):
		x=range(0, 99)
		y=self.__training_data_set[:100]
		y_nn=self.__y_nn_data_set[:100]
		# plt.plot(x, y, x, y_nn, label="y")
		plt.title("Forward / Hidden Layer:20")
		plt.plot(x,y, label="y")
		plt.xlabel("k")
		plt.ylabel("output")
		plt.plot(x,y_nn, label="y_nn")
		plt.ylim(-2, 2)
		plt.legend()

		plt.show()


	@property
	def y_nn_data_set(self):
		return self._y_nn_data_set


if __name__=="__main__":
	




	
