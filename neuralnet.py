#coding: UTF-8
import numpy as np
import math 
import random
from matplotlib import pyplot as plt

	
class NeuralNet:
    #コンストラクタ

	def __init__(self, number_of_input, number_of_hidden):
		self.hidden_weight=np.random.random_sample((number_of_hidden,number_of_input))
		self.output_weight=np.random.random_sample(number_of_hidden)


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
	def __activation_on_hidden(self,X):
		vector=np.dot(self.hidden_weight,X)
		return np.vectorize(self.__sigmoid)(vector)

	#出力層での活性化関数
	def __activation_on_output(self,V):
		return np.dot(V, self.output_weight)

	#損失関数
	def __loss(self, training_data, y):
		return training_data-y


	#隠れ層の重みの更新
	def __update_output_weight(self, training_data, V, eta, error):
		f_dot=1
	
		dj_dw=error*f_dot*V
		self.output_weight=self.output_weight+eta*dj_dw

		#print("")
		#print("Updating Weight between Output Layer and Hidden Layer")
		# print("---Output Weight---")
		# print(self.output_weight)

	#出力層の重みの更新
	def __update_hidden_weight(self, training_data, X, V, eta, error):
		#print("")
		#print("Updating Weight between Input and Hidden Layer")

		f_dot_hidden=np.dot(self.hidden_weight,X)
		f_dot_hidden=np.vectorize(self.__sigmoid_derivative)(f_dot_hidden)
		f_dot_output=1

		for i in range(self.output_weight.size):
			w=self.output_weight[i]
			f_dot_hidden_vector=f_dot_hidden[i]
			
			dj_dw=error*f_dot_hidden_vector*w*f_dot_output*X
			updated_weight=eta*dj_dw
		
			self.hidden_weight=np.delete(self.hidden_weight,i,0)
			self.hidden_weight=np.insert(self.hidden_weight,i,updated_weight,axis=0)
			

	def __back_propagation(self,training_data, X, V, eta, error):
		self.__update_output_weight(training_data, V, eta,error)
		self.__update_hidden_weight(training_data, X, V, eta,error)

	def forward(self, X):
		self.__V=self.__activation_on_hidden(X)
		return self.__activation_on_output(self.__V)


	def train(self, training_data_set, X_data_set, eta):
		i=0
		data_size=training_data_set.size
		self.y_data_set=np.zeros(data_size)
		for (X, training_data, i) in zip(X_data_set, training_data_set, range(data_size)):
			print("")
			print("")
			print("**********")
			print(str(i+1)+"回目")

			y=self.forward(X)
			self.y_data_set[i]=y
			print("Output y_nn")
			print(y)
			print("Output y")
			print(training_data)
			error=self.__loss(training_data,y)
			print("")
			print("===ERROR===")
			print(error)
			self.__back_propagation(training_data, X, self.__V, eta, error)

	def make_graph(self, training_data_set, y_data_set):
		x=range(0, 100)
		y=training_data_set[:100]
		y_nn=y_data_set[:100]
		plt.plot(x,y, x, y_nn)
		plt.show()




if __name__=="__main__":
	neuralnet=NeuralNet(4,5)
			
	print("START")
	
  	X_data_set=np.random.rand(200).reshape(50,4)
  	training_data_set=np.random.rand(50)
  	print(X_data_set)
  	print(training_data_set)
  	neuralnet.train(training_data_set, X_data_set, eta=0.2)





	
