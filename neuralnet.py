import numpy as np
import math 
import random as 

class NeuralNet:
    #コンストラクタ
	def __init__(self, n_input_layer, n_hidden_layer, n_output_layer):
		self.hidden_weight=np.random.random_sample((n_hidden_layer,n_input_layer+1))
		self.output_weight=np.random.random_sample((n_output_layer,n_hidden_layer+1))
        self.hidden_momentum=np.zeros((n_hidden_layer,n_input_layer+1))
        self.output_momentum=np.zeros((n_output_layer,n_hidden_layer+1))




    def __sigmoid(self, array):
        return np.vectorize(lamda x: 1.0/(1.0+math.exp(-x)))(array)



    def __calculate_error(self, x, target):
        return x-target

    def __update_in_output_weight(self, v, target, eta):
        




  





	
