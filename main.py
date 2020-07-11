#coding: UTF-8
from neuralnet import *
from matplotlib import pyplot as plt

class System1:
	
	def __init__(self, k):
		self.__k = k

	#システム出力の目標値
	def __calculate_system_target(self):
		#2*pi*k/50が正なら1，下回った場合-1を格納する
		x=lambda i : math.sin(2*math.pi*i/50)
		self.__r_data_set=[(x(i)>=0)-(x(i)<0) for i in range(self.__k)]

	#システム出力
	def __calculate_system_output(self, y1, y2, u1, u2):
		a1, a2 = -1, 0.35 
		b0, b1 = 1, 0.75
		return -a1*y1-a2*y2+b0*u1+b1*u2

	#システム出力の誤差
	def __calculate_error(self, r, y):
		return r-y

	#システム入力
	def __calculate_system_input(self, u1, e, e1, e2):
		Kp, Ki, Kd = 0.5, 0.1, 0.05
		return u1+Kp*(e-e1)+Ki*e+Kd*(e-2*e1+e2)

	#入力データuと教師データyの取得
	def make_u_y_data_set(self):
		u1, u2, y, y1, y2, e1, e2= 0, 0, 0, 0, 0, 0, 0
		self.__calculate_system_target()
		self.__u_data_set=np.zeros(self.__k)
		self.__y_data_set=np.zeros(self.__k)

		for i in range(self.__k):
			r=self.__r_data_set[i]
			e=self.__calculate_error(r, y)
			u=self.__calculate_system_input(u1, e, e1, e2)
			self.__y_data_set[i] = y
			self.__u_data_set[i] = u

			#y1, y2, u1, u2の更新
			u2 = u1
			u1 = u
			y2 = y1
			y1 = y
			y = self.__calculate_system_output(y1, y2, u1, u2)
			e2 = e1
			e1 = e

	#順方向の同定での入力ベクトルを作成する
	def make_neuralnet_forward_input_data(self):
		self.make_u_y_data_set()
		self.__modified_y_data_set=np.repeat(self.__y_data_set,2)[:-2].reshape((-1, 2))
		self.__modified_u_data_set=np.repeat(self.__u_data_set,2)[1:-1].reshape((-1, 2))
		self.__input_data=np.c_[self.__modified_y_data_set, self.__modified_u_data_set]

	#逆方向の同定での入力ベクトルを作成する
	def make_neuralnet_backward_input_data(self):
		self.make_u_y_data_set()
		y_data_set1=self.__y_data_set
		y_data_set2=self.__y_data_set[:self.__k-1]
		y_data_set2=np.insert(y_data_set2, 0, 0)
		y_data_set3=self.__y_data_set[:self.__k-2]
		y_data_set3=np.insert(y_data_set3, 0, [0, 0])

		u_data_set=self.__u_data_set[:self.__k-2]
		u_data_set=np.insert(u_data_set, 0, [0, 0])

		self.__input_backward_data=np.c_[y_data_set1,y_data_set2, y_data_set3, u_data_set]


	@property
	def y_data_set(self):
		return self.__y_data_set

	@property
	def u_data_set(self):
		return self.__u_data_set

	@property
	def r_data_set(self):
		return self.__r_data_set
	
	@property
	def k(self):
		return self.__k
	
	@property
	def input_data(self):
		return self.__input_data

	@property
	def input_backward_data(self):
		return self.__input_backward_data


if __name__ =="__main__":
	system1=System1(1000)
	system1.make_neuralnet_backward_input_data()

	neuralnet=NeuralNet(4, 20, system1.input_backward_data, system1.u_data_set)
	neuralnet.train(eta=0.2)

	neuralnet.make_graph()
	



