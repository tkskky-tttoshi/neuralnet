#coding: UTF-8
from neuralnet import *
from matplotlib import pyplot as plt

class System:

	def __init__(self, k):
		self.__k=k

	#システム出力の目標値
	def calculate_system_target(self):
		#2*pi*k/50が正なら1，下回った場合-1を格納する
		x=lambda i : math.sin(2*math.pi*i/50)
		return [(x(i)>=0)-(x(i)<0) for i in range(self.__k)]

#システム出力の誤差
	def __calculate_error(self, r, y):
		return r-y

	#システム入力
	def __calculate_system_input(self, u1, e, e1, e2):
		Kp, Ki, Kd = 0.5, 0.1, 0.05
		return u1+Kp*(e-e1)+Ki*e+Kd*(e-2*e1+e2)

	def __calculate_system_output(self, y1, y2, u1, u2):
		pass
		

	#目標値rから入力データuと教師データyの取得
	def make_u_y_data_set(self, r_data_set, calculate_system_output):
		u1, u2, y, y1, y2, e1, e2= 0, 0, 0, 0, 0, 0, 0
		if not r_data_set:
			print("r_data_set does not exist")
			return None

		self.__u_data_set=np.zeros(self.__k)
		self.__y_data_set=np.zeros(self.__k)

		for i in range(self.__k):
			r=r_data_set[i]
			e=self.__calculate_error(r, y)
			u=self.__calculate_system_input(u1, e, e1, e2)
			self.__y_data_set[i] = y
			self.__u_data_set[i] = u

			#y1, y2, u1, u2の更新
			u2 = u1
			u1 = u
			y2 = y1
			y1 = y
			y = calculate_system_output(y1, y2, u1, u2)
			e2 = e1
			e1 = e

	#順方向の同定での入力ベクトルを作成する
	def make_neuralnet_forward_input_data(self, y_data_set, u_data_set):
		self.__input_data=self.__modify_forward_input_data(y_data_set, u_data_set)

	#順方向の同定での入力データの修正
	def __modify_forward_input_data(self, y_data_set, u_data_set):
		
		y_data_set1=y_data_set[:self.__k-1]
		y_data_set1=np.insert(y_data_set1, 0, 0)
		y_data_set2=y_data_set[:self.__k-2]
		y_data_set2=np.insert(y_data_set2, 0, [0, 0])

		u_data_set1=u_data_set[:self.__k-1]
		u_data_set1=np.insert(u_data_set1, 0, 0)
		u_data_set2=u_data_set[:self.__k-2]
		u_data_set2=np.insert(u_data_set2, 0, [0, 0])
		
		input_data=np.c_[y_data_set1, y_data_set2, u_data_set1, u_data_set2]

		return input_data

	#逆方向の同定での入力ベクトルを作成する
	def make_neuralnet_backward_input_data(self, y_data_set, u_data_set):
		self.__input_data=self.__modify_backward_input_data(y_data_set, u_data_set)

	#逆方向の同定での入力データの修正
	def __modify_backward_input_data(self, y_data_set, u_data_set):
		y_data_set1=y_data_set
		y_data_set2=y_data_set[:self.__k-1]
		y_data_set2=np.insert(y_data_set2, 0, 0)
		y_data_set3=y_data_set[:self.__k-2]
		y_data_set3=np.insert(y_data_set3, 0, [0, 0])

		u_data_set=u_data_set[:self.__k-2]
		u_data_set=np.insert(u_data_set, 0, [0, 0])
		input_backward_data=np.c_[y_data_set1,y_data_set2, y_data_set3, u_data_set]
		return input_backward_data


	@property
	def y_data_set(self):
		return self.__y_data_set

	@property
	def u_data_set(self):
		return self.__u_data_set

	@property
	def k(self):
		return self.__k
	
	@property
	def input_data(self):
		return self.__input_data


class System1(System):

	def __init__(self, k):
		super().__init__(k)

	#オーバーライド
	def __calculate_system_output(self, y1, y2, u1, u2):
		a1, a2 = -1, 0.35 
		b0, b1 = 1, 0.75
		return -a1*y1-a2*y2+b0*u1+b1*u2

	#順方向の同定での入力ベクトルを作成する
	def make_forward_input_data(self):
		r_data_set=self.calculate_system_target()
		self.make_u_y_data_set(r_data_set, self.__calculate_system_output)
		self.make_neuralnet_forward_input_data(self.y_data_set, self.u_data_set)

	#逆方向の同定での入力ベクトルを作成する
	def make_backward_input_data(self):
		r_data_set=self.calculate_system_target()
		self.make_u_y_data_set(r_data_set, self.__calculate_system_output)
		self.make_neuralnet_backward_input_data(self.y_data_set, self.u_data_set)


class System2(System):
	def __init__(self, k):
		super().__init__(k)

	#オーバーライド
	def __calculate_system_output(self, y1, y2, u1, u2):
		return (y1*y2*(y1+0.5)+0.5*u1)/(1+y1*y1+y2*y1)+(0.2*y2*u1-0.3*u2)/(1+y1*y2)

	#順方向の同定での入力ベクトルを作成する
	def make_forward_input_data(self):
		r_data_set=self.calculate_system_target()
		self.make_u_y_data_set(r_data_set, self.__calculate_system_output)
		self.make_neuralnet_forward_input_data(self.y_data_set, self.u_data_set)

	#逆方向の同定での入力ベクトルを作成する
	def make_backward_input_data(self):
		r_data_set=self.calculate_system_target()
		self.make_u_y_data_set(r_data_set, self.__calculate_system_output)
		self.make_neuralnet_backward_input_data(self.y_data_set, self.u_data_set)


if __name__ =="__main__":

	system2=System2(1000)
	system2.make_forward_input_data()
	#順方向の場合，training_data = y
	#逆方向の場合，training_data = u
	training_data=system2.y_data_set
	input_data=system2.input_data

	# system1=System1(1000)
	# system1.make_backward_input_data()
	# #順方向の場合，training_data = y
	# #逆方向の場合，training_data = u
	# training_data=system1.u_data_set
	# input_data=system1.input_data

	print(input_data)

	neuralnet=NeuralNet(4, 20, input_data, training_data)
	neuralnet.train(eta=0.2)

	neuralnet.make_graph()
	



