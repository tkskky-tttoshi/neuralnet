

#def __make_inputs(self):
#        u=np.zeros(1000)
#        y=np.zeros(1000)
        

 #システムの出力誤算
    def __system_error(self, training_data_y, target):
        return target-training_data_y
