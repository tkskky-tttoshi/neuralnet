# neuralnet
離散時間システムの3層ニューラルネットワークによる同定
##順方向
<img src="https://latex.codecogs.com/gif.latex?e(k)=y(k)-y_{nn}(k)"/>

##逆方向では，
<img src="https://latex.codecogs.com/gif.latex?e(k)=u(k-1)-y_{nn}(k)"/>

重みの更新方法は，いずれも誤差逆伝播法
