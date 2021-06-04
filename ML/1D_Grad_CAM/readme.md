목표: 모델이 중요하게 생각하는 신호 부분 확인 <br>

Classification Model <br>
  - 16kHz 신호 1초분량에 대해 잡음이 있는지 없는지 판별. <br>
  - Determining whether or not there is noise for 1 second of a 16 kHz signal. <br>

Reference Code
https://github.com/GunhoChoi/Grad-CAM-Pytorch/blob/master/GradCAM_MNIST.ipynb
<br>
<br>
Cons of CAM <br>
  - Essential GAP(Global Averaging Pooling) Layer <br>
  - So, it is necessary to modify the structure of the model when obtaining inference and CAM <br>
<br>
Pros of Grad CAM <br>
  -Don't have to modify the structure of the model <br>
  -Using Back prop gradient information <br>
<br>
<br>
Example. <br>
<img src="https://user-images.githubusercontent.com/72729802/120778067-e4bc2580-c560-11eb-9da6-eea81c1ade33.png"/>
