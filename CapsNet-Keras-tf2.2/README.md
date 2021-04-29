# CapsNet-Keras
[![License](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/XifengGuo/CapsNet-Keras/blob/master/LICENSE)

A Keras/TensorFlow2.2 implementation of CapsNet in the paper:   
[Sara Sabour, Nicholas Frosst, Geoffrey E Hinton. Dynamic Routing Between Capsules. NIPS 2017](https://arxiv.org/abs/1710.09829)   
The current `average test error = 0.34%` and `best test error = 0.30%`.   
 
**Differences with the paper:**   
- We use the learning rate decay with `decay factor = 0.9` and `step = 1 epoch`,    
while the paper did not give the detailed parameters (or they didn't use it?).
- We only report the test errors after `50 epochs` training.   
In the paper, I suppose they trained for `1250 epochs` according to Figure A.1?
Sounds crazy, maybe I misunderstood.
- We use MSE (mean squared error) as the reconstruction loss and 
the coefficient for the loss is `lam_recon=0.0005*784=0.392`.   
This should be **equivalent** with using SSE (sum squared error) and `lam_recon=0.0005` as in the paper.


## Usage

**Step 1.
Install [TensorFlow>=2.0](https://github.com/tensorflow/tensorflow) backend.**
```
pip install tensorflow==2.2.0
```


**Step 2. Train on multi gpus**   

This requires `Keras>=2.0.9`. After updating Keras:   
```
python capsulenet-multi-gpu.py --gpus 2
```
It will automatically train on multi gpus for 50 epochs and then output the performance on test dataset.
But during training, no validation accuracy is reported.


#### Training Speed 

About `100s / epoch` on a single GTX 1070 GPU.   
About `80s / epoch` on a single GTX 1080Ti GPU.   
About `55s / epoch` on two GTX 1080Ti GPU by using `capsulenet-multi-gpu.py`.      


## Other Implementations

- PyTorch:
  - [XifengGuo/CapsNet-Pytorch](https://github.com/XifengGuo/CapsNet-Pytorch)
  - [timomernick/pytorch-capsule](https://github.com/timomernick/pytorch-capsule)
  - [gram-ai/capsule-networks](https://github.com/gram-ai/capsule-networks)
  - [nishnik/CapsNet-PyTorch](https://github.com/nishnik/CapsNet-PyTorch.git)
  - [leftthomas/CapsNet](https://github.com/leftthomas/CapsNet)
  
- TensorFlow:
  - [naturomics/CapsNet-Tensorflow](https://github.com/naturomics/CapsNet-Tensorflow.git)   
  I referred to some functions in this repository.
  - [InnerPeace-Wu/CapsNet-tensorflow](https://github.com/InnerPeace-Wu/CapsNet-tensorflow)   
  - [chrislybaer/capsules-tensorflow](https://github.com/chrislybaer/capsules-tensorflow)

- MXNet:
  - [AaronLeong/CapsNet_Mxnet](https://github.com/AaronLeong/CapsNet_Mxnet)
  
- Chainer:
  - [soskek/dynamic_routing_between_capsules](https://github.com/soskek/dynamic_routing_between_capsules)

- Matlab:
  - [yechengxi/LightCapsNet](https://github.com/yechengxi/LightCapsNet)
