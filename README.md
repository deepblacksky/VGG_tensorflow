# VGG_tensorflow
this code implement VGG16 network by this [paper](https://arxiv.org/abs/1409.1556).

# graph
![](https://github.com/deepblacksky/VGG_tensorflow/blob/master/image/1.png?raw=true)

# Training
- load pre-trained parameters from here: https://mega.nz/#!YU1FWJrA!O1ywiCS2IiOlUCtCpI6HTJOMrneN-Qdv3ywQP5poecM
- the pre-trained parameters were trained on ImageNet DataSet, 1000 clasess.
- Remove the final FC layer and add one 10 nodes FC layer to apply for Cifar10 DataSet 
https://www.cs.toronto.edu/~kriz/cifar.html

## first training
use the pre_trained convolution layer parameters, and train the FC layer parameters

### Result

![](https://github.com/deepblacksky/VGG_tensorflow/blob/master/image/2.png?raw=true)

![](https://github.com/deepblacksky/VGG_tensorflow/blob/master/image/3.png?raw=true)

![](https://github.com/deepblacksky/VGG_tensorflow/blob/master/image/4.png?raw=true)

## second training

Do not use the pre-trained parameter, the network train all parameters

### Result

![](https://github.com/deepblacksky/VGG_tensorflow/blob/master/image/5.png?raw=true)

![](https://github.com/deepblacksky/VGG_tensorflow/blob/master/image/6.png?raw=true)

![](https://github.com/deepblacksky/VGG_tensorflow/blob/master/image/7.png?raw=true)