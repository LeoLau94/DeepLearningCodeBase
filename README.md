# DeepLearningCodeBase

**Based on Pytorch0.4.0**
**It is mainly used to implement pruning research for now**
**However, we will try to make it more general to handel different jobs such as Object Detection, Semantic Segmentation and so on. \(such a long way to go\)**

#Content
+ main.py
+ utils/
+ nets/
+ prune.py
+ prune\_resnet.py

##main.py
This is the main function of the codebase in which we use parser to parse the command arguments
and then initial the datasets, models, optimizers & criterions and finally call trainer to train, validate & save checkpoint as well as moded\_parameters. Futheremore, we will add more comments soon after.

##utils/
This is a folder includes some useful functions, among which the trainer.py is the most important. Because the most implementations of deep learning training process are defined in it. We will provide more details in the README soon after.

##nets/
This is a folder includes some our own models. We will provide more details in the README soon after.

##prune.py
This is the python code to prune the plain CNN such as [VGG](https://arxiv.org/pdf/1409.1556.pdf).

##prune\_resnet.py
This is the python code to prune the non-plain CNN such [ResNet](https://arxiv.org/pdf/1512.03385.pdf). We will soon provide code to prune [DenseNet](https://arxiv.org/pdf/1608.06993.pdf).

#To Do:
+ implement a theoretical Quantilization and then convert it to a practical one
+ Make the codebase more conform to a design mode.
+ To be continued
