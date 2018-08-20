# DeepLearningCodeBase

**Based on Pytorch0.4.0**

**It is mainly used to implement pruning research for now**

**However, we will try to make it more general to handel different jobs such as Object Detection, Semantic Segmentation and so on. \(such a long way to go\)**

# Content
+ main.py
+ utils/
  - trainer.py
  - plugins.py
  - convert_DataParallel_Model.py
  - getClassesFromOfficialDataset.py
  - measure.py
+ nets/
+ prune.py
+ prune\_resnet.py

## main.py
This is the main function of the codebase in which we use parser to parse the command arguments
and then initial the datasets, models, optimizers & criterions and finally call trainer to train, validate & save checkpoint as well as moded\_parameters. Futheremore, we will add more comments soon after.

## utils/
This is a folder includes some useful functions, among which the trainer.py is the most important. Because the most implementations of deep learning training process are defined in it. We will provide more details in the README soon after.

- ### trainer.py

  The fundamental implement of training process which includes training, evaluation(validation), checkpoint saver(save), checkpoint loader(resume) and so on. We wrapped them into a **Trainer** class which consists of a few attributes such as *self.model*, *self.optimizer*, *self.dataloader(train_loader and validation_loader)* etc. and some method such as *self.start(the only method you need to call outside)*, *self.train*, *self.validate* etc. The main idea is to make the training process as flexible and low coupling as possible and thus the key core is the *self.plugins* which allow users customize the process. The *self.plugins* consists of five lists **'iteration', 'epoch', 'batch', 'update', 'dataforward'**

  1. **iteration**, typically called after the output of model. For example, we usually use this kind of plugin to accuracy and then log them out.
  2. **epoch**, usually called after an epoch has done. For example, we usually use it to write the data into [tensorboardX](https://github.com/lanpa/tensorboardX)
  3. **batch**, It can be literally explained as a preprocess before we put the batch data into model.
  4. **update**, well, this kind of plugin is mainly used to update the model during we backward the loss and then we use optimizer.step() to really update. For example, we use it to implement LASSO in model.
  5. **dataforward**, literally speaking, you just need to define one to decide how to lead the data flow and calculate the loss. For example, we apply it to implement **Knowledge Distillation** which requires the two forward procedures for teacher model and student model while we also have to calculation the *MSEloss* between them.

- ### plugins.py

  **You will basically understand what's inside, won't you? Take a look, bro.+**

- ### convert_DataParallel_Model.py

  Due to the different serialized storages of PyTorch's normal module and dataparallel module, we have to convert the serialized storage between them. This is a function to convert dataparallel module to a normal module.

- ### getClassesFromOfficialDataset.py

  It is glad to see the latest version of torchvision had added class(labels) names to the official torchvision datasets such as cifar10, cifar100 etc. though the latest version hasn't released. Therefore, I had implemented a function to get those label names according to the latest source code of [torchvision.dataset.cifar](https://github.com/pytorch/vision/blob/master/torchvision/datasets/cifar.py). It is merely used in printing the class accuracy(check the code in plugins.py)

- ### measure.py

  Well, this contains two functions which are designed to measure the total amount of parameters and FLOPs of specified model.

## nets/
This is a folder includes some our own models. We will provide more details in the README soon after.

## prune.py
This is the python code to prune the plain CNN such as [VGG](https://arxiv.org/pdf/1409.1556.pdf).

## prune\_resnet.py
This is the python code to prune the non-plain CNN such [ResNet](https://arxiv.org/pdf/1512.03385.pdf). We will soon provide code to prune [DenseNet](https://arxiv.org/pdf/1608.06993.pdf).

# To Do:
- [x] ~~elaborate the utils module~~
- [ ] add more and normative comments to the .py (incoming as long as I'm not busy)
- [ ] implement a theoretical Quantilization and then convert it to a practical one
- [ ] Make the codebase more conform to a design mode.
- [ ] To be continued
