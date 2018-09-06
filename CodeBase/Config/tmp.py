
model_dict = {
    'vgg': vgg_diy,
    'resnet': preactivation_resnet164,
    'sphereface': sphere20,
    'se_inception_v3': se_inception_v3,
}
model_kwargs_dict = {
    'vgg': {'num_classes': args.num_classes},
    'resnet': {'num_classes': args.num_classes},
    'sphereface': {'num_classes': args.num_classes},
    'se_inception_v3': {
        'num_classes': args.num_classes,
        'transform_input': True,
    },
}
plugins = []

plugins.append(LossMonitor())
plugins.append(TopKAccuracy(topk=(1, 5 if args.num_classes > 5 else 1)))
plugins.append(IterationSummaryMonitor())
plugins.append(DistributionOfBNMonitor())
if args.eca:
    plugins.append(ClassAccuracy())

if args.model not in model_dict:
    raise ValueError('Name of network unknown %s' % args.model)
else:
    if args.fine_tune:
        load_pkl = torch.load(args.fine_tune)
        model = model_dict[args.model](
            **model_kwargs_dict[args.model], cfg=load_pkl['cfg'])
        model.load_state_dict(load_pkl['model_state_dict'])
        if args.teacher_model is not None:
            teacher_model = model_dict[args.model](
                **model_kwargs_dict[args.model])
            teacher_model.load_state_dict(torch.load(args.teacher_model))
        else:
            pass
        args.save_path = os.path.join(
            args.save_path,
            'fine_tune')
    else:
        model = model_dict[args.model](**model_kwargs_dict[args.model])


kwargs = {'num_workers': args.num_workers,
          'pin_memory': True} if args.cuda else {}

dataset_root = '/data/torchvision/'
scheduler_class = None
scheduler_kwargs = None
if args.dataset == 'cifar10':
    normalize = transforms.Normalize(
        mean=[0.491, 0.482, 0.447],
        std=[0.247, 0.243, 0.262])

    base_folder = 'cifar-10-batches-py'

    train_dataset = datasets.CIFAR10(dataset_root, train=True, download=True,
                                     transform=transforms.Compose([
                                         transforms.RandomCrop(32, padding=4),
                                         transforms.RandomHorizontalFlip(),
                                         # transforms.ColorJitter(brightness=1),
                                         transforms.ToTensor(),
                                         normalize
                                         ]))
    validate_dataset = datasets.CIFAR10(dataset_root, train=False,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            normalize
                                            ]))

    train_dataset.classes = cifar_load_meta(
        dataset_root, base_folder, args.dataset)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
     **kwargs)
    validate_loader = torch.utils.data.DataLoader(
         dataset=validate_dataset,
         batch_size=args.validate_batch_size, shuffle=False, **kwargs
         )

elif args.dataset == 'cifar100':
    normalize = transforms.Normalize(
        mean=[0.507, 0.487, 0.441],
        std=[0.267, 0.256, 0.276])

    base_folder = 'cifar-100-python'

    train_dataset = datasets.CIFAR100(
        dataset_root, train=True, download=True,
        transform=transforms.Compose(
            [transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             normalize]))
    validate_dataset = datasets.CIFAR100(dataset_root, train=False,
                                         transform=transforms.Compose([
                                             transforms.ToTensor(),
                                             normalize
                                             ]))

    train_dataset.classes = cifar_load_meta(
        dataset_root, base_folder, args.dataset)
    # normalize = transforms.Normalize(mean=[.5,.5,.5], std=[.5,.5,.5])
    # normalize = transforms.Normalize((.5,.5,.5),(.5,.5,.5))
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
     **kwargs)
    validate_loader = torch.utils.data.DataLoader(
         dataset=validate_dataset,
         batch_size=args.validate_batch_size, shuffle=False, **kwargs
         )


elif args.dataset == 'webface':
    train_root = os.path.join(args.image_root_path, 'webface_train')
    validate_root = os.path.join(args.image_root_path, 'webface_val')
    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(root=train_root,
                             transform=transforms.Compose([
                                 transforms.RandomCrop(256, padding=4),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 ])
                             ),
        batch_size=args.batch_size, shuffle=True, **kwargs
    )
    validate_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            root=validate_root,
            transform=transforms.Compose([
                transforms.ToTensor()
            ])
        ),
        batch_size=args.validate_batch_size, shuffle=False, **kwargs)

    # scheduler_class = scheduler.MultiStepLR
    # scheduler_kwargs = {'milestones': [8, 14], 'gamma': 0.1}

elif args.dataset == 'xuelangR2':
    args.dataset += '_%dClasses' % args.num_classes
    def list_reader_func(fileList):
        imgList = []
        print(fileList)
        with open(fileList, 'r') as file:
            for line in file.readlines():
                results = line.strip().split(' ')
                imgPath, label = ' '.join(results[:-1]), int(results[-1])
                imgList.append((imgPath, label))
        return imgList

    if args.num_classes == 2:
        fileRoot = os.path.join(args.image_root_path, 'trainval')
        trainList = os.path.join(args.image_root_path, 'train.txt')
        valList = os.path.join(args.image_root_path, 'val.txt')


        train_T = transforms.Compose([
            transforms.RandomCrop(512, padding=8),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor()
        ])
        val_T = transforms.Compose([
            transforms.ToTensor()
        ])

        train_dataset = ImageList(
            root=fileRoot,
            fileList=trainList,
            transform=train_T,
            list_reader=list_reader_func,
        )
        train_dataset.classes = [
            'class %d' %
            i for i in range(
                args.num_classes)]
        validate_dataset = ImageList(
            root=fileRoot,
            fileList=valList,
            transform=val_T,
            list_reader=list_reader_func
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size, shuffle=True, **kwargs)
        validate_loader = torch.utils.data.DataLoader(
            validate_dataset,
            batch_size=args.validate_batch_size, shuffle=False, **kwargs)
       #  scheduler_class = scheduler.MultiStepLR
       #  scheduler_kwargs = {'milestones': [10, 20], 'gamma': 0.1}
    else:
        pass

    # fileRoot = os.path.join(args.image_root_path, 'crop_jpg')
    # trainList = os.path.join(args.image_root_path, 'train.txt')
    # valList = os.path.join(args.image_root_path, 'val.txt')

    # train_T = transforms.Compose([
    #     transforms.Resize((299, 299)),
    #     transforms.RandomCrop(299, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomVerticalFlip(),
    #     transforms.ToTensor()
    # ])
    # val_T = transforms.Compose([
    #     transforms.Resize((299, 299)),
    #     transforms.ToTensor()
    # ])

    # train_dataset = XueLangR2Dataset(
    #     fileRoot, trainList, bi_classification=True
    #     if args.num_classes == 2 else False, transform=train_T)

    # train_dataset.classes = ['class %d' % i for i in range(args.num_classes)]

    # validate_dataset = XueLangR2Dataset(
    #     fileRoot, valList, train=False, bi_classification=True
    #     if args.num_classes == 2 else False, transform=val_T)

    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     batch_size=args.batch_size, shuffle=True, **kwargs)
    # validate_loader = torch.utils.data.DataLoader(
    #     validate_dataset,
    #     batch_size=args.validate_batch_size, shuffle=False, **kwargs)
    # plugins.append(DownSampleMonitor())

else:
    raise NotImplementedError("Patience bro, coming soon")
    # fileRoot = os.path.join(args.image_root_path, 'trainval')
    # trainList = os.path.join(args.image_root_path, 'train.txt')
    # valList = os.path.join(args.image_root_path, 'val.txt')
    # train_T = None
    # val_T = None
    # list_reader = None
    # loader = None

    #     def list_reader_func(fileList):
    #         imgList = []
    #         print(fileList)
    #         with open(fileList, 'r') as file:
    #             for line in file.readlines():
    #                 results = line.strip().split(' ')
    #                 imgPath, label = ' '.join(results[:-1]), int(results[-1])
    #                 imgList.append((imgPath, label))
    #         return imgList
    #     list_reader = list_reader_func
    # train_dataset = ImageList(
    #     root=fileRoot,
    #     fileList=trainList,
    #     transform=train_T,
    #     list_reader=list_reader,
    #     loader=loader
    # )
    # train_dataset.classes = ['class %d' % i for i in range(args.num_classes)]
    # validate_dataset = ImageList(
    #     root=fileRoot,
    #     fileList=valList,
    #     transform=val_T,
    #     list_reader=list_reader,
    #     loader=loader
    # )

    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     batch_size=args.batch_size, shuffle=True, **kwargs)
    # validate_loader = torch.utils.data.DataLoader(
    #     validate_dataset,
    #     batch_size=args.validate_batch_size, shuffle=False, **kwargs)
optimizer = optim.SGD(
   filter(
       lambda p: p.requires_grad,
       model.parameters()),
   lr=args.lr,
   weight_decay=args.weight_decay,
   momentum=args.momentum,
   nesterov=True)
# optimizer = optim.Adam(
#     filter(
#         lambda p: p.requires_grad,
#         model.parameters()),
#     lr=args.lr,
#     amsgrad=True,
#     weight_decay=args.weight_decay)
criterion = nn.CrossEntropyLoss()

def dataforward(self, data, target):
    output = self.trainer.model(data)
    loss = self.trainer.criterion(output, target)
    return output, loss


def inception_v3_forward(self, data, target):
    output = self.trainer.model(data)
    loss_base = self.trainer.criterion(output[0], target)
    loss_aux = self.trainer.criterion(output[1], target)
    total_loss = loss_base + 0.4 * loss_aux
    return output[0], total_loss

dataforward_kwargs = {
    'train_forward': inception_v3_forward
    if args.model == 'se_inception_v3' else dataforward,
    'validation_forward': dataforward}

if args.sr:
    print('\nTraining With LASSO\n')
    args.save_path = os.path.join(args.save_path, 'lasso')

    def updateBN(self):
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.weight.grad.data.add_(
                    self.kwargs['penalty'] *
                    torch.sign(
                        m.weight.data))
    plugins.append(DataForward(**dataforward_kwargs))
    plugins.append(ModelGradHandler(updateBN, penalty=args.p))

elif args.fine_tune is not None and args.teacher_model is not None:
    transfer_criterion = nn.MSELoss()
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad = False
    print('\nTraining with Knowledge Distillation \n')

    def train_forward(self, data, target):
        student_output = self.trainer.model(data)
        teacher_output = self.kwargs['teacher_model'](data)

        transfer_loss = self.kwargs['transfer_criterion'](
            student_output, teacher_output)
        softmax_loss = self.trainer.criterion(student_output, target)
        loss = (1 - self.kwargs['loss_ratio']
                ) * softmax_loss + self.kwargs['loss_ratio'] * transfer_loss

        return student_output, loss

    plugins.append(
        DataForward(
            train_forward,
            validation_forward=dataforward,
            teacher_model=teacher_model,
            loss_ratio=args.loss_ratio,
         transfer_loss=transfer_loss))

else:
    print('\nNormal Training \n')
    plugins.append(DataForward(**dataforward_kwargs))
