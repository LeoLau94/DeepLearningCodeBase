import os
import sys
import argparse
import pickle
import torch
import numpy as np
import torch.backends.cudnn as cudnn
import torchvision.transforms as tf
sys.path.append('/WorkSpace')
from CodeBase.Utils import *
from CodeBase.Models import *
from CodeBase.Datasets import *
from datetime import datetime

current_time = datetime.now().strftime('%b%d_%H-%M-%S')


def inferenceToInterpretability(args):
    resultPath = os.path.join(args.save_path, 'FinalResult.pkl')
    attributeNamePath = os.path.join(args.save_path, 'AttributeNames.pkl')


    torch.manual_seed(args.seed)
    model = sphere20(num_classes=78897)
    model.load_state_dict(torch.load(args.model))
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        model.cuda()
        cudnn.benchmark = True

    kwargs = {'num_workers': args.num_workers,
            'pin_memory': True if args.cuda else False}
    print('-----------------Start To Load CelebA--------------------')
    image_root = os.path.join(args.image_root_path, 'img_align_celeba')
    fileList = os.path.join(args.image_root_path, 'Anno/identity_CelebA.txt')
    attrList = os.path.join(args.image_root_path, 'Anno/list_attr_celeba.txt')
    loader = torch.utils.data.DataLoader(
        CelebADataset(
            root=image_root,
            fileList=fileList,
            attrList=[attrList],
            transform=tf.Compose([
                tf.Resize((256, 256)),
                tf.ToTensor()
            ])
        ),
        batch_size=args.batch_size,
        shuffle=False,
        **kwargs
    )
    print('-----------------Loading CelebA Successful--------------------')

    # HiddenLayerOutpuState = np.zeros((40, 512))
    # AttributesAppearance = np.zeros((40, 1))
    Positive_HiddenLayerOutpuState = np.zeros((40, 512))
    Positive_AttributesAppearance = np.zeros((40, 1))
    Negative_HiddenLayerOutpuState = np.zeros((40, 512))
    Negative_AttributesAppearance = np.zeros((40, 1))
    img_attr = None
    img_name = loader.dataset.attrname


    def hidden_layer_hook(self, input, output):

        print('-----------------Start To Hook--------------------')
        for sample, attribute in zip(output, img_attr):
            sample = sample.detach().cpu().numpy()
            attribute = attribute.numpy()
            positive_mask = attribute == 1
            Positive_AttributesAppearance[positive_mask] += 1
            # print(AttributesAppearance)
            Positive_HiddenLayerOutpuState[positive_mask] += np.abs(sample)
            negative_mask = attribute == -1
            Negative_AttributesAppearance[negative_mask] += 1
            Negative_HiddenLayerOutpuState[negative_mask] += np.abs(sample)
        print('----------------- Hooking Successful--------------------')
    handle = model.fc5.register_forward_hook(hidden_layer_hook)

    print('-----------------Start To Inference--------------------')
    total = len(loader.dataset)
    count = 0
    for data, _, attributes in iter(loader):
        if args.cuda:
            data = data.cuda()
        count += data.size(0)
        img_attr = attributes
        model(data)
        print(
            '-------------Now Processing [{}/{}] {:.2f}%----------------'.format(
                count,
                total,
                count *
                100 /
                total))
    handle.remove()
    print('-----------------Inference Successful-------------------')

    Positive_FinalResult = Positive_HiddenLayerOutpuState / Positive_AttributesAppearance
    Negative_FinalResult = Negative_HiddenLayerOutpuState / Negative_AttributesAppearance
    FinalResult = np.abs(Positive_FinalResult - Negative_FinalResult)

    with open(resultPath, 'wb') as f1, open(attributeNamePath, 'wb') as f2:
        pickle.dump(FinalResult, f1, 0)
        pickle.dump(img_name, f2, 0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Interpretability Test')
    parser.add_argument('--model', type=str, default='',
                        help='path to model checkpoint')
    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--num-workers', type=int, default=1,
                        help='how many threads to load data(default: 1)')
    parser.add_argument(
        '--image-root-path',
        default='',
        type=str,
        metavar='PATH',
     help='path to root path of images (default: none)')
    parser.add_argument('--save-path', default='', type=str, metavar='PATH',
                        help='path to save hook results (default: none)')
    parser.add_argument('--no-inference', action='store_true', default=False,
                        help='disable model inference')
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.inference = not args.no_inference
    inferenceToInterpretability(args)
