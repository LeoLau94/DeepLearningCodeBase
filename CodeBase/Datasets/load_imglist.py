import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from random import shuffle
from PIL import Image
import os
import os.path


def default_loader(path):
    img = Image.open(path)
    return img


def default_list_reader(fileList):
    imgList = []
    classes = set()
    print(fileList)
    with open(fileList, 'r') as file:
        for line in file.readlines():
            imgPath, label = line.strip().split(' ')
            classes.add(int(label))
            imgList.append((imgPath, int(label)))
    return imgList, classes


def default_attr_reader(attrlist):
    attr = {}
    for attrfile in attrlist:
        with open(attrfile, 'r') as file:
            # line 1 is the number of pic
            file.readline()
            # line 2 are attr names
            attrname = file.readline().strip().split(' ')
            # the rest are val
            for line in file.readlines():
                val = line.strip().split()
                pic_name = val[0]
                val.pop(0)
               # img_attr = {}
               #  if pic_name in attr:
               #      img_attr = attr[pic_name]

               #  for i, name in enumerate(attrname, 0):
               #      # maybe can store as str. do not use int
               #      img_attr[name] = int(val[i])
                attr[pic_name] = list(map(int, val))
    return attr, attrname


class ImageList(data.Dataset):

    def __init__(
            self,
            root,
            fileList,
            transform=None,
            list_reader=None,
            loader=None):
        if list_reader is None:
            list_reader = default_list_reader
        self.root = root
        self.imgList, self.classes = list_reader(fileList)
        self.transform = transform
        self.loader = loader if loader is not None else default_loader

    def __getitem__(self, index):
        imgPath, target = self.imgList[index]
        img = self.loader(os.path.join(self.root, imgPath))

        if self.transform is None:
            self.transform = transforms.Compose([transforms.ToTensor()])
        else:
            pass
        img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.imgList)


# Author: GuoJie Liu
class CelebADataset(ImageList):

    def __init__(self, attr_reader=default_attr_reader, attrList=[], **kwargs):
        super(CelebADataset, self).__init__(**kwargs)
        self.attr, self.attrname = attr_reader(attrList)

    def __getitem__(self, index):
        imgPath, target = self.imgList[index]
        img = self.loader(os.path.join(self.root, imgPath))
        img_attr = self.attr[imgPath]
        if self.transform is None:
            self.transform = transforms.Compose([transforms.ToTensor()])
        else:
            pass
        img = self.transform(img)
        return img, target, torch.Tensor(img_attr)

# Author: Leo Lau


class XueLangR2Dataset(data.Dataset):

    def __init__(
        self,
        root,
        fileList,
        train=True,
        bi_classification=True,
        transform=None,
     loader=None):
        self.root = root
        self.fileList = fileList
        self.train = train
        self.bi_classification = bi_classification
        self.transform = transform
        self.loader = loader if loader is not None else default_loader
        if self.train:
            self.normal_list, self.flaw_list, self.sample_len = self.xuelang_list_reader()
            self.reset()
        else:
            self.imgList = sum(list(self.xuelang_list_reader()), [])

    def __getitem__(self, index):
        imgPath, target = self.imgList[index]
        img = self.loader(os.path.join(self.root, imgPath))

        if self.transform is None:
            self.transform = transforms.Compose([transforms.ToTensor()])
        else:
            pass
        img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.imgList)

    def xuelang_list_reader(self):
        normal_list = []
        flaw_list = []
        with open(self.fileList, 'r') as f:
            if self.train:
                sample_len = int(f.readline()) * 28
            else:
                sample_len = []
            for line in f.readlines():
                imgPath, label = line.strip().split('\t')
                label = int(label)
                if not self.bi_classification:
                    if label == 0:
                        pass
                    else:
                        flaw_list.append((imgPath, label - 1))
                else:
                    if label == 0:
                        normal_list.append((imgPath, label))
                    else:
                        flaw_list.append((imgPath, 1))

        return normal_list, flaw_list, sample_len

    def reset(self):
        if self.bi_classification:
            shuffle(self.normal_list)
            self.imgList = self.flaw_list + self.normal_list[:self.sample_len]
            shuffle(self.imgList)
        else:
            self.imgList = self.flaw_list
