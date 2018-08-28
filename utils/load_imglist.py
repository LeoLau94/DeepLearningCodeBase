import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import os
import os.path


def default_loader(path):
    img = Image.open(path)
    return img


def default_list_reader(fileList):
    imgList = []
    print(fileList)
    with open(fileList, 'r') as file:
        for line in file.readlines():
            imgPath, label = line.strip().split(' ')
            imgList.append((imgPath, int(label)))
    return imgList


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
            root=None,
            fileList=None,
            transform=None,
            list_reader=None,
            loader=None):
        if list_reader is None:
            list_reader = default_list_reader
        self.root = root
        self.imgList = list_reader(fileList)
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
