import math
import os.path as osp
from random import shuffle


__all__ = ['splitAndGenerateFileList']


# Author: Leo Lau
class DatasetSpliter:

    def __init__(
            self,
            fileList,
            ratio=0.8):
        self.fileList = fileList
        self.ratio = ratio
        self.sample_len = 0

    def start(self, train_list_path, val_list_path):
        normal_list = []
        flaw_list = [[] for i in range(10)]

        print('--------------------Start To Read FileList----------------')
        print(self.fileList)

        with open(self.fileList, 'r') as file:
            for line in file.readlines():
                results = line.strip().split('\t')
                label = int(results[-1])
                if label == 0:
                    normal_list.append(line)
                else:
                    flaw_list[label - 1].append(line)

        print('------------------FileList Reading Successful!----------------')

        print('------------------Start To Split Dataset--------------------')
        shuffle(normal_list)
        for i in range(10):
            shuffle(flaw_list[i])
        self.__split(normal_list, flaw_list)
        print('------------------Splitting Dataset Successful!--------------')

        print('------------------Start To Upsameple------------------------')
        self.__upsample()
        print('------------------Upsampling Successful!--------------------')

        print(
            '------------------Start to Write TrainList into %s--------------' %
            train_list_path)
        print(
            '------------------Start to Write ValList into %s--------------' %
            val_list_path)
        trainList = self.train_normal_list + sum(self.train_flaw_list, [])
        valList = self.val_normal_list + sum(self.val_flaw_list, [])
        print(
            'Total Train Samples:{} \t Total Validation Samples:{}'.format(
                len(trainList),
                len(valList)))
        with open(train_list_path, 'w') as f1, open(val_list_path, 'w') as f2:
            f1.write(str(self.sample_len) + '\n')
            f1.writelines('%s' % item for item in trainList)
            f2.writelines('%s' % item for item in valList)
        print('------------------Writting Successful!---------------------')

    def __split(self, normal_list, flaw_list):
        normal_split_idx = int(math.ceil(self.ratio * len(normal_list)))
        flaw_list_idx = [int(math.ceil(self.ratio * len(flaw_list[i])))
                         for i in range(10)]
        self.train_normal_list = normal_list[:normal_split_idx]
        self.train_flaw_list = [flaw_list[i][:flaw_list_idx[i]]
                                for i in range(10)]
        self.val_normal_list = normal_list[normal_split_idx:]
        self.val_flaw_list = [flaw_list[i][flaw_list_idx[i]:]
                              for i in range(10)]

    def __upsample(self):
        for i in range(10):
            list_len = len(self.train_flaw_list[i])
            assert list_len > 0, "flaw sample %d list shouldn't be empty" % (
                i + 1)
            if list_len > self.sample_len:
                self.sample_len = list_len

        self.sample_len = int(math.ceil(self.sample_len / 100.) * 100)
        print('Equilibrium Amount:%d' % self.sample_len)

        for i in range(10):
            list_len = len(self.train_flaw_list[i])
            print('Before Upsampling Class %d Amount: %d' % (i + 1, list_len))
            difference = self.sample_len - list_len
            list_copy = self.train_flaw_list[i].copy()
            while difference > 0:
                if difference > list_len:
                    self.train_flaw_list[i].extend(list_copy)
                    difference -= list_len
                else:
                    self.train_flaw_list[i].extend(
                        list_copy[:difference])
                    difference = 0
            print('After Upsampling Class %d Amount: %d' %
                  (i + 1, len(self.train_flaw_list[i])))


def splitAndGenerateFileList(fileList, train_list_path, val_list_path):
    spliter = DatasetSpliter(fileList)
    spliter.start(train_list_path, val_list_path)

if __name__ == '__main__':
    root = '/data2/public/xuelang/CropIntoPatch/'
    fileList = osp.join(root, 'crop_info.txt')
    train_list_path = osp.join(root, 'train.txt')
    val_list_path = osp.join(root, 'val.txt')
    splitAndGenerateFileList(fileList, train_list_path, val_list_path)
