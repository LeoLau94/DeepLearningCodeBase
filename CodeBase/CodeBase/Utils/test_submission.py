# coding=utf-8
# test_submission.py
import os
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F
from CodeBase.Utils import *
from CodeBase.Models import *


def default_loader(path):
    img = Image.open(path)
    return img


class XueLangTestDataset(data.Dataset):

    def __init__(
            self,
            root,
            crop_size=320):
        self.root = root
        self.crop_size = crop_size
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
        ])
        self.loader = default_loader
        self.imgList = self.__read__()

    def __read__(self):
        imgList = os.listdir(self.root)
        return imgList

    def __split_into_patch__(self, img):
        row_num = img.size[1] / self.crop_size
        col_num = img.size[0] / self.crop_size
        box_list = []
        for i in range(int(row_num)):
            for j in range(int(col_num)):
                box = np.array([j, i, j + 1, i + 1])
                box *= self.crop_size
                box_list.append(box)
        crop_imgs = [img.crop(box) for box in box_list]
        return crop_imgs

    def __getitem__(self, index):
        imgPath = self.imgList[index]
        img = self.loader(os.path.join(self.root, imgPath))
        crop_imgs = self.__split_into_patch__(img)
        return [self.transform(i) for i in crop_imgs], imgPath

    def __len__(self):
        return len(self.imgList)

if __name__ == '__main__':
    # Root Settings
    ImgRoot = "/data2/public/xuelang/Test/test_jpg"
    BinaryModelPath = "/data/leolau/models/save/se_inception_v3/xuelangR2_2Classes/Aug29_20-58-23/best_precision_model_params.pkl"
    DecimalModelPath = "/data/leolau/models/save/se_inception_v3/xuelangR2_10Classes/Aug29_20-59-56/best_precision_model_params.pkl"
    csv_file = "/data/leolau/SubmissionCSV/result.csv"

    # Dataset Settings
    test_dataset = XueLangTestDataset(ImgRoot)
    test_dataloader = data.DataLoader(
        test_dataset,
        shuffle=False, batch_size=1, num_workers=2, pin_memory=True
    )

    # Model Settings
    model1 = se_inception_v3(num_classes=2)
    model2 = se_inception_v3(num_classes=10)

    model1.load_state_dict(torch.load(BinaryModelPath))
    model2.load_state_dict(torch.load(DecimalModelPath))
    model1.cuda(0)
    model2.cuda(1)
    model1.eval()
    model2.eval()
    csv_prob = []
    csv_img_defect_names = []
    flaw_list = [
        '|defect_1',
        '|defect_2',
        '|defect_3',
        '|defect_4',
        '|defect_5',
        '|defect_6',
        '|defect_7',
        '|defect_8',
        '|defect_9',
     '|defect_10']

    def handle_overflow(x):
        if x >= 0.999999:
            return 0.999999
        elif x <= 0.000001:
            return 0.000001
        else:
            return x

    total = len(test_dataset)
    norm_count = 0
    with torch.no_grad():
        print('---------------Start To Predict----------------')
        for i, (input_data, img_names) in enumerate(test_dataloader, 1):
            print(
                '----------[{:d}/{:d}][{:.2f}%]----------'.format(i, total, i * 100 / total))
            input_data = torch.cat(input_data).cuda(0)
            output1 = F.softmax(model1(input_data), dim=1)
            output1_np = output1.detach().cpu().numpy()
            sample_idx = np.where(
                (output1_np[:, 1] - output1_np[:, 0]) > 0.2)[0]
            if len(sample_idx) > 0:
                input_data2 = input_data[sample_idx].to(device=1)
                output2 = F.softmax(model2(input_data2), dim=1)
                prob = output2.mean(dim=0).detach().cpu().numpy()
                norm_prob = min(output1_np[:, 0])
                csv_img_defect_names.append(img_names[0]+'|norm')
                csv_prob.append('{:.6f}'.format(norm_prob))
                for index, prob in enumerate(prob):
                    prob = handle_overflow(prob)
                    csv_img_defect_names.append(img_names[0]+flaw_list[index])
                    csv_prob.append('{:.6f}'.format(prob))
            else:
                norm_count += 1
                norm_prob = handle_overflow(output1_np.max(axis=0)[0])
                csv_img_defect_names.append(img_names[0]+'|norm')
                csv_prob.append('{:.6f}'.format(norm_prob))
                flaw_prob = (1 - norm_prob) / 10
                for i in range(10):
                    csv_img_defect_names.append(img_names[0]+flaw_list[i])
                    csv_prob.append('{:.6f}'.format(flaw_prob))

        print('normal sample: [{:d}/{:d}][{:2f}%]'.format(norm_count,
                                                          total, norm_count * 100 / total))
        print('---------------Prediction Successful!----------------')
        # 保存结果
        print('---------------Start To Write Prediction!--------------')
        submission = pd.DataFrame({
            "filename|defect": csv_img_defect_names,
            "probability": csv_prob})

        #submission.to_csv("submit_{}.csv".format(model_name.split('.')[0]), index=False)
        #submission.to_csv(csv_file.format(model_name.split('.')[0]), index=False)
        submission.to_csv(csv_file, index=False)
        print('---------------Writing Successful!--------------')
        # d = iter(test_dataloader).__next__()
        # print(len(d))
        # d = torch.cat(d).cuda()
        # output = F.softmax(model1(d), dim=1)
        # output_np = output.detach().cpu().numpy()
        # idx = np.where(output_np[:, 0] < output_np[:, 1])
        # d = d[idx]
        # output2 = F.softmax(model2(d), dim=1)
        # print(output2.mean(dim=0))
