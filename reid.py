from __future__ import print_function, division

import shutil

import cv2
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
from torchvision import datasets, models, transforms
import yaml
from model import ft_net, ft_net_dense, ft_net_hr, ft_net_swin, ft_net_swinv2, ft_net_efficient, ft_net_NAS, ft_net_convnext, PCB, PCB_test
from utils import fuse_all_conv_bn
from PIL import Image
import os

def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

class Reid:
    def __init__(self, model_Path, config_path, gpu_ids, ms=[1.0]):
        with open(config_path, 'r') as stream:
            self.config = yaml.load(stream, Loader=yaml.FullLoader)
        self.gpu_ids = gpu_ids
        torch.cuda.set_device(gpu_ids[0])
        cudnn.benchmark = False
        self.ms = ms
        #self.h, self.w = 256, 128
        self.h, self.w = 224, 224
        self.data_transforms = self.init_transforms()
        self.model = self.init_model(model_Path)

    def init_transforms(self):
        data_transforms = transforms.Compose([
            transforms.Resize((self.h, self.w), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return data_transforms

    def init_model(self, model_Path):
        if self.config['use_dense']:
            model_structure = ft_net_dense(self.config['nclasses'], stride=self.config['stride'], linear_num=self.config['linear_num'])
        elif self.config['use_NAS']:
            model_structure = ft_net_NAS(self.config['nclasses'], linear_num=self.config['linear_num'])
        elif self.config['use_swin']:
            model_structure = ft_net_swin(self.config['nclasses'], linear_num=self.config['linear_num'])
        elif self.config['use_swinv2']:
            model_structure = ft_net_swinv2(self.config['nclasses'], (self.h, self.w), linear_num=self.config['linear_num'])
        elif self.config['use_convnext']:
            model_structure = ft_net_convnext(self.config['nclasses'], linear_num=self.config['linear_num'])
        elif self.config['use_efficient']:
            model_structure = ft_net_efficient(self.config['nclasses'], linear_num=self.config['linear_num'])
        elif self.config['use_hr']:
            model_structure = ft_net_hr(self.config['nclasses'], linear_num=self.config['linear_num'])
        else:
            model_structure = ft_net(self.config['nclasses'], stride=self.config['stride'], ibn=self.config['ibn'], linear_num=self.config['linear_num'])

        if self.config['PCB']:
            model_structure = PCB(self.config['nclasses'])

        model_structure.load_state_dict(torch.load(model_Path))
        if self.config['PCB']:
            model_structure = PCB_test(model_structure)
        else:
            model_structure.classifier.classifier = nn.Sequential()
        model = model_structure.eval()
        model = model.cuda()
        model = fuse_all_conv_bn(model)
        print(model)
        return model

    def extract_feature(self, imgList, maxBatch=16):
        if self.config['linear_num'] <= 0:
            if self.config['use_swin'] or self.config['use_swinv2'] or self.config['use_dense'] or self.config['use_convnext']:
                self.config['linear_num'] = 1024
            elif self.config['use_efficient']:
                self.config['linear_num'] = 1792
            elif self.config['use_NAS']:
                self.config['linear_num'] = 4032
            else:
                self.config['linear_num'] = 2048
        transImgList = []
        for i in range(len(imgList)):
            print(i)
            transImgList.append(self.data_transforms(imgList[i]))
        imgs = torch.stack(transImgList, 0)
        n, c, h, w = imgs.size()
        features = torch.FloatTensor(n, self.config['linear_num'])
        batchnum = n // maxBatch + 1
        for i in range(batchnum):
            print(i)
            start = i * maxBatch
            end = min(start + maxBatch, n)
            imgBatch = imgs[start:end]
            ff = torch.FloatTensor(end-start, self.config['linear_num']).zero_().cuda()
            if self.config['PCB']:
                ff = torch.FloatTensor(end-start, 2048, 6).zero_().cuda()  # we have six parts

            for i in range(2):
                if(i==1):
                    imgBatch = fliplr(imgBatch)
                input_img = Variable(imgBatch.cuda())
                for scale in self.ms:
                    if scale != 1:
                        # bicubic is only  available in pytorch>= 1.1
                        input_img = nn.functional.interpolate(input_img, scale_factor=scale, mode='bicubic', align_corners=False)
                    outputs = self.model(input_img)
                    ff += outputs
            if  self.config['PCB']:
                # feature size (n,2048,6)
                # 1. To treat every part equally, I calculate the norm for every 2048-dim part feature.
                # 2. To keep the cosine score==1, sqrt(6) is added to norm the whole feature (2048*6).
                fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(6)
                ff = ff.div(fnorm.expand_as(ff))
                ff = ff.view(ff.size(0), -1)
            else:
                fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
                ff = ff.div(fnorm.expand_as(ff))
            features[start:end] = ff
        return features

def getAllFile(picPath, key):
    picList = []
    for root, dirs, files in os.walk(picPath):
        for file in files:
            if key in file:
                picList.append(os.path.join(root,file))
    return picList

def matchList(list1, list2):
    for i in range(len(list1)):
        if i in list2:
            return True
    return False

def selfmatch(scoreMap, idx, matchList):
    for idy in range(len(scoreMap[idx])):
        if scoreMap[idx][idy]:
            if idy not in matchList:
                matchList.append(idy)
                selfmatch(scoreMap, idy, matchList)

def reidMatch(qf, th, imgList=None, savePath=None):
    gf = torch.transpose(qf,1,0)
    scoreMap = torch.mm(qf,gf)
    print(scoreMap)
    matchIndex = torch.gt(scoreMap, th)
    print(matchIndex)
    matchMap = []
    for i in range(len(matchIndex)):
        ifMatched = False
        for j in range(len(matchMap)):
            if i in matchMap[j]:
                ifMatched = True
        if not ifMatched:
            matchList = [i]
            selfmatch(matchIndex, i, matchList)
            matchMap.append(matchList)
    print(matchMap)
    # for i in range(len(matchMap)):
    #     if not os.path.exists(savePath+'/'+str(i)):
    #         os.makedirs(savePath+'/'+str(i))
    #     for j in matchMap[i]:
    #         shutil.copy(imgList[j], savePath+'/'+str(i))
    # matchList =[]
    # for idx in range(len(matchIndex)):
    #     matchItem = [idx]
    #     for idy in range(len(matchIndex[idx])):
    #         if matchIndex[idx][idy]:
    #             matchItem.append(idy)
    #     matchList.append(matchItem)
    # while(len(matchList) > 0):
    #     popItem = matchList.pop(0)
    #     for i in range(len(matchList)):
    #         if matchList(popItem, matchList[i]):

if __name__ == '__main__':
    model_path = '/home/chase/shy/Person_reID_baseline_pytorch/model/ft_ResNet50/net_60.pth'
    config_path = '/home/chase/shy/Person_reID_baseline_pytorch/model/ft_ResNet50/opts.yaml'
    imgPath = '/home/chase/PycharmProjects/data/save7/93'
    savePath = '/home/chase/PycharmProjects/data/save3'
    test = Reid(model_path, config_path, [1])
    imgList = getAllFile(imgPath, '.jpg')
    plimgList = []
    for img in imgList:
        print(img)
        img =cv2.imread(img)
        img = Image.fromarray(img)
        plimgList.append(img)
    with torch.no_grad():
        ff = test.extract_feature(plimgList)
        reidMatch(ff,0.7, imgList, savePath)

