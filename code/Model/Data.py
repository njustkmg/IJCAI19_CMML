# -*- coding: utf-8 -*-
import torch
import pickle
import numpy as np
from PIL import Image
import torch.utils.data as Data
import torchvision.transforms as transforms

class MyDataset(Data.Dataset):
    def __init__(self, imgfilenamerecord, imgfilename, textfilename, labelfilename, train = True, supervise = True, traintestproportion = 0.667, superviseunsuperviseproportion = [7, 3]):
        super(MyDataset, self).__init__()
        self.imgfilenamerecord = imgfilenamerecord
        self.imgfilename = imgfilename
        self.textfilename = textfilename
        self.labelfilename = labelfilename
        self.train = train
        self.supervise = supervise
        self.pro1 = traintestproportion
        self.pro2 = superviseunsuperviseproportion
        
        #print(self.pro2)
        #exit(-1)

        fr = open(self.imgfilenamerecord,'rb')
        self.imgrecordlist = pickle.load(fr)
        for i in range(len(self.imgrecordlist)):
            self.imgrecordlist[i] = self.imgfilename + self.imgrecordlist[i]
        self.imgrecordlist = np.array(self.imgrecordlist)
        self.textlist = np.load(self.textfilename)
        self.labellist = np.load(self.labelfilename)
        #print(self.labellist.shape)
        #exit(-1)
        
        '''
        upset data.
        '''
        permutation = np.random.permutation(len(self.imgrecordlist))        
        self.imgrecordlist = self.imgrecordlist[permutation]
        self.textlist = self.textlist[permutation, :]
        self.labellist = self.labellist[permutation, :]    
        
        #print(int(self.pro1*self.pro2*len(self.imgrecordlist)))
        #print(int(self.pro1*len(self.imgrecordlist))- int(self.pro1*self.pro2*len(self.imgrecordlist)))
        #exit(-1)
        self.samplesize = int(int(self.pro1*len(self.imgrecordlist))/(self.pro2[0] + self.pro2[1]))
        #print(self.samplesize)
        self.superviseimgrecordlist = self.imgrecordlist[0:self.samplesize * self.pro2[0]]
        self.supervisetextlist = self.textlist[0:self.samplesize * self.pro2[0]]
        self.superviselabellist = self.labellist[0:self.samplesize * self.pro2[0]]
        #print(len(self.superviseimgrecordlist))
        self.unsuperviseimgrecordlist = self.imgrecordlist[self.samplesize * self.pro2[0]:self.samplesize * (self.pro2[0] + self.pro2[1])]
        self.unsupervisetextlist = self.textlist[self.samplesize * self.pro2[0]:self.samplesize * (self.pro2[0] + self.pro2[1])]
        self.unsuperviselabellist = self.labellist[self.samplesize * self.pro2[0]:self.samplesize * (self.pro2[0] + self.pro2[1])]
        #print(len(self.unsuperviseimgrecordlist))
        #exit(-1)
        self.testimgrecordlist = self.imgrecordlist[int(self.pro1*len(self.imgrecordlist)):len(self.imgrecordlist)]
        self.testtextlist = self.textlist[int(self.pro1*len(self.textlist)):len(self.textlist)]
        self.testlabellist = self.labellist[int(self.pro1*len(self.labellist)):len(self.labellist)]
        

    def supervise_(self):
        self.train = True
        self.supervise = True
        return self

    def test_(self):
        self.train = False
        return self

    def unsupervise_(self):
        self.train = True
        self.supervise = False
        return self

    def __getitem__(self, index):
        if self.train == True and self.supervise == True:
            img = Image.open(self.superviseimgrecordlist[index]).convert('RGB').resize((256, 256))
            text = self.supervisetextlist[index]
            label = self.superviselabellist[index]
            img = transforms.ToTensor()(img)
            text = torch.FloatTensor(text)
            label = torch.FloatTensor(label)
            feature = []
            feature.append(img)
            feature.append(text)
            return feature, label
        elif self.train == True and self.supervise == False:
            supervise_img = []
            supervise_text = []
            supervise_label = []
            for i in range(index*self.pro2[0],(index+1)*self.pro2[0]):
                temp_img = Image.open(self.superviseimgrecordlist[i]).convert('RGB').resize((256, 256))
                temp_text = self.supervisetextlist[i]
                temp_label = self.superviselabellist[i]
                temp_img = transforms.ToTensor()(temp_img)
                temp_text = torch.FloatTensor(temp_text)
                temp_label = torch.FloatTensor(temp_label)
                supervise_img.append(temp_img)
                supervise_text.append(temp_text)
                supervise_label.append(temp_label)
            unsupervise_img = []
            unsupervise_text = []
            unsupervise_label = []
            for i in range(index*self.pro2[1],(index+1)*self.pro2[1]):
                temp_img = Image.open(self.unsuperviseimgrecordlist[i]).convert('RGB').resize((256, 256))
                temp_text = self.unsupervisetextlist[i]
                temp_label = self.unsuperviselabellist[i]               
                temp_img = transforms.ToTensor()(temp_img)
                temp_text = torch.FloatTensor(temp_text)
                temp_label = torch.FloatTensor(temp_label)
                unsupervise_img.append(temp_img)
                unsupervise_text.append(temp_text)
                unsupervise_label.append(temp_label)
            feature = []
            feature.append(supervise_img)
            feature.append(supervise_text)
            feature.append(unsupervise_img)
            feature.append(unsupervise_text)
            return feature, supervise_label
        elif self.train == False:
            img = Image.open(self.testimgrecordlist[index]).convert('RGB').resize((256, 256))
            text = self.testtextlist[index]
            label = self.testlabellist[index]
            img = transforms.ToTensor()(img)
            text = torch.FloatTensor(text)
            label = torch.FloatTensor(label)
            feature = []
            feature.append(img)
            feature.append(text)
            return feature, label

    def __len__(self):
        if self.train == True and self.supervise == True:
            return len(self.superviselabellist)
        elif self.train == True and self.supervise == False:
            return int(len(self.unsuperviselabellist)/self.pro2[1])
        elif self.train == False:
            return len(self.testlabellist)