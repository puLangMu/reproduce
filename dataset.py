import os
import sys
sys.path.append(".")
import glob
import time
import math
import random
import pickle

import numpy as np 
import cv2
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

class DataLitho(torch.utils.data.Dataset): 
    def __init__(self, filesMask, filesSource, filesResist, crop=False, size=(1024, 1024), cache=False):
        super().__init__()
        
        assert len(filesMask) == len(filesSource) == len(filesResist), f"WRONG SIZE: {len(filesMask)}/{len(filesSource)}/{len(filesResist)}"

        self._filesMask, self._filesSource, self._filesResist = filesMask, filesSource, filesResist
        self._crop = crop
        self._size = size
        self._cache = cache
    
        self._imagesMask = []
        self._imagesSource = []
        self._imagesResist = []
        if self._cache: 
            print(f"Pre-loading the mask images")
            for filename in tqdm(self._filesMask): 
                self._imagesMask.append(self._loadImage(filename))
            print(f"Pre-loading the Source images")
            for filename in tqdm(self._filesSource): 
                self._imagesSource.append(self._loadImage(filename))
            print(f"Pre-loading the resist images")
            for filename in tqdm(self._filesResist): 
                self._imagesResist.append(self._loadImage(filename))

    def __getitem__(self, index): 
        mask, Source, resist = self._loadMask(index), self._loadSource(index), self._loadResist(index)
        mask = mask[None, :, :]
        Source = Source[None, :, :]
        resist = resist[None, :, :]
        if self._crop: 
            padX = self._size[0] // 12
            padY = self._size[1] // 12
            startX = random.randint(0, 2*padX-1)
            startY = random.randint(0, 2*padY-1)
            mask = F.pad(mask.unsqueeze(0), (padX, padX, padY, padY))
            Source = F.pad(Source.unsqueeze(0), (padX, padX, padY, padY))
            resist = F.pad(resist.unsqueeze(0), (padX, padX, padY, padY))
            mask = mask[0, :, startX:startX+self._size[0], startY:startY+self._size[1]]
            Source = Source[0, :, startX:startX+self._size[0], startY:startY+self._size[1]]
            resist = resist[0, :, startX:startX+self._size[0], startY:startY+self._size[1]]
            if random.randint(0, 1) == 1: 
                mask = mask.flip(1)
                Source = Source.flip(1)
                resist = resist.flip(1)
            if random.randint(0, 1) == 1: 
                mask = mask.flip(2)
                Source = Source.flip(2)
                resist = resist.flip(2)
        return mask, Source, resist

    def __len__(self): 
        return len(self._filesMask)

    def _loadImage(self, filename): 
        image = cv2.imread(filename)
        if len(image.shape) > 2: 
            image = torch.tensor(image[:, :, 0], dtype=torch.float32, device="cpu") / 255.0
        image = F.interpolate(image[None, None, :, :], self._size)[0, 0]
        return image
    
    def _loadMask(self, index): 
        if self._cache: 
            return self._imagesMask[index]
        else: 
            return self._loadImage(self._filesMask[index])
    
    def _loadSource(self, index): 
        if self._cache: 
            return self._imagesSource[index]
        else: 
            # 加载 Source 图像
            image = self._loadImage(self._filesSource[index])
            # # 调整大小为 (256, 256)
            # image = F.interpolate(image[None, None, :, :], (256, 256), mode="bilinear", align_corners=False)[0, 0]
            return image

    def _loadResist(self, index): 
        if self._cache: 
            return self._imagesResist[index]
        else: 
            return self._loadImage(self._filesResist[index])


def filesLithoSim(folder, binarized=True): 
    folderMask = os.path.join(folder, "Mask")
    folderSource = os.path.join(folder, "Source")
    folderResist = os.path.join(folder, "Resist") 
    filesMask = glob.glob(folderMask + "/*.png")
    filesSource = glob.glob(folderSource + "/*.png")
    filesResist = glob.glob(folderResist + "/*.png")
    basefunc = lambda x: os.path.basename(x)[:-4]
    setMask = set(map(basefunc, filesMask))
    setSource = set(map(basefunc, filesSource))
    setResist = set(map(basefunc, filesResist))
    basenames = setMask & setSource & setResist
    filesMask = list(filter(lambda x: basefunc(x) in basenames, filesMask))
    filesSource = list(filter(lambda x: basefunc(x) in basenames, filesSource))
    filesResist = list(filter(lambda x: basefunc(x) in basenames, filesResist))
    filesMask = sorted(filesMask, key=basefunc)
    filesSource = sorted(filesSource, key=basefunc)
    filesResist = sorted(filesResist, key=basefunc)

    return filesMask, filesSource, filesResist


def lithosim(basedir, sizeImage=(512, 512), ratioTrain=0.9, cache=False): 
    filesMask, filesSource, filesResist = filesLithoSim(basedir)
    numFiles = len(filesMask)
    numTrain = round(numFiles * ratioTrain)
    numTest  = numFiles - numTrain
    trainMask = filesMask[:numTrain]
    trainSource = filesSource[:numTrain]
    trainResist = filesResist[:numTrain]
    testMask = filesMask[numTrain:]
    testSource = filesSource[numTrain:]
    testResist = filesResist[numTrain:]
    train = DataLitho(trainMask, trainSource, trainResist, crop=True, size=sizeImage, cache=cache)
    test = DataLitho(testMask, testSource, testResist, crop=False, size=sizeImage, cache=cache)
    print(f"Training set: {numTrain}, Test set: {numTest}")

    return train, test


def loadersLitho(benchmark, image_size, batch_size, njobs, drop_last=False): 
    trainset, valset = lithosim(benchmark, sizeImage=image_size, ratioTrain=0.9, cache=False)
    train_loader = DataLoader(dataset=trainset, batch_size=batch_size, num_workers=njobs, shuffle=True, drop_last=drop_last)
    val_loader = DataLoader(dataset=valset, batch_size=batch_size, num_workers=njobs, shuffle=False, drop_last=False)
    return train_loader, val_loader