import pytorch_lightning as pl
from torch.utils.data import DataLoader
import pickle
import os
from torch.utils.data import Dataset
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.utils.data as data
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torchvision.models import mobilenet_v3_small
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class PLDataloader(pl.LightningDataModule):
    def __init__(self,root,proportions = [0.7,0.15,0.15] ,batch_size=4,num_workers=2,shuffle=True):
        super(PLDataloader, self).__init__()
        self.root = root
        self.proportions = proportions
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.setup()

    def setup(self, stage=None):
        dataset = dset.ImageFolder(root=self.root, transform=transforms.Compose([
            transforms.Resize(128),
            transforms.CenterCrop(128),
            transforms.ToTensor(),
            transforms.Normalize((0.4669, 0.4896, 0.4108),
                                 (0.1659, 0.1381, 0.1834)),
        ]))
        class_names = dataset.class_to_idx
        self.classes = []
        for k, v in class_names.items():
            self.classes.append(k)
        lengths = [int(p * len(dataset)) for p in self.proportions]
        lengths[-1] = len(dataset) - sum(lengths[:-1])
        self.train_set, self.val_set, self.test_set = data.random_split(dataset, lengths)
        print('train-set :',len(self.train_set),'val-set :',len(self.val_set),'test-set :',len(self.test_set))


    def train_dataloader(self):
        dataloader = DataLoader(self.train_set,batch_size=self.batch_size,
                                     shuffle=self.shuffle,num_workers=self.num_workers,
                                     pin_memory=False,drop_last=True)
        return dataloader

    def val_dataloader(self):
        dataloader = DataLoader(self.val_set, batch_size=self.batch_size,
                                      shuffle=self.shuffle, num_workers=self.num_workers,
                                      pin_memory=False, drop_last=True)
        return dataloader

    def test_dataloader(self):
        dataloader = DataLoader(self.test_set, batch_size=self.batch_size,
                                      shuffle=self.shuffle, num_workers=self.num_workers,
                                      pin_memory=False, drop_last=True)
        return dataloader