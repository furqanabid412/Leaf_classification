import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

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
from dataloader import PLDataloader
import warnings
warnings.filterwarnings('ignore')
import os
from datetime import datetime
import yaml
from trainer import PLModel
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

class TQDM_Bar(pl.callbacks.TQDMProgressBar):
    def on_validation_start(self, trainer, pl_module):
        pass
    def on_validation_batch_start( self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        if not self.has_dataloader_changed(dataloader_idx):
            return
def mean_std_calc(dataloader):
    dataloader_size = len(dataloader.dataset)
    mean = 0.
    std = 0.
    for images, _ in tqdm(dataloader):
        batch_samples = images.size(0)  # batch size (the last batch can have smaller size!)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)

    mean /= dataloader_size
    std /= dataloader_size

    print('mean',mean,'std',std)

def imshow(img,title):
    # img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.show()


# def show_batch(images,labels,classes):


if __name__ == '__main__':

    dataset = PLDataloader(root="D:\PlantDisease\PlantVillage-Dataset/raw\color/",proportions=[0.7,0.15,0.15],batch_size=32,num_workers=2,shuffle=True)
    model = PLModel(classes=dataset.classes)
    bar = TQDM_Bar()
    checkpoint = pl.callbacks.ModelCheckpoint(dirpath='D:\PlantDisease\classification\saved',verbose=True,
                                              every_n_epochs=1, save_last=True,monitor='val_epoch_accu', mode='max')
    wandb = WandbLogger(project='leaf_classification')
    trainer = pl.Trainer(gpus=-1, auto_scale_batch_size=False, enable_checkpointing=True,precision=32,
                             logger=wandb, callbacks=[checkpoint, bar], max_epochs=1000,strategy=None)

    trainer.fit(model,dataset)
    # print('wait here')



# batch_size = 8
# epochs = 100
# dataset = dset.ImageFolder(root="D:\PlantDisease\PlantVillage-Dataset/raw\color/",
#                                transform=transforms.Compose([
#                                    transforms.Resize(128),
#                                    transforms.CenterCrop(128),
#                                    transforms.ToTensor(),
#                                    # transforms.Normalize((0.4669, 0.4896, 0.4108),
#                                    #                      (0.1659, 0.1381, 0.1834)),
#                                ])
#                                )
# # dataloader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=True, num_workers=4)
# proportions = [.70, .15, .15]
# lengths = [int(p * len(dataset)) for p in proportions]
# lengths[-1] = len(dataset) - sum(lengths[:-1])
# train_set, val_set, test_set = data.random_split(dataset, lengths)
#
# train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
# class_names = dataset.class_to_idx
#
# classes = []
# for k,v in class_names.items():
#     classes.append(k)
#
# model = mobilenet_v3_small(weights='IMAGENET1K_V1')
# model.classifier[3] = nn.Linear(1024, len(classes))
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
#
# # show first batch
# '''
# dataiter = iter(train_dataloader)
# images, labels = next(dataiter)
# title = '  ,  '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size))
# imshow(make_grid(images),title)
# # print labels
# '''
#
# for ep in range(epochs):
#     running_loss = 0.0
#     for i, data in enumerate(train_dataloader):
#         inputs, labels = data
#         # zero the parameter gradients
#         optimizer.zero_grad()
#         # forward + backward + optimize
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#
#         # print statistics
#         running_loss += loss.item()
#         if i % 2 == 1999:  # print every 2 mini-batches
#             print(f'[{ep + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
#             running_loss = 0.0