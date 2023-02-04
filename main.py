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


if __name__ == '__main__':

    batch_size = 4

    dataset = dset.ImageFolder(root="D:\PlantDisease\PlantVillage-Dataset/raw\color/",
                               transform=transforms.Compose([
                                   transforms.Resize(128),
                                   transforms.CenterCrop(128),
                                   transforms.ToTensor(),
                                   # transforms.Normalize((0.4669, 0.4896, 0.4108),
                                   #                      (0.1659, 0.1381, 0.1834)),
                               ])
                               )
    # dataloader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=True, num_workers=4)


    proportions = [.70, .15, .15]
    lengths = [int(p * len(dataset)) for p in proportions]
    lengths[-1] = len(dataset) - sum(lengths[:-1])
    train_set, val_set, test_set = data.random_split(dataset, lengths)

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    class_names = dataset.class_to_idx

    classes = []
    for k,v in class_names.items():
        classes.append(k)

    # show first batch
    dataiter = iter(train_dataloader)
    images, labels = next(dataiter)
    title = '  ,  '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size))
    imshow(make_grid(images),title)
    # print labels


    # for i, data in enumerate(dataloader):
    #     print(data[0].size())  # input image
    #     print(data[1])

    print('wait here')
