from data import NoisyDataset
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Tkagg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from srresnet import _NetG
import os


#load model
model = _NetG().cuda()
model_root = './model_bernoulli'
model_list = os.listdir(model_root)
model = torch.load(model_root + '/%s' % model_list[-1])
print(model_root + '/%s' % model_list[-1])
model.upscale4x = nn.Dropout(p=0.0)

data_ = NoisyDataset('./input/dataset/test/', crop_size=128, clean_targ=True, train_noise_model=('multiplicative_bernoulli', 0.5)) # Default gaussian noise without clean targets
dl = DataLoader(data_, batch_size=1, shuffle=True)


def show(img, a):
    plt.figure()
    if a == 'v':
        npimg = torch.squeeze(img).data.cpu().numpy()
    else:
        npimg = torch.squeeze(img).numpy()

    img_np = np.transpose(npimg, (1, 2, 0))

    plt.imshow(img_np)
    plt.show()


i = 0
for _list in dl:
    pred = model(Variable(_list[0].cuda()))
    show(pred, 'v')
    show(_list[0], 't')
    show(_list[-1], 't')
    if i == 0:
        break

