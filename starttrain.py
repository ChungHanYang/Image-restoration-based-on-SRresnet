from train_utils import Train
from srresnet import _NetG
import torch.nn as nn
import torch
import os

pre_model = _NetG()
pre_model_dir = './pretrained'
pre_model_list = os.listdir(pre_model_dir)
print(pre_model_dir + '/%s' % pre_model_list[-1])
pre_model = torch.load(pre_model_dir + '/%s' % pre_model_list[-1])

pre_model.upscale4x = nn.Dropout(p=0.0)

params = {
    'noise_model': ('multiplicative_bernoulli', 0.8),
    'crop_size': 64,
    'clean_targs': False,
    'lr': 0.005,
    'epochs': 50,
    'bs': 32,
    'lossfn': 'l2',
    'cuda': True
}

trainer = Train(pre_model, './input/dataset/train/', './input/dataset/valid/', './model_bernoulli/', params)
trainer.train()

