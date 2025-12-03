import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import os
import torch

from ts_augmentation.augmentation_time import *
from tsaug import Quantize, Dropout, Pool
from ts_augmentation.augmentation_frequency import *


def get_freq_augmentation(torch_train_dataset, method_index):
    if method_index == 1:
        return lowpass_filter_batch(sample=torch_train_dataset, relative_cutoff=0.4)  ##  [0.01, 0.05, 0.2, 0.3, 0.4, 0.5]
    if method_index == 2:
        return ifft_phase_shift_1d(sample=torch_train_dataset)
    if method_index == 3:
        return gen_new_aug(sample=torch_train_dataset, alpha=0.5)
    if method_index == 4:
        eps = 1e-6
        train_fft = torch.fft.rfft(torch_train_dataset, dim=-1)
        amp = torch.sqrt((train_fft.real + eps).pow(2) + (train_fft.imag + eps).pow(2))
        phase = torch.atan2(train_fft.imag, train_fft.real + eps)

        amp = amp.numpy()
        phase = phase.numpy()
        
        Dropout_amp = Dropout(p=0.1).augment(amp)  ##  p is in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] default is 0.05
        Dropout_phase = Dropout(p=0.1).augment(phase)  ##  p is in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] default is 0.05

        torch_amp = torch.from_numpy(Dropout_amp)
        torch_phase = torch.from_numpy(Dropout_phase)

        aug_set_freq = torch.stack((torch_amp, torch_phase), -1)
        aug_set_freq = aug_set_freq.permute(0, 2, 1)

        return aug_set_freq
  

def get_time_augmentation(train_dataset, method_index):
    if method_index == 1:
        return Pool(size=2).augment(train_dataset)
    if method_index == 2:
        return time_warp_batch(x=train_dataset, sigma=0.1)  ## has hyp sigma=0.2, knot=4  sigma= [0.05, 0.1, 0.3, 0.4, 0.5]
    if method_index == 3:
        return magnitude_warp_batch(x=train_dataset, sigma=0.05)  ## has hyp sigma=0.2, knot=4  sigma= [0.05, 0.1, 0.3, 0.4, 0.5]
    if method_index == 4:
        return window_slice_batch(x=train_dataset)  ## has hyp reduce_ratio=0.9  reduce_ratio = [0.1, 0.3, 0.5, 0.6, 0.7, 0.8]
    if method_index == 5:
        return Dropout(p=0.1).augment(train_dataset)
    if method_index == 6:
        return Quantize(n_levels=100).augment(train_dataset)