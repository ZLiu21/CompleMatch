import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import torch
import torch.utils.data as data


class UCRDataset(data.Dataset):
    def __init__(self, dataset, target):
        self.dataset = dataset
        if len(self.dataset.shape) == 2:
            self.dataset = torch.unsqueeze(self.dataset, 1)
        self.target = target

    def __getitem__(self, index):
        return self.dataset[index], self.target[index]

    def __len__(self):
        return len(self.target)


class UCRTFAugDataset(data.Dataset):
    def __init__(self, dataset, freq_data, data_aug1, data_aug2, freq_aug1, freq_aug2, target, mask_train=None):
        self.dataset = dataset
        self.freq_data = freq_data
        self.data_aug1 = data_aug1
        self.data_aug2 = data_aug2
        self.freq_aug1 = freq_aug1
        self.freq_aug2 = freq_aug2
        self.mask_train = mask_train
        if len(self.dataset.shape) == 2:
            self.dataset = torch.unsqueeze(self.dataset, 1)
        if len(self.data_aug1.shape) == 2:
            self.data_aug1 = torch.unsqueeze(self.data_aug1, 1)
        if len(self.data_aug2.shape) == 2:
            self.data_aug2 = torch.unsqueeze(self.data_aug2, 1)

        if len(self.freq_data.shape) == 2:
            self.freq_data = torch.unsqueeze(self.freq_data, 1)

        if len(self.freq_aug1.shape) == 2:

            self.freq_aug1 = torch.unsqueeze(self.freq_aug1, 1)

        if len(self.freq_aug2.shape) == 2:
            self.freq_aug2 = torch.unsqueeze(self.freq_aug2, 1)

        self.target = target

    def __getitem__(self, index):
        if self.mask_train is None:
            return self.dataset[index], self.freq_data[index], self.data_aug1[index], self.data_aug2[index], \
                   self.freq_aug1[index], self.freq_aug2[index], self.target[index], self.target[index]
        return self.dataset[index], self.freq_data[index], self.data_aug1[index], self.data_aug2[index], \
               self.freq_aug1[index], self.freq_aug2[index], self.target[index], self.mask_train[index]

    def __len__(self):
        return len(self.target)


if __name__ == '__main__':
    pass
