from torch.utils.data import Dataset
import torch
import numpy as np


class ModularDataset(Dataset):
    def __init__(self, labels, dataset, ids, providers):
        """
        :param labels:
        :param dataset: (latitude, longitude)
        :param ids:
        :param providers:
        """
        self.labels = labels
        self.dataset = dataset
        self.ids = ids
        self.providers = providers

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        latitude = self.dataset[idx][0]
        longitude = self.dataset[idx][1]
        id_ = str(self.ids[idx])

        item = (id_, latitude, longitude)

        list_tensor = []

        for provider in self.providers:
            list_tensor.append(provider[item])

        tensor = np.concatenate(list_tensor).astype(float)
        # print("tensor:\n", tensor)

        return torch.from_numpy(tensor).float(), self.labels[idx]

    def numpy(self):
        """
        :return: a numpy dataset of 1D vectors
        """
        return np.array([torch.flatten(self[i][0]).numpy() for i in range(len(self))])
