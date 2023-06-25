import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import linecache
import csv
import pandas as pd
from torch.optim import Adam
from sklearn.preprocessing import StandardScaler,MinMaxScaler


class SscDataset(Dataset):
    def __init__(self, data_root, file_names):
        self.data_root = data_root
        self.file_names = file_names
        ssc_data_path = os.path.join(self.data_root, self.file_names['ssc_data'])
        data = np.loadtxt(ssc_data_path, delimiter=",", dtype=np.float32)

        day_path = os.path.join(self.data_root, self.file_names['day'])
        month_path = os.path.join(self.data_root, self.file_names['month'])
        year_path = os.path.join(self.data_root, self.file_names['year'])
        day = np.loadtxt(day_path, delimiter=",", dtype=np.int64)
        month = np.loadtxt(month_path, delimiter=",", dtype=np.int64)
        year = np.loadtxt(year_path, delimiter=",", dtype=np.int64)

        dropIndex = []
        existIndex = []
        lst = []
        scaler = MinMaxScaler()
        for i in range(len(data)):
            if np.count_nonzero(np.where(np.isnan(data[i]))) > data[i].shape[0] * 0.5:
                dropIndex.append(i)
            else:
                existIndex.append(i)
        self._existIndex = existIndex
        self._data = scaler.fit_transform(np.nan_to_num(np.delete(data, dropIndex, axis=0)))
        self._data = self._data * 2. - 1.
        # self._data=F.normalize(torch.nan_to_num(torch.from_numpy(np.delete(data,dropIndex,axis=0)), nan=0),p=2.0, dim = 1)
        self.day = np.delete(day, dropIndex)
        self.month = np.delete(month, dropIndex)
        self.year = np.delete(year, dropIndex)
        self._total_data = self._data.shape[0]

    def __getitem__(self, idx):
        return self._data[idx], self.day[idx], self.month[idx], self.year[idx]

    def __len__(self):
        return self._total_data

    def existindex(self):
        return self._existIndex

class DataContainer():

    def __init__(self, data_root=None, file_names=None, sample_indices=None, batch_size=16):

        self.data_root = data_root
        self.file_names = file_names
        self.sample_indices = sample_indices
        self.batch_size = batch_size


        latitude_path = os.path.join(self.data_root, self.file_names['latitude'])
        longitude_path = os.path.join(self.data_root, self.file_names['longitude'])
        df_latitude = pd.read_csv(latitude_path, sep=",", header=None)
        df_longitude = pd.read_csv(longitude_path, sep=",", header=None)
        latitude = df_latitude.iloc[0].to_numpy()
        longitude = df_longitude.iloc[:, 0].to_numpy()

        self.sLongIndex, self.sLong = self.getExactIndex(176.9, longitude)
        self.eLongIndex, self.eLong = self.getExactIndex(177.25, longitude)
        self.sLatiIndex, self.sLati = self.getExactIndex(-39.5, latitude)
        self.eLatiIndex, self.eLati = self.getExactIndex(-39.75, latitude)

        self.dataset = SscDataset(self.data_root, self.file_names)

    def getExactIndex(self, value, array):
        for i, j in enumerate(array):
            if (value > 0 and j > value) or (value < 0 and j < value):
                return i, j

    def getLoader(self):
        train_data = torch.utils.data.Subset(self.dataset, range(*self.sample_indices['train']))
        val_data = torch.utils.data.Subset(self.dataset, range(*self.sample_indices['val']))
        test_data = torch.utils.data.Subset(self.dataset, range(*self.sample_indices['test']))

        train_loader = DataLoader(train_data, shuffle=False, batch_size=self.batch_size,
                                  collate_fn=self.collate)
        val_loader = DataLoader(val_data, shuffle=False, batch_size=self.batch_size,
                                collate_fn=self.collate)
        test_loader = DataLoader(test_data, shuffle=False, batch_size=self.batch_size,
                                 collate_fn=self.collate)
        return train_loader, val_loader, test_loader

    def getLocation(self):
        return self.sLong, self.eLong, self.sLati, self.eLati

    def collate(self, batch):
        # Add channel dim, scale pixels between 0 and 1, send to GPU
        resizedBatch = []
        temArr = []
        for i in range(1, len(batch) + 1 - 7):
            for j in range(8):
                ssc, _, _, _ = batch[i - 1 + j]
                temArr.append(ssc.reshape(self.eLatiIndex - self.sLatiIndex,
                                          self.eLongIndex - self.sLongIndex))
            resizedBatch.append(temArr)
            temArr = []
        batch = torch.tensor(np.array(resizedBatch)).unsqueeze(1)
        return batch[:, :, 0:7], batch[:, :, 7]

    def collate_test(self, batch):

        # Add channel dim, scale pixels between 0 and 1, send to GPU
        resizedBatch = []
        temArr = []
        for i in range(1, len(batch) + 1 - 7):
            for j in range(8):
                ssc, _, _, _ = batch[i - 1 + j]
                temArr.append(ssc.reshape(self.eLatiIndex - self.sLatiIndex,
                                          self.eLongIndex - self.sLongIndex))
            resizedBatch.append(temArr)
            temArr = []

        batch = torch.tensor(resizedBatch).unsqueeze(1)

        return batch[:, :, 0:7], batch[:, :, 7]

