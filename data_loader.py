from __future__ import print_function, division

import pandas as pd
import os
from torch.utils.data import Dataset
import h5py
from sklearn import preprocessing
import numpy as np
from torch.utils.data import  DataLoader
import torch

class Whole_Slide_Bag(Dataset):
    def __init__(self,
                 train_bags,
                 label_file,
                 dataset='camelyon'
                 ):
        """
        Args:
            file_path (string): Path to the .h5 file containing patched data.
            pretrained (bool): Use ImageNet transforms
            custom_transforms (callable, optional): Optional transform to be applied on a sample
        """

        self.dataset= dataset
        self.label_file=label_file

        self.references = pd.read_csv(self.label_file)
        self.filenames = train_bags

    def __len__(self):
        return int(np.floor(len(self.filenames)))

    def __getitem__(self, idx):
        with h5py.File(self.filenames[idx], 'r') as hdf5_file:
            features = hdf5_file['features'][:]
            neighbor_indices = hdf5_file['indices'][:]
            values = hdf5_file['similarities_0'][:]

        values = values[:, 1:]
        normalized_matrix = preprocessing.normalize(values, norm="l2")
        similarities = np.exp(-normalized_matrix)

        values = np.concatenate((np.max(similarities, axis=1).reshape(-1, 1), similarities), axis=1)

        base_name = os.path.splitext(os.path.basename(self.filenames[idx]))[0]

        bag_label = self.references["slide_label"].loc[self.references["slide_id"] == base_name].values.tolist()[0]

        return torch.from_numpy(features),torch.from_numpy(neighbor_indices),torch.from_numpy(values), torch.unsqueeze(torch.tensor(bag_label).float(), dim=0)








