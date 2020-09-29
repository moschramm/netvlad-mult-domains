"""
A module used for building a dataset of day images only.

This implementation is adopted from: https://github.com/Nanne/pytorch-NetVlad
The datasets are using images from the Oxford Robotcar Dataset
(https://robotcar-dataset.robots.ox.ac.uk/).
The image folders for the datasets should be placed in `root_dir`. For more
information on which images to use see 'README.md'.
"""

import torch
import torchvision.transforms as transforms
import torch.utils.data as data

from os.path import join, exists
from scipy.io import loadmat
import numpy as np
from random import randint, random
from collections import namedtuple
from PIL import Image

from sklearn.neighbors import NearestNeighbors
import h5py

load_size = 286 # size to scale images to
nDatasetSteps = 3 # number of sequences in train set
root_dir = './Robotcar/'
if not exists(root_dir):
    raise FileNotFoundError('root_dir is not found, please adjust to point to '
                            'Oxford Robotcar dataset')

struct_dir = join(root_dir, 'datasets/')
queries_dir = root_dir

def input_transform(arch):
    if arch == 'todaygan':
        # transform for ToDayGAN-NetVLAD model
        transform_list = [transforms.Resize(load_size, Image.BICUBIC),
                          transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]
    else:
        # transform for VGG16-NetVLAD model
        transform_list = [transforms.Resize(load_size, Image.BICUBIC),
                          transforms.ToTensor(),
                          transforms.Normalize((0.485, 0.456, 0.406),
                                               (0.229, 0.224, 0.225))]
    return transforms.Compose(transform_list)

def get_whole_training_set(arch, onlyDB=False):
    structFile = [join(
    struct_dir,
    'robotcar_train_day_' + str(set_id) \
    + '.mat') for set_id in range(nDatasetSteps)]
    return [WholeDatasetFromStruct(file,
                             input_transform=input_transform(arch),
                             onlyDB=onlyDB) for file in structFile]

def get_whole_val_set(arch):
    structFile = join(struct_dir, 'robotcar_val.mat')
    return WholeDatasetFromStruct(structFile,
                             input_transform=input_transform(arch))

def get_whole_test_set(arch):
    structFile = join(struct_dir, 'robotcar_test.mat')
    return WholeDatasetFromStruct(structFile,
                             input_transform=input_transform(arch))

def get_test_hard_set(arch):
    structFile = join(struct_dir, 'robotcar_test_hard.mat')
    return WholeDatasetFromStruct(structFile,
                             input_transform=input_transform(arch))

def get_training_query_set(arch, margin=0.1):
    structFile = [join(
    struct_dir,
    'robotcar_train_day_' + str(set_id) \
    + '.mat') for set_id in range(nDatasetSteps)]
    return [QueryDatasetFromStruct(
        file, input_transform=input_transform(arch),
        margin=margin) for file in structFile]

def get_val_query_set(arch):
    structFile = join(struct_dir, 'robotcar_val.mat')
    return QueryDatasetFromStruct(structFile,
                             input_transform=input_transform(arch))

dbStruct = namedtuple('dbStruct', ['whichSet', 'dataset',
    'dbImage', 'qImage', 'numDb', 'numQ', 'gtHard', 'gtSoft'])

def parse_dbStruct(Path):
    mat = loadmat(Path)
    matStruct = mat['dbStruct'].item()

    dataset = 'robotcar'

    whichSet = matStruct[0].item()

    dbImage = [f[0].item() for f in matStruct[1]]
    qImage = [f[0].item() for f in matStruct[2]]

    numDb = matStruct[3].item()
    numQ = matStruct[4].item()

    gtHard = matStruct[5]
    gtSoft = matStruct[6]

    return dbStruct(whichSet, dataset, dbImage, qImage,
        numDb, numQ, gtHard, gtSoft)

class WholeDatasetFromStruct(data.Dataset):
    def __init__(self, structFile, input_transform=None, onlyDB=False):
        super().__init__()

        self.input_transform = input_transform

        self.dbStruct = parse_dbStruct(structFile)
        self.images = [join(root_dir, dbIm) for dbIm in self.dbStruct.dbImage]
        if not onlyDB:
            self.images += \
                [join(queries_dir, qIm) for qIm in self.dbStruct.qImage]

        self.whichSet = self.dbStruct.whichSet
        self.dataset = self.dbStruct.dataset

        self.positives = None
        # self.distances = None

    def __getitem__(self, index):
        img = Image.open(self.images[index])

        if self.input_transform:
            img = self.input_transform(img)

        return img, index

    def __len__(self):
        return len(self.images)

    def getPositives(self):
        # positives for evaluation are those within trivial threshold range
        # get them with the GTsoft matrix
        if self.positives is None:
            idx = np.arange(self.dbStruct.gtSoft.shape[0])
            self.positives = [] # array of arrays with indeces of positives
            for i in range(self.dbStruct.gtSoft.shape[1]):
                self.positives.append(idx[self.dbStruct.gtSoft[:, i].nonzero()])
            self.positives = np.asarray(self.positives)

        return self.positives

def collate_fn(batch):
    """Creates mini-batch tensors from the list of tuples
    (query, positive, negatives).

    Args:
        data: list of tuple (query, positive, negatives).
            - query: torch tensor of shape (3, h, w).
            - positive: torch tensor of shape (3, h, w).
            - negative: torch tensor of shape (n, 3, h, w).
    Returns:
        query: torch tensor of shape (batch_size, 3, h, w).
        positive: torch tensor of shape (batch_size, 3, h, w).
        negatives: torch tensor of shape (batch_size, n, 3, h, w).
    """

    batch = list(filter (lambda x:x is not None, batch))
    if len(batch) == 0: return None, None, None, None, None

    query, positive, negatives, indices = zip(*batch)

    query = data.dataloader.default_collate(query)
    positive = data.dataloader.default_collate(positive)
    negCounts = data.dataloader.default_collate([x.shape[0] for x in negatives])
    negatives = torch.cat(negatives, 0)
    import itertools
    indices = list(itertools.chain(*indices))

    return query, positive, negatives, negCounts, indices

class QueryDatasetFromStruct(data.Dataset):
    def __init__(self, structFile, nNegSample=1000, nNeg=10, margin=0.1,
                 input_transform=None):
        super().__init__()

        self.input_transform = input_transform
        self.margin = margin

        self.dbStruct = parse_dbStruct(structFile)
        self.whichSet = self.dbStruct.whichSet
        self.dataset = self.dbStruct.dataset
        self.nNegSample = nNegSample # number of negatives to randomly sample
        self.nNeg = nNeg # number of negatives used for training

        self.gtHard =  self.dbStruct.gtHard
        self.gtSoft = self.dbStruct.gtSoft

        # potential positives are those within nontrivial threshold range
        idx = np.arange(self.gtHard.shape[0])
        self.nontrivial_positives = [] # array of indices
        for i in range(self.gtHard.shape[1]):
            self.nontrivial_positives.append(idx[self.gtHard[:, i].nonzero()])
        self.nontrivial_positives = np.asarray(self.nontrivial_positives)

        # it is possible some queries don't have any non trivial potential
        # positives, let's filter those out
        self.queries = np.where(
            np.array([len(x) for x in self.nontrivial_positives])>0)[0]

        idx = np.arange(self.gtSoft.shape[0])
        potential_positives = [] # array of indices
        for i in range(self.gtSoft.shape[1]):
            potential_positives.append(idx[self.gtSoft[:, i].nonzero()])
        potential_positives = np.asarray(potential_positives)

        # potential negatives are those outside of posDistThr range
        self.potential_negatives = [] # array of indices
        for pos in potential_positives:
            self.potential_negatives.append(np.setdiff1d(
                np.arange(self.dbStruct.numDb), pos, assume_unique=True))

        # filepath of HDF5 containing feature vectors for images
        self.cache = None

        self.negCache = [np.empty((0,)) for _ in range(self.dbStruct.numQ)]

    def __getitem__(self, index):
        index = self.queries[index] # re-map index to match dataset
        with h5py.File(self.cache, mode='r') as h5:
            h5feat = h5.get("features")

            qOffset = self.dbStruct.numDb
            qFeat = h5feat[index+qOffset]

            posFeat = h5feat[self.nontrivial_positives[index].tolist()]
            knn = NearestNeighbors(n_jobs=-1)
            knn.fit(posFeat)

            dPos, posNN = knn.kneighbors(qFeat.reshape(1,-1), 1)
            dPos = dPos.item()

            posIndex = self.nontrivial_positives[index][posNN[0]].item()

            negSample = np.random.choice(
                self.potential_negatives[index], self.nNegSample)
            negSample = np.unique(
                np.concatenate([self.negCache[index], negSample]))

            negFeat = h5feat[negSample.tolist()]
            knn.fit(negFeat)

            # to quote netvlad paper code: 10x is hacky but fine
            dNeg, negNN = knn.kneighbors(qFeat.reshape(1,-1), self.nNeg*10)
            dNeg = dNeg.reshape(-1)
            negNN = negNN.reshape(-1)

            # try to find negatives that are within margin
            # if there aren't any return none
            violatingNeg = dNeg < dPos + self.margin**0.5

            if np.sum(violatingNeg) < 1:
                #if none are violating then skip this query
                return None

            negNN = negNN[violatingNeg][:self.nNeg]
            negIndices = negSample[negNN].astype(np.int32)
            self.negCache[index] = negIndices

        query = Image.open(
            join(queries_dir, self.dbStruct.qImage[index])).convert('RGB')
        positive = Image.open(
            join(root_dir, self.dbStruct.dbImage[posIndex])).convert('RGB')

        if self.input_transform:
            query = self.input_transform(query)
            positive = self.input_transform(positive)

        negatives = []
        for negIndex in negIndices:
            negative = Image.open(
                join(root_dir, self.dbStruct.dbImage[negIndex])).convert('RGB')
            if self.input_transform:
                negative = self.input_transform(negative)
            negatives.append(negative)

        negatives = torch.stack(negatives, 0)

        return query, positive, negatives, [index, posIndex]+negIndices.tolist()

    def __len__(self):
        return len(self.queries)
