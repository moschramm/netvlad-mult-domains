"""
The main script for training and evaluating NetVLAD models.

This implementation is based on: https://github.com/Nanne/pytorch-NetVlad
It is adopted to work with custom NetVLAD architectures and custom RobotCar
datasets based on the Oxford Robotcar Dataset
(https://robotcar-dataset.robots.ox.ac.uk/).
The training and evaluation is completely controllable via commandline
arguments.
"""

from __future__ import print_function
import argparse
from math import log10, ceil
import random, shutil, json
from os.path import join, exists, isfile, realpath, dirname
from os import makedirs, remove, chdir, environ, system
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.data.dataset import Subset
import torchvision.transforms as transforms
from PIL import Image
from datetime import datetime
import torchvision.datasets as datasets
import torchvision.models as models
import h5py
import faiss
from tensorboardX import SummaryWriter
from scipy.io import savemat
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from netvlad import NetVLAD
from encoder import Encoder, DualModel, DualModelShared, AlexNet, VGG16
from eval_metrics import precision_recall

parser = argparse.ArgumentParser(description='NetVLAD model for different input domains')
parser.add_argument('--mode', type=str, default='train', help='Mode', choices=['train', 'test', 'cluster'])
parser.add_argument('--name', type=str, default='experiment_' + datetime.now().strftime('%b%d_%H-%M-%S'),
                    help='Name of experiment, used for saving the model.')
parser.add_argument('--batchSize', type=int, default=4,
                    help='Number of triplets (query, pos, negs). Each triplet consists of 12 images.')
parser.add_argument('--cacheBatchSize', type=int, default=72, help='Batch size for caching and testing')
parser.add_argument('--cacheRefreshRate', type=int, default=1000,
                    help='How often to refresh cache, in number of queries. 0 for off')
parser.add_argument('--nEpochs', type=int, default=10, help='Number of epochs to train for')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--nGPU', type=int, default=1, help='number of GPU to use.')
parser.add_argument('--optim', type=str, default='SGD', help='optimizer to use', choices=['SGD', 'ADAM'])
parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate.')
parser.add_argument('--lrStep', type=float, default=5, help='Decay LR every N steps.')
parser.add_argument('--lrGamma', type=float, default=0.5, help='Multiply LR by Gamma for decaying.')
parser.add_argument('--weightDecay', type=float, default=0.001, help='Weight decay for SGD.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD.')
parser.add_argument('--nocuda', action='store_true', help='Dont use cuda')
parser.add_argument('--threads', type=int, default=8, help='Number of threads for each data loader to use')
parser.add_argument('--seed', type=int, default=123, help='Random seed to use.')
parser.add_argument('--dataPath', type=str, default='/home/moritz/BA/netvlad/data/', help='Path for centroid data.')
parser.add_argument('--runsPath', type=str, default='/home/moritz/BA/netvlad/runs/', help='Path to save runs to.')
parser.add_argument('--savePath', type=str, default='checkpoints',
                    help='Path to save checkpoints to in logdir. Default=checkpoints/')
parser.add_argument('--cachePath', type=str, default='/tmp/pyvlad_cache', help='Path to save cache to.')
parser.add_argument('--resume', type=str, default='', help='Path to load checkpoint from, for resuming training or testing.')
parser.add_argument('--ckpt', type=str, default='latest',
                    help='Resume from latest or best checkpoint.', choices=['latest', 'best'])
parser.add_argument('--evalEvery', type=int, default=1,
                    help='Do a validation set run, and save, every N epochs.')
parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping. 0 is off.')
parser.add_argument('--dataset', type=str, default='robotcar',
                    help='Dataset to use', choices=['pittsburgh', 'robotcar',
                    'robotcar_day', 'robotcar_synthetic', 'robotcar_db_gtmat', 'robotcar_db_nogt'])
parser.add_argument('--arch', type=str, default='vgg16',
                    help='basenetwork to use', choices=['vgg16', 'alexnet', 'todaygan'])
parser.add_argument('--trainLayers',  nargs='+', default=['13'],
                    help='List with numbers of the layers (ResBlocks) to train.')
parser.add_argument('--encoderPath', type=str, default='/home/moritz/BA/ToDayGAN/checkpoints/night2day_l2/',
                    help='Path for ToDayGAN checkpoint.')
parser.add_argument('--cnnArch', type=str, default='single', help='Which CNN architecture to use.',
                    choices=['single', 'dual'])
parser.add_argument('--addConv', action='store_true', help='Add an additional downsampling layer at the end.')
parser.add_argument('--vladv2', action='store_true', help='Use VLAD v2')
parser.add_argument('--pooling', type=str, default='netvlad', help='type of pooling to use',
                    choices=['netvlad', 'max', 'avg'])
parser.add_argument('--num_clusters', type=int, default=64, help='Number of NetVlad clusters. Default=64')
parser.add_argument('--margin', type=float, default=0.1, help='Margin for triplet loss. Default=0.1')
parser.add_argument('--split', type=str, default='val', help='Data split to use for testing. Default is test',
                    choices=['test', 'test250k', 'train', 'val', 'test_hard'])
parser.add_argument('--fromscratch', action='store_true', help='Train from scratch rather than using pretrained models')
parser.add_argument('--saveCosSim', action='store_true', help='Calculate and save cosine similarity matrix when testing')
parser.add_argument('--calcPR', action='store_true', help='Calculate precision-recall metrics when testing')

def train(epoch):
    """A function used to train the model for one epoch.

    It builds the cache every `opt.cacheRefreshRate` iterations and trains the
    model via backpropagation on the training set.

    Parameters
    ----------
    epoch : int
        number of current epoch (starting at 1) of training
    """

    nDatasetSteps = 3
    print('Using train set at index: ' + str((epoch-1) % nDatasetSteps))
    train_set = train_set_list[(epoch-1) % nDatasetSteps]
    whole_training_dl = whole_training_data_loader[(epoch-1) % nDatasetSteps]
    nDB = train_set.dbStruct.numDb
    epoch_loss = 0
    startIter = 1 # keep track of batch iter across subsets for logging

    if opt.cacheRefreshRate > 0:
        subsetN = ceil(len(train_set) / opt.cacheRefreshRate)
        subsetIdx = np.array_split(np.arange(len(train_set)), subsetN)
    else:
        subsetN = 1
        subsetIdx = [np.arange(len(train_set))]

    nBatches = (len(train_set) + opt.batchSize - 1) // opt.batchSize

    for subIter in range(subsetN):
        print('====> Building Cache')
        model.eval()
        train_set.cache = join(
            opt.cachePath, train_set.whichSet + '_feat_cache.hdf5')
        with h5py.File(train_set.cache, mode='w') as h5:
            pool_size = encoder_dim
            if opt.pooling.lower() == 'netvlad': pool_size *= opt.num_clusters
            h5feat = h5.create_dataset(
                "features",
                [len(whole_train_set[(epoch-1) % nDatasetSteps]), pool_size],
                dtype=np.float32)
            with torch.no_grad():
                for iteration, (input, indices) \
                    in enumerate(whole_training_dl, 1):
                    input = input.to(device)
                    image_encoding = model.encoder(
                        input, opt.cacheBatchSize - indices[-1] + nDB - 1)
                    vlad_encoding = model.pool(image_encoding)
                    h5feat[indices.detach().numpy(), :] = \
                        vlad_encoding.detach().cpu().numpy()
                    del input, image_encoding, vlad_encoding

        sub_train_set = Subset(dataset=train_set, indices=subsetIdx[subIter])

        training_data_loader = DataLoader(
            dataset=sub_train_set, num_workers=opt.threads,
            batch_size=opt.batchSize, shuffle=True,
            collate_fn=dataset.collate_fn, pin_memory=cuda)

        print('Allocated:', torch.cuda.memory_allocated())
        print('Cached:', torch.cuda.memory_cached())

        model.train()
        for iteration, (query, positives, negatives, negCounts, indices) \
            in enumerate(training_data_loader, startIter):
            # some reshaping to put query, pos, negs in a single (N, 3, H, W)
            # tensor, where N = batchSize * (nQuery + nPos + nNeg)

            if query is None: continue # in case we get an empty batch

            B, C, H, W = query.shape
            nNeg = torch.sum(negCounts)
            input = torch.cat([positives, negatives, query])

            input = input.to(device)
            image_encoding = model.encoder(input, 11*B)
            vlad_encoding = model.pool(image_encoding)
            vladP, vladN, vladQ = torch.split(vlad_encoding, [B, nNeg, B])

            optimizer.zero_grad()

            # calculate loss for each Query, Positive, Negative triplet
            # due to potential difference in number of negatives have to
            # do it per query, per negative
            loss = 0
            for i, negCount in enumerate(negCounts):
                for n in range(negCount):
                    negIx = (torch.sum(negCounts[:i]) + n).item()
                    loss += criterion(
                        vladQ[i:i+1], vladP[i:i+1], vladN[negIx:negIx+1])

            loss /= nNeg.float().to(device) # normalise by actual number of
                                            # negatives
            loss.backward()
            optimizer.step()
            del input, image_encoding, vlad_encoding, vladQ, vladP, vladN
            del query, positives, negatives

            batch_loss = loss.item()
            epoch_loss += batch_loss

            if iteration % 50 == 0 or nBatches <= 10:
                print("==> Epoch[{}]({}/{}): Loss: {:.4f}".format(
                    epoch, iteration, nBatches, batch_loss), flush=True)
                writer.add_scalar('Train/Loss', batch_loss,
                                  ((epoch-1) * nBatches) + iteration)
                writer.add_scalar(
                    'Train/nNeg', nNeg, ((epoch-1) * nBatches) + iteration)
                print('Allocated:', torch.cuda.memory_allocated())
                print('Cached:', torch.cuda.memory_cached())

        startIter += len(training_data_loader)
        del training_data_loader, loss
        optimizer.zero_grad()
        torch.cuda.empty_cache()
        remove(train_set.cache) # delete HDF5 cache

    avg_loss = epoch_loss / nBatches

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, avg_loss),
            flush=True)
    writer.add_scalar('Train/AvgLoss', avg_loss, epoch)

def test(eval_set, epoch=0, write_tboard=False, cos_sim=False, calc_pr=False):
    """A function used to evaluate the model on `eval_set`.

    Used for validating and testing.

    Parameters
    ----------
    eval_set : torch.utils.data.Dataset
        dataset to use for evaluation process
    epoch: int, optional
        current epoch of training (default is 0)
    write_tboard : bool, optional
        whether or not to write recalls to tensorboard file (default is False)
    cos_sim : bool, optional
        whether or not to save the cosinus similarity matrix (default is False)
    calc_pr : bool, optional
        whether or not to calculate precision-recall-metrics (default is False)

    Returns
    -------
    dict
        a dictionary containing the top-N-recall for N = [1,5,10,20]
    """

    nDB = eval_set.dbStruct.numDb
    test_data_loader = DataLoader(
        dataset=eval_set, num_workers=opt.threads,
        batch_size=opt.cacheBatchSize, shuffle=False, pin_memory=cuda)

    model.eval()
    with torch.no_grad():
        print('====> Extracting Features')
        pool_size = encoder_dim
        if opt.pooling.lower() == 'netvlad': pool_size *= opt.num_clusters
        dbFeat = np.empty((len(eval_set), pool_size), dtype=np.float32)

        for iteration, (input, indices) in enumerate(test_data_loader, 1):
            input = input.to(device)
            image_encoding = model.encoder(
                input, opt.cacheBatchSize - indices[-1] + nDB - 1)
            vlad_encoding = model.pool(image_encoding)

            dbFeat[indices.detach().numpy(), :] = \
                vlad_encoding.detach().cpu().numpy()

            if iteration % 50 == 0 or len(test_data_loader) <= 10:
                print("==> Batch ({}/{})".format(iteration,
                    len(test_data_loader)), flush=True)

            del input, image_encoding, vlad_encoding
    del test_data_loader

    # extracted for both db and query, now split in own sets
    qFeat = dbFeat[nDB:].astype('float32')
    dbFeat = dbFeat[:nDB].astype('float32')

    if cos_sim:
        print('====> Calculating cosine similarity')
        cos_mat = cosine_similarity(dbFeat, qFeat)
        savemat(join('./test', 'cos_sim_' + str(opt.name) + '.mat'),
            {'cosSim': cos_mat})

    if calc_pr:
        print('====> Calculating precision and recall metrics')
        precision_recall(
            eval_set.dbStruct.gtSoft, eval_set.dbStruct.gtHard,
            np.matmul(dbFeat, qFeat.transpose()))

    print('====> Building faiss index')
    faiss_index = faiss.IndexFlatL2(pool_size)
    faiss_index.add(dbFeat)

    print('====> Calculating recall @ N')
    n_values = [1,5,10,20]

    # get indices of NN in db
    _, predictions = faiss_index.search(qFeat, max(n_values))

    # for each query get those within threshold distance
    gt = eval_set.getPositives()

    correct_at_n = np.zeros(len(n_values))
    for qIx, pred in enumerate(predictions):
        for i, n in enumerate(n_values):
            # if in top N then also in top NN, where NN > N
            if np.any(np.in1d(pred[:n], gt[qIx])):
                correct_at_n[i:] += 1
                break
    recall_at_n = correct_at_n / eval_set.dbStruct.numQ

    recalls = {} # make dict for output
    for i,n in enumerate(n_values):
        recalls[n] = recall_at_n[i]
        print("====> Recall@{}: {:.4f}".format(n, recall_at_n[i]))
        if write_tboard: writer.add_scalar(
            'Val/Recall@' + str(n), recall_at_n[i], epoch)

    return recalls

def get_clusters(cluster_set):
    """A function to calculate the clusters for the NetVLAD layer.

    The clustering has to be performed for every model and every dataset before
    training or testing. The cluster are saved to disk and can be reused later.

    Parameters
    ----------
    cluster_set : torch.utils.data.Dataset
        dataset used for clustering, usually the same as the training set
    """

    nDescriptors = 50000
    nPerImage = 100
    nIm = ceil(nDescriptors/nPerImage/2)

    sampler_db = SubsetRandomSampler(
        np.random.choice(cluster_set.dbStruct.numDb, nIm, replace=False))
    data_loader_db = DataLoader(
        dataset=cluster_set, num_workers=opt.threads,
        batch_size=int(opt.cacheBatchSize/2), shuffle=False, pin_memory=cuda,
        sampler=sampler_db)

    sampler_q = SubsetRandomSampler(
        np.random.choice(np.arange(cluster_set.dbStruct.numDb,
        cluster_set.dbStruct.numDb + cluster_set.dbStruct.numQ), nIm,
        replace=False))
    data_loader_q = DataLoader(
        dataset=cluster_set, num_workers=opt.threads,
        batch_size=int(opt.cacheBatchSize/2), shuffle=False, pin_memory=cuda,
        sampler=sampler_q)

    if not exists(join(opt.dataPath, 'centroids')):
        makedirs(join(opt.dataPath, 'centroids'))

    initcache = join(
        opt.dataPath, 'centroids',
        opt.arch.lower() + '_' + cluster_set.dataset + '_' \
        + str(opt.num_clusters) + '_desc_cen.hdf5')
    with h5py.File(initcache, mode='w') as h5:
        with torch.no_grad():
            model.eval()
            print('====> Extracting Descriptors')
            dbFeat = h5.create_dataset(
                "descriptors", [nDescriptors, encoder_dim], dtype=np.float32)

            for iteration, input_list in enumerate(
                zip(data_loader_db, data_loader_q), 1):
                input_db, input_q = input_list[0][0], input_list[1][0]
                input = torch.cat([input_db, input_q])
                input = input.to(device)
                image_descriptors = model.encoder(
                    input, int(opt.cacheBatchSize/2)).view(input.size(0),
                    encoder_dim, -1).permute(0, 2, 1)

                batchix = (iteration-1)*opt.cacheBatchSize*nPerImage
                for ix in range(image_descriptors.size(0)):
                    # sample different location for each image in batch
                    sample = np.random.choice(
                        image_descriptors.size(1), nPerImage, replace=False)
                    startix = batchix + ix*nPerImage
                    dbFeat[startix:startix+nPerImage, :] = \
                        image_descriptors[ix, sample, :].detach().cpu().numpy()

                if iteration % 50 == 0 or len(data_loader_db)*2 <= 10:
                    print("==> Batch ({}/{})".format(iteration,
                        ceil(nIm*2/opt.cacheBatchSize)), flush=True)
                del input, image_descriptors

        print('====> Clustering..')
        niter = 100
        kmeans = faiss.Kmeans(
            encoder_dim, opt.num_clusters, niter=niter, verbose=False)
        kmeans.train(dbFeat[...])

        print('====> Storing centroids', kmeans.centroids.shape)
        h5.create_dataset('centroids', data=kmeans.centroids)
        print('====> Done!')

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """A function used to save the model checkpoint to disk.

    Parameters
    ----------
    state : dict
        dictionary containing model state information
    is_best : bool
        indicates if current model state achives highest recalls
    filename : string, optional
        filename to use for saving checkpoint (default is 'checkpoint.pth.tar')
    """

    model_out_path = join(opt.savePath, filename)
    torch.save(state, model_out_path)
    if is_best:
        shutil.copyfile(
            model_out_path, join(opt.savePath, 'model_best.pth.tar'))

def get_lr(optimizer):
    """A function to get the currently used learning rate.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        optimizer used for training the model

    Returns
    -------
    float
        value of current learning rate of the optimizer
    """

    for param_group in optimizer.param_groups:
        return param_group['lr']

class Flatten(nn.Module):
    """Class to create a layer for flattening input but keep batch size"""
    def forward(self, input):
        return input.view(input.size(0), -1)

class L2Norm(nn.Module):
    """Class to create a layer to L2-normalize input along dimension 1"""
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return F.normalize(input, p=2, dim=self.dim)

if __name__ == "__main__":
    opt = parser.parse_args()

    restore_var = ['lr', 'lrStep', 'lrGamma', 'weightDecay', 'momentum',
                   'runsPath', 'savePath', 'arch', 'num_clusters', 'pooling',
                   'optim', 'margin', 'seed', 'patience', 'cnnArch', 'addConv']
    if opt.resume:
        flag_file = join(opt.resume, 'checkpoints', 'flags.json')
        if exists(flag_file):
            with open(flag_file, 'r') as f:
                stored_flags = {'--'+k : str(v) for k,v in json.load(f).items()
                    if k in restore_var}
                to_del = []
                for flag, val in stored_flags.items():
                    for act in parser._actions:
                        if act.dest == flag[2:]:
                            # store_true / store_false args don't accept
                            # arguments, filter these
                            if type(act.const) == type(True):
                                if val == str(act.default):
                                    to_del.append(flag)
                                else:
                                    stored_flags[flag] = ''
                for flag in to_del: del stored_flags[flag]

                train_flags = [x for x in list(sum(stored_flags.items(),
                                                   tuple())) if len(x) > 0]
                print('Restored flags:', train_flags)
                opt = parser.parse_args(train_flags, namespace=opt)

    print(opt)

    if opt.dataset.lower() == 'pittsburgh':
        import pittsburgh as dataset
    elif opt.dataset.lower() == 'robotcar':
        import robotcar_gtmat as dataset
    elif opt.dataset.lower() == 'robotcar_day':
        import robotcar_day as dataset
    elif opt.dataset.lower() == 'robotcar_synthetic':
        import robotcar_synth as dataset
    elif opt.dataset.lower() == 'robotcar_db_gtmat':
        import robotcar_db_gtmat as dataset
    elif opt.dataset.lower() == 'robotcar_db_nogt':
        import robotcar_db as dataset
    else:
        raise Exception('Unknown dataset')

    cuda = not opt.nocuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run with --nocuda")

    device = torch.device("cuda" if cuda else "cpu")

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    print('===> Loading dataset(s)')
    if opt.mode.lower() == 'train':
        # a list of train sets
        whole_train_set = dataset.get_whole_training_set(opt.arch.lower())
        whole_training_data_loader = [DataLoader(
            dataset=set, num_workers=opt.threads, batch_size=opt.cacheBatchSize,
            shuffle=False, pin_memory=cuda) for set in whole_train_set]

        # a list of train sets
        train_set_list = dataset.get_training_query_set(
            opt.arch.lower(), opt.margin)

        print('====> Training query set:', len(train_set_list[0]))
        whole_test_set = dataset.get_whole_val_set(opt.arch.lower())
        print('===> Evaluating on val set, query count:',
              whole_test_set.dbStruct.numQ)
    elif opt.mode.lower() == 'test':
        if opt.split.lower() == 'test':
            whole_test_set = dataset.get_whole_test_set(opt.arch.lower())
            print('===> Evaluating on test set')
        elif opt.split.lower() == 'test_hard':
            whole_test_set = dataset.get_test_hard_set(opt.arch.lower())
            print('===> Evaluating on test hard')
        elif opt.split.lower() == 'test250k':
            if opt.dataset.lower() == 'pittsburgh':
                whole_test_set = dataset.get_250k_test_set(opt.arch.lower())
                print('===> Evaluating on test250k set')
            else:
                raise Exception(str(opt.dataset) + ' has no test250k set')
        elif opt.split.lower() == 'train':
            whole_test_set = dataset.get_whole_training_set(opt.arch.lower())
            print('===> Evaluating on train set')
        elif opt.split.lower() == 'val':
            whole_test_set = dataset.get_whole_val_set(opt.arch.lower())
            print('===> Evaluating on val set')
        else:
            raise ValueError('Unknown dataset split: ' + opt.split)
        print('====> Query count:', whole_test_set.dbStruct.numQ)
    elif opt.mode.lower() == 'cluster':
        whole_train_set = dataset.get_whole_training_set(
            opt.arch.lower(), onlyDB=False)[0]

    pretrained = not opt.fromscratch
    add_l2 = opt.mode.lower() == 'cluster' and not opt.vladv2
    print('===> Building model')
    if opt.cnnArch.lower() == 'single':
        if opt.arch.lower() == 'alexnet':
            encoder_dim = 256
            enc_module = AlexNet(add_l2=add_l2, pretrained=pretrained)

        elif opt.arch.lower() == 'vgg16':
            encoder_dim = 512
            enc_module = VGG16(add_l2=add_l2, pretrained=pretrained)

        elif opt.arch.lower() == 'todaygan':
            encoder_dim = 256
            enc_module = Encoder(add_l2=add_l2)
            if pretrained:
                enc_module.load_state_dict(
                    torch.load(join(opt.encoderPath, '190_net_G0.pth')))
    else:
        if opt.arch.lower() == 'alexnet':
            encoder_dim = 256
        elif opt.arch.lower() == 'vgg16':
            encoder_dim = 512
        elif opt.arch.lower() == 'todaygan':
            encoder_dim = 256
        if opt.addConv:
            encoder_dim *= 2
            enc_module = DualModelShared(
                opt.arch.lower(), opt.encoderPath, add_l2=add_l2,
                pretrained=pretrained)
        else:
            enc_module = DualModel(
                opt.arch.lower(), opt.encoderPath, add_l2=add_l2,
                pretrained=pretrained)

    model = nn.Module()
    model.add_module('encoder', enc_module)


    if opt.mode.lower() != 'cluster':
        if opt.pooling.lower() == 'netvlad':
            net_vlad = NetVLAD(num_clusters=opt.num_clusters,
                                       dim=encoder_dim, vladv2=opt.vladv2)
            if not opt.resume:
                if opt.mode.lower() == 'train':
                    initcache = join(
                        opt.dataPath, 'centroids',
                        opt.arch.lower() + '_' + train_set_list[0].dataset
                        + '_' + str(opt.num_clusters) +'_desc_cen.hdf5')
                else:
                    initcache = join(
                        opt.dataPath, 'centroids',
                        opt.arch.lower() + '_' + whole_test_set.dataset \
                        + '_' + str(opt.num_clusters) +'_desc_cen.hdf5')

                if not exists(initcache):
                    raise FileNotFoundError('Could not find clusters,' \
                        + 'please run with --mode=cluster before proceeding')

                with h5py.File(initcache, mode='r') as h5:
                    clsts = h5.get("centroids")[...]
                    traindescs = h5.get("descriptors")[...]
                    net_vlad.init_attributes(clsts, traindescs)
                    del clsts, traindescs

            model.add_module('pool', net_vlad)
        elif opt.pooling.lower() == 'max':
            global_pool = nn.AdaptiveMaxPool2d((1,1))
            model.add_module(
                'pool', nn.Sequential(*[global_pool, Flatten(), L2Norm()]))
        elif opt.pooling.lower() == 'avg':
            global_pool = nn.AdaptiveAvgPool2d((1,1))
            model.add_module(
                'pool', nn.Sequential(*[global_pool, Flatten(), L2Norm()]))
        else:
            raise ValueError('Unknown pooling type: ' + opt.pooling)

    isParallel = False
    if opt.nGPU > 1 and torch.cuda.device_count() > 1:
        model.encoder = nn.DataParallel(model.encoder)
        if opt.mode.lower() != 'cluster':
            model.pool = nn.DataParallel(model.pool)
        isParallel = True

    if not opt.resume:
        model = model.to(device)
        if opt.arch.lower() == 'todaygan':
            if opt.cnnArch.lower() == 'dual':
                ['model.' + str(l) for l in opt.trainLayers]
            model.encoder.set_train_layers(opt.trainLayers)

    if opt.mode.lower() == 'train':
        if opt.optim.upper() == 'ADAM':
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=opt.lr) # betas=(0,0.9))

        elif opt.optim.upper() == 'SGD':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad,
                model.parameters()), lr=opt.lr,
                momentum=opt.momentum,
                weight_decay=opt.weightDecay)
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=opt.lrStep, gamma=opt.lrGamma)
        else:
            raise ValueError('Unknown optimizer: ' + opt.optim)

        # sqrt() the margin, because the distances are sqrt()
        criterion = nn.TripletMarginLoss(
            margin=opt.margin**0.5, p=2, reduction='sum').to(device)

    if opt.resume:
        if opt.ckpt.lower() == 'latest':
            resume_ckpt = join(opt.resume, 'checkpoints', 'checkpoint.pth.tar')
        elif opt.ckpt.lower() == 'best':
            resume_ckpt = join(opt.resume, 'checkpoints', 'model_best.pth.tar')

        if isfile(resume_ckpt):
            print("=> loading checkpoint '{}'".format(resume_ckpt))
            checkpoint = torch.load(
                resume_ckpt, map_location=lambda storage, loc: storage)
            opt.start_epoch = checkpoint['epoch']
            best_metric = checkpoint['best_score']
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            model = model.to(device)
            if opt.arch.lower() == 'todaygan':
                if opt.cnnArch.lower() == 'dual':
                    ['model.' + str(l) for l in opt.trainLayers]
                model.encoder.set_train_layers(opt.trainLayers)
            if opt.mode == 'train':
                if opt.optim.upper() == 'ADAM':
                    optimizer = optim.Adam(
                        filter(lambda p: p.requires_grad, model.parameters()),
                        lr=opt.lr) # betas=(0,0.9))

                elif opt.optim.upper() == 'SGD':
                    optimizer = optim.SGD(filter(lambda p: p.requires_grad,
                        model.parameters()), lr=opt.lr,
                        momentum=opt.momentum,
                        weight_decay=opt.weightDecay)
                    scheduler = optim.lr_scheduler.StepLR(
                        optimizer, step_size=opt.lrStep, gamma=opt.lrGamma)
                else:
                    raise ValueError('Unknown optimizer: ' + opt.optim)
                optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume_ckpt, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume_ckpt))

    if opt.mode.lower() == 'test':
        print('===> Running evaluation step')
        epoch = 1
        recalls = test(whole_test_set, epoch, write_tboard=False,
                       cos_sim=opt.saveCosSim, calc_pr=opt.calcPR)
    elif opt.mode.lower() == 'cluster':
        print('===> Calculating descriptors and clusters')
        get_clusters(whole_train_set)

    elif opt.mode.lower() == 'train':
        print('===> Training model')
        writer = SummaryWriter(log_dir=join(
            opt.runsPath,
            datetime.now().strftime('%b%d_%H-%M-%S') + '_' + opt.arch.lower() \
            + '_' + opt.pooling))

        # write checkpoints in logdir
        logdir = writer.file_writer.get_logdir()
        opt.savePath = join(logdir, opt.savePath)
        if not opt.resume:
            makedirs(opt.savePath)

        with open(join(opt.savePath, 'flags.json'), 'w') as f:
            f.write(json.dumps(
                {k:v for k,v in vars(opt).items()}
                ))
        print('===> Saving state to:', logdir)

        not_improved = 0
        best_score = 0
        for epoch in range(opt.start_epoch+1, opt.nEpochs + 1):
            print('===> Using learning rate:', get_lr(optimizer))
            writer.add_scalar('LR', get_lr(optimizer), epoch)
            train(epoch)
            if opt.optim.upper() == 'SGD':
                scheduler.step(epoch)
            if (epoch % opt.evalEvery) == 0:
                recalls = test(whole_test_set, epoch, write_tboard=True)
                is_best = recalls[5] > best_score
                if is_best:
                    not_improved = 0
                    best_score = recalls[5]
                else:
                    not_improved += 1

                save_checkpoint({
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'recalls': recalls,
                        'best_score': best_score,
                        'optimizer' : optimizer.state_dict(),
                        'parallel' : isParallel,
                }, is_best)

                if opt.patience > 0 and not_improved > \
                    (opt.patience / opt.evalEvery):
                    print('Performance did not improve for', opt.patience,
                          'epochs. Stopping.')
                    break

        print("=> Best Recall@5: {:.4f}".format(best_score), flush=True)
        writer.close()
