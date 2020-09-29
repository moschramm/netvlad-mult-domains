"""
A script to evaluate encoder pairs of a ToDayGAN model.

The checkpoints to use for the encoders need to be specified. Futhermore
the similarity measures to use can be specified via `--sim_measures`.
This implementation reuses some code from:
https://github.com/Nanne/pytorch-NetVlad
"""

import torch
import robotcar_gtmat as dataset
import argparse
import faiss
import numpy as np
from torch.utils.data import DataLoader
from datetime import datetime
from os.path import join, exists
from scipy import sparse
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
from sklearn.random_projection import SparseRandomProjection
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import manhattan_distances
from eval_metrics import precision_recall
import pandas as pd
from encoder import Encoder

parser = argparse.ArgumentParser(description='Evaluate Encoder pairs of ToDayGAN')
parser.add_argument('--batchSize', type=int, default=72,
        help='Number batches used for inference.')
parser.add_argument('--projectionSize', type=int, default=8192,
        help='Number of dimensions the image encoding gets projected to.')
parser.add_argument('--threads', type=int, default=8, help='Number of threads for each data loader to use')
parser.add_argument('--split', type=str, default='test_hard', help='Data split to encode. Default is test_hard',
        choices=['test', 'train', 'val', 'test_hard'])
parser.add_argument('--sim_measures', choices=['cos_sim', 'l2', 'l1'], default=['cos_sim'], nargs="+",
        help='Similarity measures to use. Default is cosine similarity.')
parser.add_argument('--name', type=str, default='eval_' + datetime.now().strftime('%b%d_%H-%M-%S'),
        help='Name of experiment, used for saving.')
parser.add_argument('--checkpointDay', type=str, help='Path to load day encoder checkpoint from.')
parser.add_argument('--checkpointNight', type=str, help='Path to load night encoder checkpoint from.')
parser.add_argument('--savePath', type=str, default='./results',
        help='Path to save results to. Default=results/')
parser.add_argument('--save_feat', action='store_true', help='Store encoded features.')

def test(eval_set, save_feat=False):
    """A function used to evaluate the encoders on `eval_set`.

    The results get saved as csv files.

    Parameters
    ----------
    eval_set : torch.utils.data.Dataset
        dataset to use for evaluation process
    save_feat : bool, optional
        whether or not to save calculated image features (default is False)
    """

    n_values = [1,5,10,20] # values for calculating top-N-recall
    rng = np.random.RandomState(42) # seed for sparse random projection
    nDB = eval_set.dbStruct.numDb
    test_data_loader = DataLoader(dataset=eval_set, num_workers=opt.threads,
                                  batch_size=opt.batchSize, shuffle=False,
                                  pin_memory=True)

    model_day.eval()
    model_night.eval()
    with torch.no_grad():
        print('====> Extracting Features')
        feat_size = opt.projectionSize
        transformer = SparseRandomProjection(
            n_components=feat_size, random_state=rng)
        transformer = transformer.fit(rng.rand(opt.batchSize, 72*96*256))
        dbFeat = np.empty((len(eval_set), feat_size), dtype=np.float32)

        for iteration, (input, indices) in enumerate(test_data_loader, 1):
            input = input.to(device)
            split_idx = opt.batchSize - indices[-1] + nDB - 1
            image_encoding = torch.flatten(
                split_input(input, split_idx), start_dim=1)

            dbFeat[indices.detach().numpy(), :] = transformer.transform(
                image_encoding.detach().cpu().numpy())
            if iteration % 50 == 0 or len(test_data_loader) <= 10:
                print("==> Batch ({}/{})".format(iteration,
                    len(test_data_loader)), flush=True)

            del input, image_encoding
    del test_data_loader

    # extracted for both db and query, now split into seperate arrays
    qFeat = dbFeat[nDB:].astype('float32')
    dbFeat = dbFeat[:nDB].astype('float32')

    if save_feat:
        savemat(join(opt.savePath, 'feat_' + str(opt.name) + '.mat'),
            {'dbFeat': dbFeat, 'qFeat': qFeat})

    if 'cos_sim' in opt.sim_measures:
        print('====> Calculating cosine similarity')
        cos_sim = cosine_similarity(dbFeat, qFeat)

        print('====> Calculating recall @ N')
        recalls = recall_from_sim_mat(
            cos_sim, eval_set.getPositives(), n_values)
        print('====> Calculating precision and recall metrics')
        bestF, bestP, bestR, AP, R, P = precision_recall(
            eval_set.dbStruct.gtSoft, eval_set.dbStruct.gtHard, cos_sim)
        metrics_list = [opt.name + '_cos_sim']
        metrics_list.extend(list(recalls.values()))
        metrics_list.extend([bestF, bestP, bestR, AP])

        df_metrics = pd.DataFrame([metrics_list],
                                  columns=['Name', 'R@1', 'R@5', 'R@10', 'R@20',
                                           'bestF', 'bestP', 'bestR', 'AP'])
        df_metrics.to_csv(
            join(opt.savePath, 'encoder_metrics.csv'), mode='a', header=False)

        df_pr = pd.DataFrame([[opt.name + '_cos_sim', R, P]],
                             columns=['Name', 'Recall', 'Precision'])
        df_pr.to_csv(
            join(opt.savePath, 'encoder_pr.csv'), mode='a', header=False)
        del cos_sim

    if 'l2' in opt.sim_measures:
        print('====> Calculating L2 similarity')
        l2 = euclidean_distances(dbFeat, qFeat)
        l2 = 1. - (l2 / np.amax(l2))

        print('====> Calculating recall @ N')
        recalls = recall_from_sim_mat(l2, eval_set.getPositives(), n_values)
        print('====> Calculating precision and recall metrics')
        bestF, bestP, bestR, AP, R, P = precision_recall(
            eval_set.dbStruct.gtSoft, eval_set.dbStruct.gtHard, l2)
        metrics_list = [opt.name + '_L2']
        metrics_list.extend(list(recalls.values()))
        metrics_list.extend([bestF, bestP, bestR, AP])

        df_metrics = pd.DataFrame([metrics_list],
                                  columns=['Name', 'R@1', 'R@5', 'R@10', 'R@20',
                                           'bestF', 'bestP', 'bestR', 'AP'])
        df_metrics.to_csv(
            join(opt.savePath, 'encoder_metrics.csv'), mode='a', header=False)

        df_pr = pd.DataFrame(
            [[opt.name + '_L2', R, P]], columns=['Name', 'Recall', 'Precision'])
        df_pr.to_csv(
            join(opt.savePath, 'encoder_pr.csv'), mode='a', header=False)
        del l2


        print('====> Calculating L2 similarity for L2-normalized')
        l2 = euclidean_distances(
            dbFeat/np.linalg.norm(dbFeat, ord=2, axis=1, keepdims=True),
            qFeat/np.linalg.norm(qFeat, ord=2, axis=1, keepdims=True))
        l2 = 1. - (l2 / np.amax(l2))

        print('====> Calculating recall @ N')
        recalls = recall_from_sim_mat(l2, eval_set.getPositives(), n_values)
        print('====> Calculating precision and recall metrics')
        bestF, bestP, bestR, AP, R, P = precision_recall(
            eval_set.dbStruct.gtSoft, eval_set.dbStruct.gtHard, l2)
        metrics_list = [opt.name + '_L2_norm']
        metrics_list.extend(list(recalls.values()))
        metrics_list.extend([bestF, bestP, bestR, AP])

        df_metrics = pd.DataFrame([metrics_list],
                                  columns=['Name', 'R@1', 'R@5', 'R@10', 'R@20',
                                           'bestF', 'bestP', 'bestR', 'AP'])
        df_metrics.to_csv(
            join(opt.savePath, 'encoder_metrics.csv'), mode='a', header=False)

        df_pr = pd.DataFrame([[opt.name + '_L2_norm', R, P]],
                             columns=['Name', 'Recall', 'Precision'])
        df_pr.to_csv(
            join(opt.savePath, 'encoder_pr.csv'), mode='a', header=False)
        del l2

    if 'l1' in opt.sim_measures:
        print('====> Calculating L1 similarity')
        l1 = manhattan_distances(dbFeat, qFeat)
        l1 = 1. - (l1 / np.amax(l1))

        print('====> Calculating recall @ N')
        recalls = recall_from_sim_mat(l1, eval_set.getPositives(), n_values)
        print('====> Calculating precision and recall metrics')
        bestF, bestP, bestR, AP, R, P = precision_recall(
            eval_set.dbStruct.gtSoft, eval_set.dbStruct.gtHard, l1)
        metrics_list = [opt.name + '_L1']
        metrics_list.extend(list(recalls.values()))
        metrics_list.extend([bestF, bestP, bestR, AP])

        df_metrics = pd.DataFrame([metrics_list],
                                  columns=['Name', 'R@1', 'R@5', 'R@10', 'R@20',
                                           'bestF', 'bestP', 'bestR', 'AP'])
        df_metrics.to_csv(
            join(opt.savePath, 'encoder_metrics.csv'), mode='a', header=False)

        df_pr = pd.DataFrame([[opt.name + '_L1', R, P]],
                             columns=['Name', 'Recall', 'Precision'])
        df_pr.to_csv(
            join(opt.savePath, 'encoder_pr.csv'), mode='a', header=False)
        del l1


def split_input(x, split_size=0):
    """A function to divide the input `x` between the encoders.

    Splits the input between the encoders at index `split_size`.

    Parameters
    ----------
    x : torch.float
        input tensor consisting of concatenated image tensors
    split_size: int, optional
        index at which to split input between encoders (default is 0)

    Returns
    -------
    torch.float
        a tensor consisting of concatenated image feature tensors
    """

    split_size = int(split_size)
    if split_size >= x.size()[0]:
        return model_day(x)
    elif split_size > 0:
        x_day = model_day(x[0:split_size])
        x_night = model_night(x[split_size:])
        return torch.cat([x_day, x_night])
    else:
        return model_night(x)

def recall_from_sim_mat(sim_mat, gt_mat, n_values=[1, 5, 10, 20]):
    """A function to calculate the top-N-recall from a similarity matrix.

    Parameters
    ----------
    sim_mat : array_like
        similarity matrix of database and querys
    gt_mat : array_like
        array of logical values representing the ground truth
    n_values : list, optional
        list of values to calculate top-N-recall for (default is [1, 5, 10, 20])

    Returns
    -------
    dict
        dictionary containing top-N-recalls
    """

    # find max(n_values) largest values for each query & return indices
    predictions = (-sim_mat).argsort(axis=0)[:max(n_values), :]

    correct_at_n = np.zeros(len(n_values))
    for qIx, pred in enumerate(predictions.transpose()):
        for i,n in enumerate(n_values):
            # if in top N then also in top NN, where NN > N
            if np.any(np.in1d(pred[:n], gt_mat[qIx])):
                correct_at_n[i:] += 1
                break
    recall_at_n = correct_at_n / whole_test_set.dbStruct.numQ

    recalls = {} # make dict for output
    for i,n in enumerate(n_values):
        recalls[n] = recall_at_n[i]
        print("====> Recall@{}: {:.4f}".format(n, recall_at_n[i]))
    return recalls


if __name__ == "__main__":
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if opt.split.lower() == 'test':
        whole_test_set = dataset.get_whole_test_set()
        print('===> Evaluating on test set')
    elif opt.split.lower() == 'train':
        whole_test_set = dataset.get_whole_training_set()
        print('===> Evaluating on train set')
    elif opt.split.lower() == 'val':
        whole_test_set = dataset.get_whole_val_set()
        print('===> Evaluating on val set')
    elif opt.split.lower() == 'test_hard':
        whole_test_set = dataset.get_test_hard_set()
        print('===> Evaluating on test hard set')
    else:
        raise ValueError('Unknown dataset split: ' + opt.split)
    print('===> Query count:', whole_test_set.dbStruct.numQ)

    print('===> Building model')
    encoder_dim = 256
    model_day = Encoder()
    model_day.load_state_dict(torch.load(opt.checkpointDay))
    model_night = Encoder()
    model_night.load_state_dict(torch.load(opt.checkpointNight))

    model_day = model_day.to(device)
    model_night = model_night.to(device)

    print('===> Running evaluation step')
    test(whole_test_set, save_feat=opt.save_feat)
