"""
A module used for building the NetVLAD layer.

This implementation is adopted from: https://github.com/Nanne/pytorch-NetVlad
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
import numpy as np

# based on https://github.com/lyakaap/NetVLAD-pytorch/blob/master/netvlad.py
class NetVLAD(nn.Module):
    """
    A class implementing the NetVLAD layer.

    The cluster centers are implemented as a convolutional layer. Two different
    variants of the NetVLAD layer are available.

    Attributes
    ----------
    num_clusters : int
        the number of clusters (default is 64)
    dim : int
        depth of descriptors (default is 128)
    normalize_input : bool
        if true, descriptor-wise L2-normalization is applied to input
    vladv2 : bool
        if true, use vladv2 otherwise use vladv1
    alpha : float
        parameter of initialization, larger value is harder assignment

    Methods
    -------
    forward(x, split_size=0)
        The standard PyTorch method for forward passing an input.
    set_train_layers(layer_names=[])
        Determines which layers should be trainable.
    """

    def __init__(self, num_clusters=64, dim=128,
                 normalize_input=True, vladv2=False):
        """
        Parameters
        ----------
            num_clusters : int
                the number of clusters (default is 64)
            dim : int
                depth of descriptors (default is 128)
            normalize_input : bool
                if true, descriptor-wise L2-normalization is applied to input
            vladv2 : bool
                if true, use vladv2 otherwise use vladv1
        """

        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = 0
        self.vladv2 = vladv2
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(
            dim, num_clusters, kernel_size=(1, 1), bias=vladv2)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))

    def init_attributes(self, clsts, traindescs):
        """Initializes the attributes of a NetVLAD instance.

        Parameters
        ----------
        clsts : array_like
            array of cluster centers
        traindescs: array_like
            array of image descriptors used for building the clusters
        """

        if self.vladv2 == False:
            clstsAssign = clsts / np.linalg.norm(clsts, axis=1, keepdims=True)
            dots = np.dot(clstsAssign, traindescs.T)
            dots.sort(0)
            dots = dots[::-1, :] # sort, descending

            self.alpha = (-np.log(0.01) / np.mean(dots[0,:] - dots[1,:])).item()
            self.centroids = nn.Parameter(torch.from_numpy(clsts))
            self.conv.weight = nn.Parameter(
                torch.from_numpy(
                self.alpha*clstsAssign).unsqueeze(2).unsqueeze(3))
            self.conv.bias = None
        else:
            knn = NearestNeighbors(n_jobs=-1)
            knn.fit(traindescs)
            del traindescs
            dsSq = np.square(knn.kneighbors(clsts, 2)[1])
            del knn
            self.alpha = (-np.log(0.01) / np.mean(dsSq[:,1] - dsSq[:,0])).item()
            self.centroids = nn.Parameter(torch.from_numpy(clsts))
            del clsts, dsSq

            self.conv.weight = nn.Parameter(
                (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
            )
            self.conv.bias = nn.Parameter(
                - self.alpha * self.centroids.norm(dim=1)
            )

    def forward(self, x):
        """The standard PyTorch method for forward passing an input.

        Returns the NetVLAD descriptors for a given bacht of inputs `x`.

        Parameters
        ----------
        x : torch.float
            input tensor of concatenated image feature tensors

        Returns
        -------
        torch.float
            a tensor of concatenated NetVLAD descriptors
        """

        N, C = x.shape[:2] # get batch size and number of channels

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # along channel dim

        # soft-assignment
        soft_assign = self.conv(x).view(N, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)

        x_flatten = x.view(N, C, -1) # squeeze image features

        # calculate residuals to each clusters
        vlad = torch.zeros([N, self.num_clusters, C], dtype=x.dtype,
                           layout=x.layout, device=x.device)
        # slower than non-looped, but lower memory usage
        for C in range(self.num_clusters):
            residual = x_flatten.unsqueeze(0).permute(1, 0, 2, 3) \
                - self.centroids[C:C+1, :].expand(x_flatten.size(-1), -1, -1) \
                                          .permute(1, 2, 0) \
                                          .unsqueeze(0)
            residual *= soft_assign[:,C:C+1,:].unsqueeze(2)
            vlad[:,C:C+1,:] = residual.sum(dim=-1)
        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(x.size(0), -1)  # flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize

        return vlad
