# NetVLAD for multiple input domains

This project is an extension of the
[NetVLAD](https://www.di.ens.fr/willow/research/netvlad/) architecture that
makes it possible to apply it to datasets with multiple image domains. The code
for training and evaluating it on the
[Pittsburgh Dataset](http://www.ok.ctrl.titech.ac.jp/~torii/project/repttile/)
and the "RobotCar Dataset" (based on the
[Oxford RobotCar Dataset](https://robotcar-dataset.robots.ox.ac.uk/)) is
provided.

## Setup

### Dependencies

1. [PyTorch](https://pytorch.org/get-started/locally/) (at least v0.4.0)
2. [Faiss](https://github.com/facebookresearch/faiss)
3. [scipy](https://www.scipy.org/)
    - [numpy](http://www.numpy.org/)
    - [sklearn](https://scikit-learn.org/stable/)
    - [h5py](https://www.h5py.org/)
4. [tensorboardX](https://github.com/lanpa/tensorboardX)
5. [Pillow](https://pillow.readthedocs.io/en/stable/)
6. [matplotlib](https://matplotlib.org/)
7. [pandas](https://pandas.pydata.org/)

The listet packages can be installed using [pip](https://pip.pypa.io/en/stable/)
or [Anaconda](https://www.anaconda.com/products/individual).

### Data

#### RobotCar Dataset

The first group of datasets that can be used are based on the
[Oxford RobotCar Dataset](https://robotcar-dataset.robots.ox.ac.uk/). The
following sequences are required for the different datasets available:
- 2014-11-14-16-34-33
- 2014-12-02-15-30-08
- 2014-12-09-13-21-02
- 2014-12-16-18-44-24
- 2015-02-03-19-43-11

Only the images of the Bumblebee XB3 camera are used!
The images of each sequence should be placed in the `Robotcar` directory
directly into the matching folder. The folders with the extension `_GAN` are
used for storing the results of translating images to the other domain using
[ToDayGAN](https://github.com/AAnoosheh/ToDayGAN). They are required for some
of the provided datasets. The dataset specifications (.mat files) are located in
the `datasets` folder in the `Robotcar` directory. When using one of the
provided datasets make sure to place the required images in the matching folder:

| dataset | required sequences |
| --- | --- |
| `robotcar` | 2014-12-09-13-21-02 <br> 2014-12-16-18-44-24 |
| `robotcar_day` | 2014-12-09-13-21-02 <br> 2014-12-02-15-30-08 |
| `robotcar_synthetic` | 2014-12-09-13-21-02 <br> 2014-12-02-15-30-08_GAN |
| `robotcar_db_gtmat` | 2014-12-09-13-21-02 <br> 2014-12-09-13-21-02_GAN |
| `robotcar_db_nogt` | 2014-12-09-13-21-02 <br> 2014-12-09-13-21-02_GAN |
| `test` | 2014-12-09-13-21-02 <br> 2015-02-03-19-43-11 |
| `test_hard` | 2014-12-02-15-30-08 <br> 2014-11-14-16-34-33 |
| `val` | 2014-12-09-13-21-02 <br> 2015-02-03-19-43-11 |

#### Pittsburgh dataset

The other option is to train on Pittsburgh 250k (available
[here](http://www.ok.ctrl.titech.ac.jp/~torii/project/repttile/)), and the
dataset specifications for the Pittsburgh dataset (available
[here](https://www.di.ens.fr/willow/research/netvlad/data/netvlad_v100_datasets.tar.gz)).
`pittsburgh.py` contains a hardcoded path to a directory, where the code expects
directories `000` to `010` with the various Pittsburgh database images, a
directory `queries_real` with subdirectories `000` to `010` with the query
images, and a directory `datasets` with the dataset specifications (.mat files).

## Usage

`main.py` contains the majority of the code, and has three different modes
(`train`, `test`, `cluster`) which will be explained in more detail below.
Make sure to point `--cachePath` to a directory to save the cache to. This can
be a tempory folder in the `/tmp` directory.

### Train

In order to initialise the NetVlad layer it is necessary to first run `main.py`
with the correct settings and `--mode=cluster` (see section Cluster). After
which a model can be trained using (the following default flags):
```bash
python main.py --mode=train --arch=vgg16 --pooling=netvlad --num_clusters=64
```

The architecture can be changed with the commandline arguments `--arch`,
`--cnnArch`, `--addConv`, `--pooling` and `--num_clusters`. For example, to
train a model with 2 ToDayGAN encoders and an added shared layer:
```bash
python main.py --mode=train --arch=todaygan --cnnArch=dual --addConv --encoderPath=<path_encoder_checkpoint> --pooling=netvlad --num_clusters=64
```

The commandline args, the tensorboard data, and the model state will all be
saved to `opt.runsPath`, which subsequently can be used for testing, or to
resume training.

For more information on all commandline arguments run:
```bash
python main.py --help
```

### Test

To test a previously trained model on the `RobotCar Test Hard` testset (replace
resume directory with correct dir for your case):
```bash
python main.py --mode=test --resume=runs/<checkpoint_folder> --split=test_hard
```

The commandline arguments for training were saved, so you shouldn't need to
specify them for testing.
Additionally, to obtain the 'off the shelf' performance you can also omit the
resume directory (only for VGG16 and AlexNet):
```bash
python main.py --mode=test --arch=vgg16
```

### Cluster

In order to initialise the NetVlad layer we need to first sample from the data
and obtain `opt.num_clusters` centroids. This step is necessary for each
configuration of the network and for each dataset. To create clusters run:
```bash
python main.py --mode=cluster --arch=vgg16 --pooling=netvlad --num_clusters=64
```
with the correct values for any additional commandline arguments.

## Acknowledgment

This project is based on this
[NetVLAD implementation](https://github.com/Nanne/pytorch-NetVlad) in PyTorch
and reuses its code. The original NetVLAD paper is available
[here](https://arxiv.org/abs/1511.07247).

The implementation of the ToDayGAN encoder recreates the encoder network from
the [ToDayGAN implementation](https://github.com/AAnoosheh/ToDayGAN) by the
authors of the [ToDayGAN paper](https://arxiv.org/abs/1809.09767).

## License
[MIT](https://choosealicense.com/licenses/mit/)
