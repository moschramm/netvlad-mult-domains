"""
A module used for building the encoder module of a NetVLAD model.

It is espacially used for building an encoder consisting of multiple (two) CNNs.
"""

import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models
from os.path import join

class DualModel(nn.Module):
    """
    A class used to include two CNNs into one model.

    Every CNN is used for one domain of input images.

    Attributes
    ----------
    arch : {'alexnet', 'vgg16', 'todaygan'}
        the name of the CNN architecture to use
    encoder_path : str
        the path to the ToDayGAN encoder checkpoints
    add_l2 : bool, optional
        whether or not to add a L2-Normalize-Layer (default is False)
    pretrained : bool, optional
        whether or not to load pretrained weights for CNNs (default is True)

    Methods
    -------
    forward(x, split_size=0)
        The standard PyTorch method for forward passing an input.
    set_train_layers(layer_names=[])
        Determines which layers should be trainable.
    """

    def __init__(self, arch, encoder_path, add_l2=False, pretrained=True):
        """
        Parameters
        ----------
        arch : {'alexnet', 'vgg16', 'todaygan'}
            the name of the CNN architecture to use
        encoder_path : str
            the path to the ToDayGAN encoder checkpoints
        add_l2 : bool, optional
            whether or not to add a L2-Normalize-Layer (default is False)
        pretrained : bool, optional
            whether or not to load pretrained weights for CNNs (default is True)
        """

        super(DualModel, self).__init__()
        if arch.lower() == 'alexnet':
            self.day_encoder = AlexNet(add_l2=add_l2, pretrained=pretrained)
            self.night_encoder = AlexNet(add_l2=add_l2, pretrained=pretrained)
        elif arch.lower() == 'vgg16':
            self.day_encoder = VGG16(add_l2=add_l2, pretrained=pretrained)
            self.night_encoder = VGG16(add_l2=add_l2, pretrained=pretrained)
        elif arch.lower() == 'todaygan':
            self.day_encoder = Encoder(add_l2)
            self.day_encoder.load_state_dict(
                torch.load(join(encoder_path, '190_net_G0.pth')))
            self.night_encoder = Encoder(add_l2)
            self.night_encoder.load_state_dict(
                torch.load(join(encoder_path, '190_net_G1.pth')))

    def forward(self, x, split_size=0):
        """The standard PyTorch method for forward passing an input.

        Splits the input between the CNNs at index `split_size`.

        Parameters
        ----------
        x : torch.float
            input tensor consisting of concatenated image tensors
        split_size : int, optional
            index at which to split input between CNNs (default is 0)

        Returns
        -------
        torch.float
            a tensor consisting of concatenated image feature tensors
        """

        split_size = int(split_size)
        if split_size >= x.size()[0]:
            return self.day_encoder(x)
        elif split_size > 0:
            x_day = self.day_encoder(x[0:split_size])
            x_night = self.night_encoder(x[split_size:])
            return torch.cat([x_day, x_night])
        else:
            return self.night_encoder(x)

    def set_train_layers(self, layer_names=[]):
        """Determines which layers should be trainable.

        The specified parameters' requires_grad is not set to False.

        Parameters
        ----------
        layer_names : list, optional
            The name of the parameters which should stay trainable
            (default is [])
        """

        for k, p in self.day_encoder.named_parameters():
            bool_list = []
            for layer in layer_names:
                bool_list.append(k.startswith(layer))
            if not any(bool_list):
                p.requires_grad = False
        for k, p in self.night_encoder.named_parameters():
            bool_list = []
            for layer in layer_names:
                bool_list.append(k.startswith(layer))
            if not any(bool_list):
                p.requires_grad = False

class DualModelShared(nn.Module):
    """
    A class used to include two CNNs and a shared convolutional layer into one
    model.

    Every CNN is used for one domain of input images. The shared convolutional
    layer is used for multiple domains.

    Attributes
    ----------
    arch : {'alexnet', 'vgg16', 'todaygan'}
        the name of the CNN architecture to use
    encoder_path : str
        the path to the ToDayGAN encoder checkpoints
    add_l2 : bool, optional
        whether or not to add a L2-Normalize-Layer (default is False)
    pretrained : bool, optional
        whether or not to load pretrained weights for CNNs (default is True)

    Methods
    -------
    forward(x, split_size=0)
        The standard PyTorch method for forward passing an input.
    set_train_layers(layer_names=['model.14'])
        Determines which layers should be trainable.
    """

    def __init__(self, arch, encoder_path, add_l2=False, pretrained=True):
        """
        Parameters
        ----------
        arch : {'alexnet', 'vgg16', 'todaygan'}
            the name of the CNN architecture to use
        encoder_path : str
            the path to the ToDayGAN encoder checkpoints
        add_l2 : bool, optional
            whether or not to add a L2-Normalize-Layer (default is False)
        pretrained : bool, optional
            whether or not to load pretrained weights for CNNs (default is True)
        """

        super(DualModelShared, self).__init__()
        self.day_encoder = Encoder(False)
        self.day_encoder.load_state_dict(
            torch.load(join(encoder_path, '190_net_G0.pth')), strict=False)

        self.night_encoder = Encoder(False)
        self.night_encoder.load_state_dict(
            torch.load(join(encoder_path, '190_net_G1.pth')), strict=False)
        self.shared_layer = Downsample(add_l2)

    def forward(self, x, split_size=0):
        """The standard PyTorch method for forward passing an input.

        Splits the input between the CNNs at index `split_size`.

        Parameters
        ----------
        x : torch.float
            input tensor consisting of concatenated image tensors
        split_size : int, optional
            index at which to split input between CNNs (default is 0)

        Returns
        -------
        torch.float
            a tensor consisting of concatenated image feature tensors
        """

        split_size = int(split_size)
        if split_size >= x.size()[0]:
            return self.shared_layer(self.day_encoder(x))
        elif split_size > 0:
            x_day = self.day_encoder(x[0:split_size])
            x_night = self.night_encoder(x[split_size:])
            return self.shared_layer(torch.cat([x_day, x_night]))
        else:
            return self.shared_layer(self.night_encoder(x))

    def set_train_layers(self, layer_names=['model.14']):
        """Determines which layers should be trainable.

        The specified parameters' requires_grad is not set to False. The shared
        convolutional layer ('model.14') is always set trainable.

        Parameters
        ----------
        layer_names : list, optional
            The name of the parameters which should stay trainable
            (default is ['model.14'])
        """

        if 'model.14' in layer_names:
            layer_names.remove('model.14')
        else:
            for p in self.shared_layer.parameters():
                p.requires_grad = False
        for k, p in self.day_encoder.named_parameters():
            bool_list = []
            for layer in layer_names:
                bool_list.append(k.startswith(layer))
            if not any(bool_list):
                p.requires_grad = False
        for k, p in self.night_encoder.named_parameters():
            bool_list = []
            for layer in layer_names:
                bool_list.append(k.startswith(layer))
            if not any(bool_list):
                p.requires_grad = False


class AlexNet(nn.Module):
    """
    A class used to build an AlexNet CNN model.

    Attributes
    ----------
    add_l2 : bool, optional
        whether or not to add a L2-Normalize-Layer (default is False)
    pretrained : bool, optional
        whether or not to load pretrained weights (default is True)

    Methods
    -------
    forward(input, split_size=None)
        The standard PyTorch method for forward passing an input.
    """

    def __init__(self, add_l2=False, pretrained=True):
        """
        Parameters
        ----------
        add_l2 : bool, optional
            whether or not to add a L2-Normalize-Layer (default is False)
        pretrained : bool, optional
            whether or not to load pretrained weights(default is True)
        """

        super(AlexNet, self).__init__()
        encoder = models.alexnet(add_l2=False, pretrained=pretrained)
        # capture only features and remove last relu and maxpool
        layers = list(encoder.features.children())[:-2]
        if pretrained:
            # if using pretrained only train conv5
            for l in layers[:-1]:
                for p in l.parameters():
                    p.requires_grad = False
        if add_l2:
            layers.append(L2Norm())

        self.model = nn.Sequential(*layers)

    def forward(self, input, split_size=None):
        """The standard PyTorch method for forward passing an input.

        Parameters
        ----------
        input : torch.float
            input tensor consisting of concatenated image tensors
        split_size : None, optional
            only implemented for compatibility reasons, shoud always be None

        Returns
        -------
        torch.float
            a tensor consisting of concatenated image feature tensors
        """
        return self.model(input)


class VGG16(nn.Module):
    """
    A class used to build an VGG16 CNN model.

    Attributes
    ----------
    add_l2 : bool, optional
        whether or not to add a L2-Normalize-Layer (default is False)
    pretrained : bool, optional
        whether or not to load pretrained weights (default is True)

    Methods
    -------
    forward(input, split_size=None)
        The standard PyTorch method for forward passing an input.
    """

    def __init__(self, add_l2=False, pretrained=True):
        """
        Parameters
        ----------
        add_l2 : bool, optional
            whether or not to add a L2-Normalize-Layer (default is False)
        pretrained : bool, optional
            whether or not to load pretrained weights (default is True)
        """

        super(VGG16, self).__init__()
        encoder = models.vgg16(pretrained=pretrained)
        # capture only feature part and remove last relu and maxpool
        layers = list(encoder.features.children())[:-2]
        if pretrained:
            # if using pretrained then only train conv5_1, conv5_2, and conv5_3
            for l in layers[:-5]:
                for p in l.parameters():
                    p.requires_grad = False
        if add_l2:
            layers.append(L2Norm())

        self.model = nn.Sequential(*layers)

    def forward(self, input, split_size=None):
        """The standard PyTorch method for forward passing an input.

        Parameters
        ----------
        input : torch.float
            input tensor consisting of concatenated image tensors
        split_size : None, optional
            only implemented for compatibility reasons, shoud always be None

        Returns
        -------
        torch.float
            a tensor consisting of concatenated image feature tensors
        """
        return self.model(input)


class Downsample(nn.Module):
    """
    A class used to build a shared convolutional layer for ToDayGAN models.

    Attributes
    ----------
    add_l2 : bool, optional
        whether or not to add a L2-Normalize-Layer (default is False)
    pretrained : bool, optional
        whether or not to load pretrained weights (default is True)

    Methods
    -------
    forward(input)
        The standard PyTorch method for forward passing an input.
    """

    def __init__(self, add_l2=False):
        """
        Parameters
        ----------
        add_l2 : bool, optional
            whether or not to add a L2-Normalize-Layer (default is False)
        """

        super(Downsample, self).__init__()
        mult = 4
        model = [nn.Conv2d(64 * mult, 64 * mult * 2, kernel_size=3,
                            stride=2, padding=1, bias=True)]
        if add_l2:
            model += [L2Norm()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """The standard PyTorch method for forward passing an input.

        Parameters
        ----------
        input : torch.float
            input tensor consisting of concatenated image feature tensors from
            the previous layer

        Returns
        -------
        torch.float
            a tensor consisting of concatenated image feature tensors
        """
        return self.model(input)


class Encoder(nn.Module):
    """
    A class used to build an encoder model similar to ToDayGAN.

    This implementation is adopted from:
    https://github.com/AAnoosheh/ToDayGAN

    Attributes
    ----------
    add_l2 : bool, optional
        whether or not to add a L2-Normalize-Layer (default is False)

    Methods
    -------
    forward(input, split_size=None)
        The standard PyTorch method for forward passing an input.
    set_train_layers(layer_names=[])
        Determines which layers should be trainable.
    """

    def __init__(self, add_l2=False):
        """
        Parameters
        ----------
        add_l2 : bool, optional
            whether or not to add a L2-Normalize-Layer (default is False)
        """

        super(Encoder, self).__init__()
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(3, 64, kernel_size=7, padding=0, bias=True),
                 nn.InstanceNorm2d(64),
                 nn.PReLU()]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(64 * mult, 64 * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=True),
                      nn.InstanceNorm2d(64 * mult * 2),
                      nn.PReLU()]

        mult = 2**n_downsampling
        n_blocks = 9 // 2
        for _ in range(n_blocks):
            model += [ResnetBlock(64 * mult)]
        if add_l2:
            model += [L2Norm()]

        self.model = nn.Sequential(*model)

    def forward(self, input, split_size=None):
        """The standard PyTorch method for forward passing an input.

        Parameters
        ----------
        input : torch.float
            input tensor consisting of concatenated image tensors
        split_size : None, optional
            only implemented for compatibility reasons, shoud always be None

        Returns
        -------
        torch.float
            a tensor consisting of concatenated image feature tensors
        """

        return self.model(input)

    def set_train_layers(self, layer_names=[]):
        """Determines which layers should be trainable.

        The specified parameters' requires_grad is not set to False.

        Parameters
        ----------
        layer_names : list, optional
            The name of the parameters which should stay trainable
            (default is [])
        """

        for k, p in self.model.named_parameters():
            bool_list = []
            for layer in layer_names:
                bool_list.append(k.startswith(layer))
            if not any(bool_list):
                p.requires_grad = False


class ResnetBlock(nn.Module):
    """
    A class used to build a Resnet block which is used in the class Encoder.

    Attributes
    ----------
    dim : int
        number of filters to use in convolutional layers of the ResnetBlock

    Methods
    -------
    forward(input)
        The standard PyTorch method for forward passing an input.
    """

    def __init__(self, dim):
        """
        Parameters
        ----------
        dim : int
            number of filters to use in convolutional layers of the ResnetBlock
        """

        super(ResnetBlock, self).__init__()
        conv_block = [nn.ReflectionPad2d(1),
                      nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),
                      nn.InstanceNorm2d(dim),
                      nn.PReLU(),
                      nn.ReflectionPad2d(1),
                      nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),
                      nn.InstanceNorm2d(dim)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, input):
        """The standard PyTorch method for forward passing an input.

        Parameters
        ----------
        input : torch.float
            input tensor consisting of concatenated image feature tensors of
            previous layers

        Returns
        -------
        torch.float
            a tensor consisting of concatenated image feature tensors
        """
        return input + self.conv_block(input)


class L2Norm(nn.Module):
    """
    A class used to build a L2-Normalize layer to append to an encoder model.

    Attributes
    ----------
    dim : int, optional
        dimension along which to normalize the input (default is 1)

    Methods
    -------
    forward(input)
        The standard PyTorch method for forward passing an input.
    """

    def __init__(self, dim=1):
        """
        Parameters
        ----------
        dim : int, optional
            dimension along which to normalize the input (default is 1)
        """

        super().__init__()
        self.dim = dim

    def forward(self, input):
        """The standard PyTorch method for forward passing an input.

        Parameters
        ----------
        input : torch.float
            input tensor consisting of concatenated image feature tensors

        Returns
        -------
        torch.float
            a tensor consisting of concatenated image feature tensors,
            L2-normalized along `self.dim`
        """

        return F.normalize(input, p=2, dim=self.dim)
