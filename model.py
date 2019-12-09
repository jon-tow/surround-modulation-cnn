from typing import Tuple, List

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

π = math.pi


# Gaussian Functions.


def gaussian1d(kernel_size: int, σ: float) -> torch.Tensor:
    """
    Returns a 1-dimensional Gaussian kernel (filter) tensor.

    Arguments:
    - kernel_size: The size of the resulting 1D Gaussian kernel.
    - σ: The standard deviation of the Gaussian.
    """
    assert kernel_size % 2 != 0, f'Kernel size must be an odd integer > 0. Got: {kernel_size}'
    assert σ > 0, f'Standard deviation must be non-negative. Got: {σ}'

    center = kernel_size // 2
    x = torch.arange(0, kernel_size, dtype=torch.float32)
    normalization = 1 / (σ * torch.sqrt(torch.tensor(2 * π)))
    kernel = normalization * torch.exp(-(x - center)**2 / (2 * σ**2))
    return kernel


def gaussian2d(kernel_sizes: Tuple[int, int], σ: float) -> torch.Tensor:
    """
    Returns a 2-dimensional Gaussian kernel (filter) tensor constructed by
    multiplying two 1-dimensional Gaussian kernels (See: Gaussian Separability).

    Arguments:
    - kernel_sizes: A 2-tuple of kernel sizes for the height and width spatial
                    dimensions of the resulting 2D Gaussian.
    - σ: The standard deviation of the 2-dimensional Gaussian.
    """
    size_x, size_y = kernel_sizes[0], kernel_sizes[1]
    kernel_x = gaussian1d(size_x, σ)
    kernel_y = gaussian1d(size_y, σ)
    # Note: `torch.ger` is the outer product (weird name from BLAS...).
    kernel = torch.ger(kernel_x, kernel_y)  # By separability of 1D Gaussians.
    return kernel


def DoG(kernel_sizes: Tuple[int, int], σs: Tuple[float, float]) -> torch.Tensor:
    """
    Returns the difference of Gaussians (DoG) between two Gaussian kernels with
    formed by the given standard deviations. See paper's equation `(1)`.

    Arguments:
    - kernel_sizes: A 2-tuple of the kernel sizes for the height and width
                    spatial dimensions of the resulting differece of Gaussians.
    - σs: A 2-tuple of standard deviations used to construct the two
          2-dimensional Gaussian kernels.
    """
    σ_1, σ_2 = σs
    kernel = gaussian2d(kernel_sizes, σ_1) - gaussian2d(kernel_sizes, σ_2)
    return kernel


# Surround Modulation.


def surround_modulation(kernel_size: int, σ_e: float, σ_i: float) -> torch.Tensor:
    """
    Returns the Surround Modulation (`sm`) kernel tensor with the given
    excitatory and inhibitory standard deviations as described in equation
    `(2)`.

    Arguments:
    - kernel_size: The size of the Surround Modulation kernel. The size must be
                   an odd integer > 0 so that the surround modulation kernel
                   can have shape (2k + 1) x (2k + 1), where `k` is the padding
                   size used on an activation input.
    - σ_e: The standard deviation of the excitatory Gaussian.
    - σ_i: The standard deviation of the inhibitory Gaussian.
    """
    assert kernel_size % 2 != 0, f'The kernel size must be odd. Got: {kernel_size}'

    center = kernel_size // 2
    dog = DoG((kernel_size, kernel_size), σs=(σ_e, σ_i))
    kernel = dog / dog[center][center]
    return kernel


class SMConv(nn.Module):
    """ A module for Surround Modulated (SM) Convolutions. """

    def __init__(
        self,
        kernel_size: int = 5,
        σ_e: float = 1.2,
        σ_i: float = 1.4,
        mask_p: float = 0.5
    ):
        """
        Creates a Surround Modulated Convolution layer.

        _Note_: This module only works with 2D convolutions.

        Arguments:
        - kernel_size: The size of the Surround Modulation kernel. The size must
                       be an odd integer > 0 so that the surround modulation
                       kernel can have shape (2k + 1) x (2k + 1), where `k` is
                       where `k` is the padding size used on an activation
                       input.
        - σ_e: The standard deviation of the excitatory Gaussian.
               Default: 1.2
        - σ_i: The standard deviation of the inhibitory Gaussian.
               Default: 1.4
        - mask_p: The approximate percent of activation maps to mask. Approximate
               because the selected indices are not sampled uniquely. A large
               percent means NOT modulating a large number of maps in the input.
               Default: 0.5
        """
        assert mask_p >= 0.0 and mask_p <= 1.0, \
            f'The percent of activation maps to mask must lie in [0.0, 1.0]. Got: {mask_p}'

        super().__init__()
        self.kernel_size = kernel_size
        self.σ_e = σ_e
        self.σ_i = σ_i
        self.mask_p = mask_p
        # Convolution padding size to keep spatial dimensions the same.
        self.same = self.kernel_size // 2
        self.register_buffer(
            'kernel', surround_modulation(kernel_size, σ_e, σ_i))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Returns the surround modulated activation map of the input activations.

        Arguments
        - input: An activation tensor map of shape:
          `[batch, channel, height, width]`.
        """
        # FIX: Currently inefficient random sampling of channels.
        channel_size = input.shape[1]
        # Spread the kernel over the batch and channel dimensions of the input.
        kernel = self.kernel.repeat(channel_size, channel_size, 1, 1)
        # Use the dirac delta (convolutional identity) kernel to mask the sm
        # kernel so as to avoid convolving certain input activations.
        mask_size = math.floor(self.mask_p * channel_size)
        i = torch.randint(low=0, high=channel_size, size=(mask_size, ))
        kernel[i] = nn.init.dirac_(kernel[i])
        return F.conv2d(input, kernel, padding=self.same)


# Surround Modulation CNN (`SMCNN`).


class SMCNN(nn.Module):
    """
    An interpretation of the `main` Surround Modulation Convolutional Neural
    Network variant from the paper:
    "Surround Modulation: A Bio-inspired Connectivity Structure for CNNs".

    Architecture:
               FEATURES
    - input -> |=> (1 * [conv-relu -> smconv])
               |=> (4 * [conv-relu -> maxpool]) -> (1 * [conv-relu -> adaptivemaxpool])
               |=> (1 * [conv-relu -> flatten])
               CLASSIFIER
               |=> ((2 * [linear-relu -> dropout])
               |=> (1 * [linear])

    Arguments:
    - num_classes: The number of class scores to return.
    - init_channels: The number of output channels in the initial
                     convolution + surround modulation layer.
    - hidden_channels: A list containing the number of output channels
                       in the hidden convolution layers.
    """

    def __init__(
        self,
        num_classes: int = 200
    ):
        super().__init__()

        input_channel = 16
        last_channel = 256
        feature_size = 4

        # Convolution `features` pipeline.
        self.features = []
        self.features.extend([
            nn.Conv2d(3, input_channel, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            SMConv(kernel_size=5)])
        hidden_channels = [input_channel, 32, 64, 64, 128, 256]
        for in_channels, out_channels in zip(hidden_channels, hidden_channels[1:]):
            self.features.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 2), stride=2)])
        self.features.pop()  # Remove last `MaxPool2d`.
        self.features.extend([
            nn.AdaptiveMaxPool2d(output_size=feature_size),
            nn.Conv2d(256, last_channel, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten()])
        self.features = nn.Sequential(*self.features)

        # Fully-connected `classifier` pipeline.
        self.classifier = []
        feature_sizes = [last_channel * feature_size**2, 256, 256, num_classes]
        for in_features, out_features in zip(feature_sizes, feature_sizes[1:]):
            self.classifier.extend([
                nn.Linear(in_features, out_features),
                nn.ReLU(inplace=True),
                nn.Dropout()])
        # Remove last `Dropout` and `ReLU` to output only class scores.
        self.classifier = self.classifier[:-2]
        self.classifier = nn.Sequential(*self.classifier)

        # Weight Initialization. Author's use Xavier (Glorot) everywhere.
        for module in self.children():
            if isinstance(module, nn.Conv2d) \
                    or isinstance(module, nn.Linear) \
                    or isinstance(module, nn.MaxPool2d) \
                    or isinstance(module, nn.AdaptiveMaxPool2d):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        scores = self.classifier(self.features(input))
        return scores
