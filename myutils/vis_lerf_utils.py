# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from jaxtyping import Bool, Float

import sklearn
import sklearn.decomposition
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
util_cmap = plt.cm.viridis
import os
import cv2

from torch import Tensor

# This is taken from nerfstudio
Colormaps = ["default", "turbo", "viridis", "magma", "inferno", "cividis", "gray", "pca"]

@dataclass(frozen=True)
class ColormapOptions:
    """Options for colormap"""

    colormap: str = "default"
    """ The colormap to use """
    normalize: bool = False
    """ Whether to normalize the input tensor image """
    colormap_min: float = 0
    """ Minimum value for the output colormap """
    colormap_max: float = 1
    """ Maximum value for the output colormap """
    invert: bool = False
    """ Whether to invert the output colormap """

VIS_PCA = sklearn.decomposition.PCA(3, random_state=42)

def imask_viz(imask):
    imask_cpu = (255 * np.squeeze(imask.cpu().numpy())).astype(np.uint8)
    imask_cpu = cv2.cvtColor(imask_cpu, cv2.COLOR_GRAY2RGB) # white = inlier, black = outlier
    #imask_cpu = (255 * cm(1-imask_cpu)[:,:,:3]).astype(np.uint8)
    return imask_cpu

def colored_depthmap(depth, d_min=None, d_max=None):
    if d_min is None:
        d_min = np.min(depth)
    if d_max is None:
        d_max = np.max(depth)
    depth_relative = (depth - d_min) / (d_max - d_min)
    depth_relative[depth_relative>1.0] = 1.0
    return 255 * util_cmap(depth_relative)[:,:,:3] # H, W, C

def util_add_row(img_merge, row):
    if (type(row) == tuple) or (type(row) == list):
        row_mat = np.hstack(row)
    else:
        row_mat = row
    if img_merge is None:
        return row_mat
    if (type(img_merge) == tuple) or (type(img_merge) == list):
        img_merge = np.hstack(img_merge)
    return np.vstack([img_merge, row_mat])


def util_save_image(img_merge, filename):
    img_merge = Image.fromarray(img_merge.astype('uint8'))
    img_merge.save(filename)


def get_shown_feat_map(featmap_, name:str=None):
    # featmap; (C, H, W)
    featmap = torch.nan_to_num(featmap_.detach().float(), nan=1e-6).cpu()
    fmap = F.normalize(featmap.unsqueeze(dim=0), dim=1).cpu() # (1, C, H, W)
    f_samples = fmap.permute(0, 2, 3, 1).reshape(-1, fmap.shape[1]).numpy() # [::3].cpu().numpy()
    transformed = VIS_PCA.fit_transform(f_samples)
    feature_pca_mean = torch.tensor(f_samples.mean(0)).float() # .cuda()
    feature_pca_components = torch.tensor(VIS_PCA.components_).float() # .cuda()
    q1, q99 = np.percentile(transformed, [1, 99])
    feature_pca_postprocess_sub = q1
    feature_pca_postprocess_div = (q99 - q1)
    del f_samples; del transformed
    vis_feature = (fmap.permute(0, 2, 3, 1).reshape(-1, fmap.shape[1]) - feature_pca_mean[None, :]) @ feature_pca_components.T
    vis_feature = (vis_feature - feature_pca_postprocess_sub) / feature_pca_postprocess_div
    vis_feature = vis_feature.clamp(0.0, 1.0).float().reshape((fmap.shape[2], fmap.shape[3], 3))# (H, W, 3)
    # Three ways to visualize
    # Way 1:
    # show_featmap = Image.fromarray((vis_feature.cpu().numpy()  * 255).astype(np.uint8)) # .save(os.path.join(outdir, outname + "_feature_vis.png"))
    # show_featmap.show(title=name)

    # Way 2:
    # cv2.imshow('name', (vis_feature.cpu().numpy()  * 255).astype('uint8'))
    # cv2.waitKey()

    # Way 3:
    # plt.imshow((vis_feature.cpu().numpy() * 255).astype(np.uint8))
    # plt.axis('off')  # Hide the axis
    # plt.title(name)
    # plt.show()

    return vis_feature

def apply_boolean_colormap(
    image: Bool[Tensor, "*bs 1"],
    true_color: Float[Tensor, "*bs 3"] =  torch.tensor([1.0, 1.0, 1.0]),
    false_color: Float[Tensor, "*bs 3"] =  torch.tensor([0.0, 0.0, 0.0]),
) -> Float[Tensor, "*bs 3"]:
    """Converts a depth image to color for easier analysis.

    Args:
        image: Boolean image.
        true_color: Color to use for True.
        false_color: Color to use for False.

    Returns:
        Colored boolean image
    """

    colored_image = torch.ones(image.shape[:-1] + (3,))
    colored_image[image[..., 0], :] = true_color
    colored_image[~image[..., 0], :] = false_color
    return colored_image

def apply_pca_colormap(image: Float[Tensor, "*bs dim"]) -> Float[Tensor, "*bs 3"]:
    """Convert feature image to 3-channel RGB via PCA. The first three principle
    components are used for the color channels, with outlier rejection per-channel

    Args:
        image: image of arbitrary vectors

    Returns:
        Tensor: Colored image
    """
    original_shape = image.shape
    image = image.view(-1, image.shape[-1])
    _, _, v = torch.pca_lowrank(image)
    image = torch.matmul(image, v[..., :3])
    d = torch.abs(image - torch.median(image, dim=0).values)
    mdev = torch.median(d, dim=0).values
    s = d / mdev
    m = 3.0  # this is a hyperparam controlling how many std dev outside for outliers
    rins = image[s[:, 0] < m, 0]
    gins = image[s[:, 1] < m, 1]
    bins = image[s[:, 2] < m, 2]

    image[:, 0] -= rins.min()
    image[:, 1] -= gins.min()
    image[:, 2] -= bins.min()

    image[:, 0] /= rins.max() - rins.min()
    image[:, 1] /= gins.max() - gins.min()
    image[:, 2] /= bins.max() - bins.min()

    image = torch.clamp(image, 0, 1)
    image_long = (image * 255).long()
    image_long_min = torch.min(image_long)
    image_long_max = torch.max(image_long)
    assert image_long_min >= 0, f"the min value is {image_long_min}"
    assert image_long_max <= 255, f"the max value is {image_long_max}"
    return image.view(*original_shape[:-1], 3)


def apply_float_colormap(image: Float[Tensor, "*bs 1"], colormap: str = "viridis") -> Float[Tensor, "*bs 3"]:
    """Convert single channel to a color image.

    Args:
        image: Single channel image.
        colormap: Colormap for image.

    Returns:
        Tensor: Colored image with colors in [0, 1]
    """
    if colormap == "default":
        colormap = "turbo"

    image = torch.nan_to_num(image, 0)
    if colormap == "gray":
        return image.repeat(1, 1, 3)
    image_long = (image * 255).long()
    image_long_min = torch.min(image_long)
    image_long_max = torch.max(image_long)
    assert image_long_min >= 0, f"the min value is {image_long_min}"
    assert image_long_max <= 255, f"the max value is {image_long_max}"
    return torch.tensor(matplotlib.colormaps[colormap].colors, device=image.device)[image_long[..., 0]]


def apply_colormap(
    image: Float[Tensor, "*bs channels"],
    colormap_options: ColormapOptions = ColormapOptions(),
    eps: float = 1e-9,
) -> Float[Tensor, "*bs 3"]:
    """
    Applies a colormap to a tensor image.
    If single channel, applies a colormap to the image.
    If 3 channel, treats the channels as RGB.
    If more than 3 channel, applies a PCA reduction on the dimensions to 3 channels

    Args:
        image: Input tensor image.
        eps: Epsilon value for numerical stability.

    Returns:
        Tensor with the colormap applied.
    """

    # default for rgb images
    if image.shape[-1] == 3:
        return image

    # rendering depth outputs
    if image.shape[-1] == 1 and torch.is_floating_point(image):
        output = image
        if colormap_options.normalize:
            output = output - torch.min(output)
            output = output / (torch.max(output) + eps)
        output = (
            output * (colormap_options.colormap_max - colormap_options.colormap_min) + colormap_options.colormap_min
        )
        output = torch.clip(output, 0, 1)
        if colormap_options.invert:
            output = 1 - output
        return apply_float_colormap(output, colormap=colormap_options.colormap)

    # rendering boolean outputs
    if image.dtype == torch.bool:
        return apply_boolean_colormap(image)

    if image.shape[-1] > 3:
        return apply_pca_colormap(image)

    raise NotImplementedError



def get_composited_relevancy_map(image, relevancy,  alpha=0.2):
    '''
    image (H, W, 3)
    relevancy: a list of (n_positive_embeds, H, W, 1). the length of the relevancy is number of positives for query.
    return: a list of (H, W, 3)
    '''
    outputs=[None for _ in range(len(relevancy))]
    for i in range(len(relevancy)):
        p_i = torch.clip(relevancy[i] - 0.5, 0, 1)
        p_i = p_i / (p_i.max() + 1e-6)
        outputs[i] = apply_colormap(p_i, ColormapOptions("turbo"))
        # mask = (relevancy[i] < 0.5).squeeze()
        mask = ((relevancy[i] < 0.5)|(p_i < p_i.median())).squeeze()

        # mask = (relevancy[i] < 0.5).squeeze()
        # mask = mask | (p_i < p_i[~mask].median()).squeeze()

        outputs[i][~mask, :] = (1-alpha)*(outputs[i][~mask, :]) + alpha*image[~mask, :]
        outputs[i][mask, :] = image[mask, :] # (H, W, 3)
        outputs[i] = outputs[i].cpu().numpy()

    return outputs # A list with len n_positive_embeds, [(H, W, 3)]



