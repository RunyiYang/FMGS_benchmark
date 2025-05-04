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
from torch.autograd import Variable
from math import exp

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def clip_loss(network_output, gt): # Actually, this is L1 loss
    # TODO: Debug this
    unreduced_clip = torch.nn.functional.huber_loss(network_output, gt, delta=1.25, reduction="none")
    loss = unreduced_clip.sum(dim=-1).nanmean()
    return loss if (not loss.isnan()) else torch.tensor(0.0, dtype=network_output.dtype, device=network_output.device)

def dino_loss(network_output, gt):
    unreduced_dino = torch.nn.functional.mse_loss(network_output, gt, reduction="none")
    loss = unreduced_dino.sum(dim=-1).nanmean()
    return loss if (not loss.isnan()) else torch.tensor(0.0, dtype=network_output.dtype, device=network_output.device)

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


# The dot product similarity
def dotp_withneighbors(fmap, window_size=3):
    """
    fmap: CxHxW
    """
    C,H,W = fmap.shape
    half_window = window_size // 2
    fmap_padded = F.pad(fmap, (half_window, half_window, half_window, half_window), mode='reflect')

    dotp_fmap = torch.zeros((window_size*window_size-1, H, W), dtype=fmap.dtype, device=fmap.device)
    dotp_ch = 0
    for i in range(-half_window, half_window + 1):
        for j in range(-half_window, half_window + 1):
            if i == 0 and j == 0:
                continue
            neighbor_fmap = fmap_padded[:, half_window + i: half_window + i + H, half_window + j: half_window + j +W]
            deno = torch.clamp(fmap.norm(dim=0)*neighbor_fmap.norm(dim=0), 1e-6) # (H, W)
            dotp_fmap[dotp_ch] = (fmap*neighbor_fmap).sum(dim=0)/deno # (H, W)
            dotp_ch+=1
    return dotp_fmap

def dotp_sim(fmap, fmap_ref, window_size=3):
    """
    fmap, fmap_ref: CxHxW
    """
    dotp_fmap_ref = dotp_withneighbors(fmap_ref.detach()).detach() # (window_size*window_size-1, H, W)
    dotp_fmap = dotp_withneighbors(fmap) # (window_size*window_size-1, H, W)
    return l1_loss(dotp_fmap, dotp_fmap_ref)