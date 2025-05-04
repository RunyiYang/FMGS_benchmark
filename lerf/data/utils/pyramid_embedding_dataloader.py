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

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch.nn.functional as F

import numpy as np
import torch
from lerf.data.utils.feature_dataloader import FeatureDataloader
from lerf.data.utils.patch_embedding_dataloader import PatchEmbeddingDataloader
from lerf.encoders.image_encoder import BaseImageEncoder
from tqdm import tqdm

from myutils.vis_lerf_utils import get_shown_feat_map, util_add_row
def visualize_pyramid_featmap(fmap_list):
    H, W = 738, 994
    shown_fmaps = []
    for fmap in fmap_list:
        shown_fmaps.append(get_shown_feat_map(fmap)) # fmap (C, H, W), out (H, W, 3)
    merged_shown_fmaps = util_add_row(None, shown_fmaps)
    plt.imshow((merged_shown_fmaps*255).astype(np.uint8))
    plt.axis('off')
    plt.title("Fmaps_HybirdFmap")
    plt.show()




class PyramidEmbeddingDataloader(FeatureDataloader):
    def __init__(
        self,
        cfg: dict,
        device: torch.device,
        model: BaseImageEncoder,
        image_list: torch.Tensor = None,
        cache_path: str = None,
    ):
        assert "tile_size_range" in cfg
        assert "tile_size_res" in cfg
        assert "stride_scaler" in cfg
        assert "image_shape" in cfg
        assert "model_name" in cfg

        self.tile_sizes = torch.linspace(*cfg["tile_size_range"], cfg["tile_size_res"]).to(device)
        self.strider_scaler_list = [self._stride_scaler(tr.item(), cfg["stride_scaler"]) for tr in self.tile_sizes]

        self.model = model
        self.embed_size = self.model.embedding_dim
        self.data_dict = {}
        super().__init__(cfg, device, image_list, cache_path)
        del image_list

    def _stride_scaler(self, tile_ratio, stride_scaler):
        return np.interp(tile_ratio, [0.05, 0.15], [1.0, stride_scaler])

    def load(self):
        # don't create anything, PatchEmbeddingDataloader will create itself
        cache_info_path = self.cache_path.with_suffix(".info")

        # check if cache exists
        if not cache_info_path.exists():
            raise FileNotFoundError

        # if config is different, remove all cached content
        with open(cache_info_path, "r") as f:
            cfg = json.loads(f.read())
        if cfg != self.cfg:
            for f in os.listdir(self.cache_path):
                os.remove(os.path.join(self.cache_path, f))
            raise ValueError("Config mismatch")

        raise FileNotFoundError  # trigger create

    def create(self, image_list):
        os.makedirs(self.cache_path, exist_ok=True)
        for i, tr in enumerate(tqdm(self.tile_sizes, desc="Scales")):
            stride_scaler = self.strider_scaler_list[i]
            self.data_dict[i] = PatchEmbeddingDataloader(
                cfg={
                    "tile_ratio": tr.item(),
                    "stride_ratio": stride_scaler,
                    "image_shape": self.cfg["image_shape"],
                    "model_name": self.cfg["model_name"],
                },
                device=self.device,
                model=self.model,
                image_list=image_list,
                cache_path=Path(f"{self.cache_path}/level_{i}.npy"),
            )

    def save(self):
        cache_info_path = self.cache_path.with_suffix(".info")
        with open(cache_info_path, "w") as f:
            f.write(json.dumps(self.cfg))
        # don't save anything, PatchEmbeddingDataloader will save itself
        pass

    def _random_scales(self, img_points):
        # img_points: (B, 3) # (img_ind, x, y)
        # return: (B, 512), some random scale (between 0, 1)
        img_points = img_points.to(self.device)
        random_scale_bin = torch.randint(self.tile_sizes.shape[0] - 1, size=(img_points.shape[0],), device=self.device)
        random_scale_weight = torch.rand(img_points.shape[0], dtype=torch.float16, device=self.device)

        stepsize = (self.tile_sizes[1] - self.tile_sizes[0]) / (self.tile_sizes[-1] - self.tile_sizes[0])

        bottom_interp = torch.zeros((img_points.shape[0], self.embed_size), dtype=torch.float16, device=self.device)
        top_interp = torch.zeros((img_points.shape[0], self.embed_size), dtype=torch.float16, device=self.device)

        for i in range(len(self.tile_sizes) - 1):
            ids = img_points[random_scale_bin == i]
            bottom_interp[random_scale_bin == i] = self.data_dict[i](ids)
            top_interp[random_scale_bin == i] = self.data_dict[i + 1](ids)

        return (
            torch.lerp(bottom_interp, top_interp, random_scale_weight[..., None]),
            (random_scale_bin * stepsize + random_scale_weight * stepsize)[..., None],
        )

    def _uniform_scales(self, img_points, scale): # Actually, this is not used.
        # img_points: (B, 3) # (img_ind, x, y)
        scale_bin = torch.floor(
            (scale - self.tile_sizes[0]) / (self.tile_sizes[-1] - self.tile_sizes[0]) * (self.tile_sizes.shape[0] - 1)
        ).to(torch.int64)
        scale_weight = (scale - self.tile_sizes[scale_bin]) / (
            self.tile_sizes[scale_bin + 1] - self.tile_sizes[scale_bin]
        )
        interp_lst = torch.stack([interp(img_points) for interp in self.data_dict.values()])
        point_inds = torch.arange(img_points.shape[0])
        interp = torch.lerp(
            interp_lst[scale_bin, point_inds],
            interp_lst[scale_bin + 1, point_inds],
            torch.Tensor([scale_weight]).half().to(self.device)[..., None],
        )
        return interp / interp.norm(dim=-1, keepdim=True), scale

    def imgpoints_call(self, img_points, scale=None):
        # img_points: (B, 3) # (img_ind, x, y)
        if scale is None:
            return self._random_scales(img_points)
        else:
            return self._uniform_scales(img_points, scale)

    def img_call(self, img_ind):
        # img_ind is a scalar, output (HxW, 512)
        interval = self.data_dict[0].stride - 3 # 32 = 35 - 3
        assert interval > 0
        img_coords = torch.stack(torch.meshgrid(torch.arange(0, self.cfg["image_shape"][0], interval), torch.arange(0, self.cfg["image_shape"][1], interval)),dim=-1).to(self.device) # (Hs, Ws, 2)
        Hs = img_coords.shape[0]; Ws = img_coords.shape[1] # Hs = 24, Ws = 32
        img_coords = img_coords.view(-1, 2) # (HxW, 2)
        img_points = torch.cat((img_ind*torch.ones(img_coords.shape[0],device=self.device).unsqueeze(dim=-1), img_coords), dim=1) # (HxW, 3)
        img_points = img_points.long()

        # Loop through the features at all buffered levels, and compute the mean
        feat_at_allsizes = []
        for scale_i, tr in enumerate(self.tile_sizes):
            # Note: regarding the self.data_dict, for an image with shape 738x994
            # self.data_dict[0], the highest resolution  (177, 23, 30, 512]), stride is 35
            # self.data_dict[-1], the lowest resolution  (177, 14, 19, 512]), stride is 57
            feat_at_allsizes.append(self.data_dict[scale_i](img_points)) # [(HxW, 512)]

        hybrid_feat_map = torch.stack(feat_at_allsizes, dim=0).mean(dim=0) # (HxW, 512)
        hybrid_feat_map = hybrid_feat_map/hybrid_feat_map.norm(dim=-1, keepdim=True)

        hybrid_feat_map = hybrid_feat_map.view(Hs, Ws, -1) # tmp (24, 32, 512)
        hybrid_feat_map = hybrid_feat_map.permute(2, 0, 1).contiguous()  # tmp (512, 24, 32)


        # ## Visualize the featre map at multple scales
        # fmaps = torch.stack(feat_at_allsizes, dim=0).view( len(feat_at_allsizes), Hs, Ws, -1) # (N, H, W, 512)
        # fmaps_list = list(fmaps.permute(0, 3, 1, 2).contiguous().unbind(dim=0))
        # fmaps_list.append(hybrid_feat_map)
        # visualize_pyramid_featmap(fmaps_list)

        return hybrid_feat_map.to(self.device)