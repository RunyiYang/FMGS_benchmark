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

import typing

import torch
from lerf.data.utils.dino_extractor import ViTExtractor
from lerf.data.utils.feature_dataloader import FeatureDataloader
from tqdm import tqdm
import torch.nn.functional as F
import torchvision.transforms as transforms
from myutils.vis_lerf_utils import get_shown_feat_map
import matplotlib.pyplot as plt

class DinoDataloader(FeatureDataloader):
    dino_model_type = "dino_vits8"
    dino_stride = 8

    # # # For dinov2, use this:
    # dino_model_type = "dinov2_vitb14" # "dinov2_vitb14", "dinov2_vitb14_reg"
    # dino_stride = 14 # 14, 7

    dino_load_size = 500 # 500 -> 720 # This will be the smaller one of the rescaled height, or width
    dino_layer = 11
    dino_facet = "key"
    dino_bin = False

    def __init__(
        self,
        cfg: dict,
        device: torch.device,
        image_list: torch.Tensor,
        cache_path: str = None,
    ):
        assert "image_shape" in cfg
        super().__init__(cfg, device, image_list, cache_path)
        del image_list

        # # Do distillation-preprocessing as noted in N3F:
        # # The features are then L2-normalized and reduced with PCA to 64 dimensions before distillation.
        # data_shape = self.data.shape
        # self.data = self.data / self.data.norm(dim=-1, keepdim=True)
        # self.data = torch.pca_lowrank(self.data.reshape(-1, data_shape[-1]), q=64)[0].reshape((*data_shape[:-1], 64))

    def create(self, image_list):
        extractor = ViTExtractor(self.dino_model_type, self.dino_stride)
        preproc_image_lst = extractor.preprocess(image_list, self.dino_load_size)[0].to(self.device)

        dino_embeds = []
        for image in tqdm(preproc_image_lst, desc="dino", total=len(image_list), leave=False):
            # image nees to be resized w.s.t. H, W are divisible by dino_stride
            if "dinov2" in self.dino_model_type:
                # print("--- resize the input image with shape to shape: ", image.shape, ((image.shape[1]//self.dino_stride)*self.dino_stride, (image.shape[2]//self.dino_stride)*self.dino_stride))
                deno = 14
                image = transforms.Resize((
                    (image.shape[1]//deno)*deno,
                    (image.shape[2]//deno)*deno,
                    ))(image)


            with torch.no_grad():
                descriptors = extractor.extract_descriptors(
                    image.unsqueeze(0),
                    [self.dino_layer],
                    self.dino_facet,
                    self.dino_bin,
                )
            descriptors = descriptors.reshape(extractor.num_patches[0], extractor.num_patches[1], -1)
            dino_embeds.append(descriptors.cpu().detach()) # (H, W, C)

            # ## Visualize the dino feature
            # render_fmap_show = get_shown_feat_map(descriptors.permute(2,0,1).contiguous(), "RenderFeat_scaled").cpu().numpy()
            # print("descriptors.shape, render_fmap_show.shape: ", descriptors.shape, render_fmap_show.shape)
            # # plt.imsave(self.cache_path + f'/dino_tmp.png', render_fmap_show)
            # # Display the image using Matplotlib
            # plt.imshow(render_fmap_show)
            # plt.axis('off')  # Hide the axis
            # plt.title('fmap_dino')
            # plt.show()

        self.data = torch.stack(dino_embeds, dim=0) # (N, H, W, C) tmp (177, 62, 84, 384)

    def imgpoints_call(self, img_points):
        # img_points: (B, 3) # (img_ind, x, y)
        img_scale = (
            self.data.shape[1] / self.cfg["image_shape"][0],
            self.data.shape[2] / self.cfg["image_shape"][1],
        )
        x_ind, y_ind = (img_points[:, 1] * img_scale[0]).long(), (img_points[:, 2] * img_scale[1]).long()
        return (self.data[img_points[:, 0].long(), x_ind, y_ind]).to(self.device)

    def img_call(self, img_ind):
        # img_ind is a scalar with int type
        dino_feat_map = (self.data[img_ind]).clone().to(self.device) # (H, W, C) tmp (62, 84, 384)
        dino_feat_map = dino_feat_map.permute(2,0,1).contiguous() # (C, H, W)  tmp (384, 62, 84)
        # dino_feat_map = F.interpolate(dino_feat_map.unsqueeze(dim=0), size=(self.cfg["image_shape"][0], self.cfg["image_shape"][1]), mode="bilinear") # (1, C, Hl, Wl) tmp (1, 384, 738, 994)
        return dino_feat_map