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
import math
import numpy as np
import matplotlib.pyplot as plt
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from third_party.scene.gaussian_model import GaussianModel
from third_party.myutils.sh_utils import eval_sh
from lerf.lerf_fieldheadnames import LERFFieldHeadNames

from simple_diff_gaussian_rasterization import GaussianRasterizer as SimpleGaussianRasterizer

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, lerfmodel=None, bvl_feature_precomp=False, fmap_resolution=-1, fmap_render_radiithre=2):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    pc_lerfout: the lerfoutput of the pc (gaussians)
    bvl_feature_precomp: whether we should decode the position encodings to VL features with python firstly before the rendering process
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    if bvl_feature_precomp:
        if fmap_resolution > 0:
            raster_settings = GaussianRasterizationSettings(
                image_height=int(viewpoint_camera.image_height//fmap_resolution),
                image_width=int(viewpoint_camera.image_width//fmap_resolution),
                tanfovx=tanfovx,
                tanfovy=tanfovy,
                bg=bg_color,
                scale_modifier=scaling_modifier,
                viewmatrix=viewpoint_camera.world_view_transform,
                projmatrix=viewpoint_camera.full_proj_transform,
                sh_degree=pc.active_sh_degree,
                campos=viewpoint_camera.camera_center,
                prefiltered=False,
                debug=pipe.debug
            )

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None # Rocky: note that this can also be high-dimensional features besides color image
    colors_ex_precomp = None

    if override_color is None:
        if pipe.convert_SHs_python:
            assert False # Rocky: disable this one
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0) # Shape (N, 3)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    rendered_image = None
    radii_rendered_image = None
    if bvl_feature_precomp:
        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        # Disable the gradient backpropagation to  the geometry and shs attributes of the gassians
        with torch.no_grad():
            # colors_precomp = colors_precomp.detach() if cov3D_precomp is not None else None
            cov3D_precomp = cov3D_precomp.detach() if cov3D_precomp is not None else None
            shs = shs.detach() if shs is not None else None
            rasterizer = GaussianRasterizer(raster_settings=raster_settings)
            rendered_image, radii_rendered_image = rasterizer(
                means3D=means3D.detach(),
                means2D=means2D.detach(),
                shs=shs,
                colors_precomp=colors_precomp,
                opacities=opacity.detach(),
                scales=scales.detach(),
                rotations=rotations.detach(),
                cov3D_precomp=cov3D_precomp)
    else:
        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        # Training the geometry and shs attributes of the gassians
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        rendered_image, radii_rendered_image = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp)


    ## Render the VL feature map
    rendered_featmap = None
    rendered_featmap_ex = None


    if bvl_feature_precomp:
        # You may adjust these hyperparameters
        valid_gaussian_mask = (opacity > 0.25).squeeze(-1).detach() & (radii_rendered_image > fmap_render_radiithre).detach() # By default
        # while ((valid_gaussian_mask.sum() < 1000) and (radii_threshold>0)):
        #     print(f"=============== For radii_threshold {radii_threshold}, valid_gaussian_mask.sum() is less than 100! We decrease it by 2!"
        #           f" The valid portion is {valid_gaussian_mask.sum().item()}/{valid_gaussian_mask.shape[0]}!")
        #     radii_threshold = radii_threshold - 2
        #     valid_gaussian_mask = (radii_rendered_image > radii_threshold).detach()
        # print(f" The valid portion is {valid_gaussian_mask.sum().item()/valid_gaussian_mask.shape[0]}, {valid_gaussian_mask.sum().item()}/{valid_gaussian_mask.shape[0]}!")

        # assert (valid_gaussian_mask.sum() > 1000), f" The valid portion is too low: {valid_gaussian_mask.sum().item()}/{valid_gaussian_mask.shape[0]}!"

        clip_scales = radii_rendered_image[valid_gaussian_mask].detach() # TODO: the value of clip_scales needs to be checked.
        lerf_field_outputs = lerfmodel(pc, clip_scales, valid_gaussian_mask) # dict with keys "LERFFieldHeadNames.HASHGRID, LERFFieldHeadNames.CLIP, LERFFieldHeadNames.DINO"

        feature_dinomap_precomp = lerf_field_outputs[LERFFieldHeadNames.DINO]
        feature_clipmap_precomp = lerf_field_outputs[LERFFieldHeadNames.CLIP]
        print(feature_clipmap_precomp.shape, feature_dinomap_precomp.shape, means3D.shape)
        simple_rasterizer = SimpleGaussianRasterizer(raster_settings=raster_settings)
        cov3D_precomp_ = cov3D_precomp_[valid_gaussian_mask].detach() if cov3D_precomp is not None else None
        rendered_featmap, rendered_featmap_ex, radii_rendered_featmap = simple_rasterizer(
            means3D=means3D[valid_gaussian_mask].detach(),
            means2D=means2D[valid_gaussian_mask].detach(),
            shs=None,
            colors_precomp=feature_dinomap_precomp.float(),
            colors_ex_precomp=feature_clipmap_precomp.float(),
            opacities=opacity[valid_gaussian_mask].detach(),
            scales=scales[valid_gaussian_mask].detach(),
            rotations=rotations[valid_gaussian_mask].detach(),
            cov3D_precomp=cov3D_precomp_)


    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii_rendered_image > 0,
            "radii": radii_rendered_image,
            "rendered_featmap": rendered_featmap,
            "rendered_featmap_ex": rendered_featmap_ex,}
