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

import os
import torch
from random import randint
from myutils.loss_utils import l1_loss, ssim, clip_loss, dino_loss, dotp_sim
from gaussian_renderer import render, network_gui
import sys
from third_party.scene import Scene, GaussianModel
from third_party.myutils.general_utils import safe_state
import uuid
from tqdm import tqdm
from third_party.myutils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from third_party.arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

from lerf.lerf import LERFModel
from lerf.data.lerf_datamanager import  LERFFeatManager
from lerf.encoders.openclip_encoder import OpenCLIPNetworkConfig, OpenCLIPNetwork

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import torch.nn.functional as F
from myutils.vis_lerf_utils import get_shown_feat_map, get_composited_relevancy_map
from myutils.lerf_optimizer_scedulers import ExponentialDecaySchedulerConfig,  ExponentialDecayScheduler

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, opt_vlrenderfeat_from, args):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)

    lerf_model = None
    lerf_featmap_manger = None
    lerf_optimizer = None
    lerf_opt_lr_scheduler = None
    lerf_image_encoder = None
    b_vlrenderfeat = False
    if opt_vlrenderfeat_from > 0: # We initalize a lerfModel.
        b_vlrenderfeat = True
        lerf_model = LERFModel()
        lerf_model.cuda()
        lerf_model.train()
        print('Number of model parameters in Lerf: {}'.format(sum([p.data.nelement() for p in lerf_model.parameters()])))
        print('Number of optmizable model parameters in Lerf: {}'.format(sum([p.data.nelement() for p in lerf_model.parameters() if p.requires_grad])))

        lerf_optimizer = torch.optim.RAdam(lerf_model.parameters(), lr=args.fmap_lr, eps=1e-15, betas=(0.9, 0.999), weight_decay=1e-9)
        lerf_opt_scheduler_cfg = ExponentialDecaySchedulerConfig(lr_final=2e-3, max_steps=10000)


    # gaussians = GaussianModel(dataset.sh_degree, intermed_vlfeat_dim = lerf_model.intermed_vlfeat_dim) # This is for the Vl embedding attached gaussians.
    gaussians = GaussianModel(dataset.sh_degree)  # This is for vanilla gaussians.
    scene = Scene(dataset, gaussians, dataformat=args.dataformat)
    gaussians.training_setup(opt)
    scene_id = scene.model_path.split("/")[-2]
    # gaussians.load_ply(os.path.join(scene.model_path, 
                                                        #    "../../../mcmc_3dgs/", scene_id, "ckpts/point_cloud_9000.ply"))
    if checkpoint:
        print(f"Loading checkpoint from: {checkpoint}")
        loaded_statedict = torch.load(checkpoint)
        if isinstance(loaded_statedict, tuple):
            gaussians_model_params,first_iter = loaded_statedict
        else:
            gaussians_model_params = loaded_statedict['gaussians']
            first_iter = loaded_statedict['iteration']
        gaussians.restore(gaussians_model_params, opt)

        # Restore the lerf relevant states
        if b_vlrenderfeat and 'lerf_model' in loaded_statedict:
            lerf_model.load_state_dict(loaded_statedict['lerf_model'], strict=False)
            lerf_optimizer.param_groups[0]['initial_lr'] = loaded_statedict['lerf_optimizer']['param_groups'][0]['lr']
            lerf_optimizer.param_groups[0]['lr'] = loaded_statedict['lerf_optimizer']['param_groups'][0]['lr']

    if opt_vlrenderfeat_from > 0:
        LERF_LR_SCHEDULER = ExponentialDecayScheduler()
        lerf_opt_lr_scheduler = LERF_LR_SCHEDULER.get_scheduler(lerf_optimizer, lr_init=lerf_optimizer.param_groups[0]['lr'], config_=lerf_opt_scheduler_cfg)

    if b_vlrenderfeat:
        dataname = os.path.basename(dataset.source_path)
        lerf_image_encoder_config = OpenCLIPNetworkConfig(clip_model_type="ViT-B-16", clip_model_pretrained="laion2b_s34b_b88k", clip_n_dims=512)
        # print("for debug, lerf_image_encoder_config.negatives, clip_model_type: ", lerf_image_encoder_config.negatives, lerf_image_encoder_config.clip_model_pretrained, lerf_image_encoder_config.clip_model_type)
        lerf_image_encoder = OpenCLIPNetwork(lerf_image_encoder_config)
        lerf_featmap_manger = LERFFeatManager(dataname, scene, lerf_image_encoder, device="cuda")

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    num_viewpoints = None
    viewpoints_opts = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    # first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if (iteration in checkpoint_iterations):
            print("\n[ITER {}] Saving Checkpoint at {}".format(iteration, scene.model_path + "/chkpnt" + str(iteration) + ".pth"))
            if not b_vlrenderfeat:
                torch.save({'gaussians': gaussians.capture(),
                            'iteration': iteration}, scene.model_path + "/chkpnt" + str(iteration) + ".pth")
            else:
                torch.save({'gaussians': gaussians.capture(),
                            'iteration': iteration,
                            'lerf_model': lerf_model.state_dict(),
                            'lerf_optimizer': lerf_optimizer.state_dict()}, scene.model_path + "/chkpnt" + str(iteration) + ".pth")
            torch.cuda.empty_cache()
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()
        gaussians.update_learning_rate(iteration)

        bvl_feature_precomp = b_vlrenderfeat and (iteration > opt_vlrenderfeat_from - 1)
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if (not bvl_feature_precomp) and (iteration % 1000 == 0):
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if (not viewpoint_stack) or (not viewpoints_opts):
            viewpoint_stack = scene.getTrainCameras().copy()
            num_viewpoints = len(viewpoint_stack)
            viewpoints_opts = torch.arange(num_viewpoints).tolist()
            viewpoint_stack = dict(zip(viewpoints_opts, viewpoint_stack))

        viewpoint_pop = randint(0, len(viewpoints_opts) - 1)
        viewpoint_indx = viewpoints_opts.pop(viewpoint_pop)
        viewpoint_cam = viewpoint_stack.pop(viewpoint_indx) # .pop(viewpoint_indx) [viewpoint_indx]

        fmap_resolution = args.fmap_resolution
        # # We reset the fmap_resolution if it is testing iteration for a better visualization
        # if iteration in testing_iterations:
        #     fmap_resolution = -1


        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        if bvl_feature_precomp:
            if iteration == opt_vlrenderfeat_from:
                gaussians.optimizer.zero_grad()
                for param_group in gaussians.optimizer.param_groups:
                    for param in param_group['params']:
                        param.requires_grad = False

        # Note that, in my current implementation, when bvl_feature_precomp is enabled, the geometry and shs attributes of the gaussians are frozen. While, only the params related to lerf_model is optimized.
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, lerfmodel=lerf_model, bvl_feature_precomp=bvl_feature_precomp, fmap_resolution=fmap_resolution, fmap_render_radiithre=args.fmap_render_radiithre)
        rendered_image, viewspace_point_tensor, visibility_filter, radii, rendered_featmap_dino, rendered_featmap_clip = (render_pkg["render"], render_pkg["viewspace_points"],
                                                                                                   render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["rendered_featmap"], render_pkg["rendered_featmap_ex"])

        gt_image = viewpoint_cam.original_image.cuda() # (C, H, W) tmp (3, 738, 994)
        if bvl_feature_precomp and fmap_resolution>0: # Resize the image to 1/4 resolution
            H_gtimg, W_gtimg = gt_image.shape[-2], gt_image.shape[-1]
            gt_image = F.interpolate(gt_image.unsqueeze(dim=0), size=(int(H_gtimg//fmap_resolution), int(W_gtimg//fmap_resolution)),
                                                 mode="bilinear").squeeze(dim=0)
        H_gtimg, W_gtimg = gt_image.shape[-2], gt_image.shape[-1]

        L_dino_loss = torch.tensor(0.0)
        L_clip_loss = torch.tensor(0.0)
        L_dotp_sim_loss = torch.tensor(0.0)
        if bvl_feature_precomp:
            ## Visualize the gt_image
            # plt.imshow(gt_image.permute(1, 2, 0).contiguous().cpu().detach().numpy())
            # plt.title("GtImg"); plt.axis('off'); plt.show()
            # print("rendered_featmap.shape: ", rendered_featmap_dino.shape) # tmp [384, 738, 994]
            # print("viewpoint_indx, gt_image.shape: ", viewpoint_indx, gt_image.shape)

            # Extract the featrue map via fundation models
            dino_feat_map, clip_feat_map = lerf_featmap_manger(viewpoint_indx) # (C-384, Hs0, Ws0) tmp (384, 62, 84) # (C-512, Hs0, Ws0), tmp (512, 24, 32)
            # TODO: Visualize the generated hybrid feat map, and match it with a heuristic text prompt
            # # print("Shape of dino_feat_map, clip_feat_map: ", dino_feat_map.shape, clip_feat_map.shape)
            # # dino_feat_map_scaled = F.interpolate(dino_feat_map.unsqueeze(dim=0), size=(H_gtimg, W_gtimg), mode="bilinear").squeeze(dim=0)
            # clip_feat_map_scaled = F.interpolate(clip_feat_map.unsqueeze(dim=0), size=(H_gtimg, W_gtimg), mode="bilinear").squeeze(dim=0)
            # pos_prob_maps = lerf_model.get_relevancy_img(clip_feat_map_scaled, lerf_image_encoder) # (n_positive_embeds, H, W, 1)
            # composited_relevancy_maps = get_composited_relevancy_map(gt_image.permute(1,2,0).contiguous(), pos_prob_maps) # n_positive_embeds [(H, W, 1)]

            # ## Visualize the relevancy map
            # # plt.imshow((composited_relevancy_maps[0] * 255).astype(np.uint8))
            # # plt.title("relevancy_flower"); plt.axis('off');plt.show()
            # fig, ax = plt.subplots()
            # img_ax = ax.imshow((composited_relevancy_maps[0] * 255).astype(np.uint8), cmap='turbo')
            # plt.axis('off')
            # divider = make_axes_locatable(ax)
            # cax = divider.append_axes('right', size='5%', pad=0.05)
            # cbar = plt.colorbar(img_ax, cax=cax, orientation='vertical', ticks=[0, 100, 255])
            # cbar.set_label('Relevancy')
            # plt.show()

            # ## Visualize the feature map
            # DinoFeat_scaled = get_shown_feat_map(dino_feat_map_scaled, "DinoFeat_scaled").cpu().numpy()
            # ClipFeat_scaled = get_shown_feat_map(clip_feat_map_scaled, "ClipFeat_scaled").cpu().numpy()
            # plt.imshow((DinoFeat_scaled * 255).astype(np.uint8))
            # plt.title("DinoFeat"); plt.axis('off');plt.show()
            # plt.imshow((ClipFeat_scaled * 255).astype(np.uint8))
            # plt.title("ClipFeat"); plt.axis('off');plt.show()


            H_dino, W_dino = dino_feat_map.shape[1], dino_feat_map.shape[2]
            H_clip, W_clip = clip_feat_map.shape[1], clip_feat_map.shape[2]

            ### Scale down both the rendered dino and clip feature maps to the raw dino feature map resolution, and formulate the loss
            rendered_featmap_dino_eval = F.interpolate(rendered_featmap_dino.unsqueeze(dim=0), size=(H_dino, W_dino),
                                                 mode="bilinear").squeeze(dim=0)
            rendered_featmap_clip_eval = F.interpolate(rendered_featmap_clip.unsqueeze(dim=0), size=(H_dino, W_dino),
                                                 mode="bilinear").squeeze(dim=0)
            dino_feat_map_gt = dino_feat_map
            clip_feat_map_gt = F.interpolate(clip_feat_map.unsqueeze(dim=0), size=(H_dino, W_dino),
                                                 mode="bilinear").squeeze(dim=0)

            # ### Scale up raw dino and clip feature maps to the image resolution, and formulate the loss
            # dino_feat_map_gt = F.interpolate(dino_feat_map.unsqueeze(dim=0), size=(H_gtimg, W_gtimg),
            #                                      mode="bilinear").squeeze(dim=0)
            # clip_feat_map_gt = F.interpolate(clip_feat_map.unsqueeze(dim=0), size=(H_gtimg, W_gtimg),
            #                                      mode="bilinear").squeeze(dim=0)
            # rendered_featmap_dino_eval = rendered_featmap_dino
            # rendered_featmap_clip_eval = rendered_featmap_clip


            L_dino_loss = dino_loss(rendered_featmap_dino_eval, dino_feat_map_gt.float())
            L_clip_loss = clip_loss(rendered_featmap_clip_eval, clip_feat_map_gt.float())
            if args.dotp_simloss_w>0:
                L_dotp_sim_loss = dotp_sim(rendered_featmap_clip_eval, rendered_featmap_dino_eval)

            if (L_dino_loss < 1e-6 or L_clip_loss < 1e-6):
                print(f"~~ Warning!  L_dino_loss = None, rendered_featmap_dino_eval {rendered_featmap_dino_eval.mean()}, \n dino_feat_map_gt {dino_feat_map_gt.mean()}")
                print(f"~~ Warning!  L_clip_loss = None, rendered_featmap_clip_eval {rendered_featmap_clip_eval.mean()}, \n clip_feat_map_gt {clip_feat_map_gt.mean()}")
                lerf_optimizer.zero_grad()
                lerf_opt_lr_scheduler.step()
                print(f"~~ Warning! We encounter Nan loss, thus we skip this iteration!")
                continue

            ## Visualzie the rendered feature map
            if iteration in testing_iterations:
                torch.cuda.empty_cache()
                img_save_path = scene.model_path + "/vis_test_iteration"
                os.makedirs(img_save_path, exist_ok=True)
                print(f"Saving visualization in folder {img_save_path}")

                ## Visualize the dino feature
                render_fmap_show = get_shown_feat_map(rendered_featmap_dino_eval, "RenderFeat_scaled").cpu().numpy()
                gt_fmap_show = get_shown_feat_map(dino_feat_map_gt, "GtFeat_scaled").cpu().numpy() # (H, W, 3)
                show_img = np.hstack((render_fmap_show, gt_fmap_show))
                plt.imsave(img_save_path + f'/{iteration}_dino_{viewpoint_indx:06d}.png', show_img)
                # # Display the image using Matplotlib
                # plt.imshow(show_img)
                # plt.axis('off')  # Hide the axis
                # plt.title('fmap_rendered_dino & fmap_gt')
                # plt.show()

                ## Visualize the Clip feature
                render_fmap_show = get_shown_feat_map(rendered_featmap_clip_eval, "RenderFeat_scaled").cpu().numpy()
                gt_fmap_show = get_shown_feat_map(clip_feat_map_gt, "GtFeat_scaled").cpu().numpy()  # (H, W, 3)
                show_img = np.hstack((render_fmap_show, gt_fmap_show))
                plt.imsave(img_save_path + f'/{iteration}_clip_{viewpoint_indx:06d}.png', show_img)
                # # Display the image using Matplotlib
                # plt.imshow(show_img)
                # plt.axis('off')  # Hide the axis
                # plt.title('fmap_rendered_clip & fmap_gt')
                # plt.show()

                ## Visualize the relevancy map for rendered CLIP at the image resolution
                # rendered_clip_feat_map_scaled = F.interpolate(rendered_featmap_clip_scaled.unsqueeze(dim=0), size=(H_gtimg, W_gtimg), mode="bilinear").squeeze(dim=0)
                pos_prob_maps_rendered = lerf_model.get_relevancy_img(rendered_featmap_clip, lerf_image_encoder) # (n_positive_embeds, H, W, 1)
                rendered_composited_relevancy_maps = get_composited_relevancy_map(gt_image.permute(1,2,0).contiguous(), pos_prob_maps_rendered) # n_positive_embeds [(H, W, 1)]

                clip_feat_map_scaled = F.interpolate(clip_feat_map.unsqueeze(dim=0), size=(H_gtimg, W_gtimg), mode="bilinear").squeeze(dim=0) # (C, H, W)
                pos_prob_maps = lerf_model.get_relevancy_img(clip_feat_map_scaled, lerf_image_encoder) # (n_positive_embeds, H, W, 1)
                composited_relevancy_maps = get_composited_relevancy_map(gt_image.permute(1,2,0).contiguous(), pos_prob_maps) # n_positive_embeds [(H, W, 1)]

                shown_relevancy_maps = np.hstack((rendered_composited_relevancy_maps[0], composited_relevancy_maps[0]))
                fig, ax = plt.subplots()
                img_ax = ax.imshow((shown_relevancy_maps * 255).astype(np.uint8), cmap='turbo')
                plt.axis('off')
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                cbar = plt.colorbar(img_ax, cax=cax, orientation='vertical', ticks=[0, 100, 255])
                cbar.set_label('Relevancy_rendered&gt')
                # plt.show()
                plt.savefig(img_save_path + f'/{iteration}_relevancy_{viewpoint_indx:06d}.png', dpi=1000)
                print("iteration, L_dino_loss and L_clip_loss loss for the rendered fmap: ", iteration, L_dino_loss.item(), L_clip_loss.item())
                del render_fmap_show; del gt_fmap_show; del clip_feat_map_scaled; del rendered_composited_relevancy_maps; del pos_prob_maps_rendered; del pos_prob_maps;
                torch.cuda.empty_cache()


        # Reconstruction Loss
        Ll1_rec = l1_loss(rendered_image, gt_image)

        ######  Visualization  ######
        ## Visualzie the RGB image
        # if iteration in testing_iterations:
        #     image_to_show = rendered_image.cpu().detach().numpy()
        #     image_gt_to_show = gt_image.cpu().detach().numpy()
        #     show_img = np.hstack((image_to_show, image_gt_to_show))
        #     # Display the image using Matplotlib
        #     plt.imshow(show_img.transpose(1, 2, 0))  # Transpose for the correct shape
        #     plt.axis('off')  # Hide the axis
        #     plt.title('image_rendered & image_gt')
        #     plt.show()
        #     print("iteration, L1 loss for the rendered image: ",  iteration, Ll1_rec)

        loss = 0.0
        if bvl_feature_precomp:
            w_cliploss_lamda = opt.lambda_clip if (iteration - opt_vlrenderfeat_from > 20) else 0.2*opt.lambda_clip
            # print(f"w_cliploss_lamda {w_cliploss_lamda}, and opt.lambda_clip {opt.lambda_clip}")
            loss = (1.0 - w_cliploss_lamda) * L_dino_loss + w_cliploss_lamda * L_clip_loss
            if args.dotp_simloss_w > 0:
                loss += args.dotp_simloss_w*L_dotp_sim_loss
                # print(f"In training, iteration {iteration}, L_dino_loss: {L_dino_loss.item()}, L_clip_loss: {L_clip_loss.item()}, L_dotp_sim_loss: {L_dotp_sim_loss.item()}")
            else:
                pass
                # print(f"In training, iteration {iteration}, L_dino_loss: {L_dino_loss.item()}, L_clip_loss: {L_clip_loss.item()}")

            # assert (not loss.isnan().any()), f"The loss is Nan!"
            loss.backward()

            # Calculate the gradient norm
            gradient_norm = torch.stack([torch.norm(p.grad) for p in lerf_model.parameters()])
            if not gradient_norm.mean().isnan():
                torch.nn.utils.clip_grad_norm_(lerf_model.parameters(), 1.0)
                lerf_optimizer.step()
            else:
                print("~~~~~~~~~ Gradient_norm contains Nan, we will skip! gradient_norm: ",  gradient_norm)

            lerf_optimizer.zero_grad()
            lerf_opt_lr_scheduler.step()
        else: # Reconstruciton loss
            loss = (1.0 - opt.lambda_dssim) * Ll1_rec + opt.lambda_dssim * (1.0 - ssim(rendered_image, gt_image))
            loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if (iteration < opt.densify_until_iter) and (not bvl_feature_precomp):
                # Keep track of max radii in image-space for pruning

                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if (not bvl_feature_precomp) and (iteration < opt.iterations):
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            # if (iteration in checkpoint_iterations):
            #     print("\n[ITER {}] Saving Checkpoint at {}".format(iteration, scene.model_path + "/chkpnt" + str(iteration) + ".pth"))
            #     if not b_vlrenderfeat:
            #         torch.save({'gaussians': gaussians.capture(),
            #                     'iteration': iteration}, scene.model_path + "/chkpnt" + str(iteration) + ".pth")
            #     else:
            #         torch.save({'gaussians': gaussians.capture(),
            #                     'iteration': iteration,
            #                     'lerf_model': lerf_model.state_dict(),
            #                     'lerf_optimizer': lerf_optimizer.state_dict()}, scene.model_path + "/chkpnt" + str(iteration) + ".pth")
            #     torch.cuda.empty_cache()

            # Log and save
            if not bvl_feature_precomp:
                training_report(tb_writer, iteration, Ll1_rec, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            else:
                # print("fmap_lr: ", lerf_opt_lr_scheduler.get_last_lr()[0])
                training_fmap_report(tb_writer, iteration, L_dino_loss, L_clip_loss, L_dotp_sim_loss, loss, iter_start.elapsed_time(iter_end), testing_iterations,
                                     scene, render, lerf_model, (pipe, background, fmap_resolution, args.fmap_render_radiithre))
def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1_rec, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/Ll1_rec_loss', Ll1_rec.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

def training_fmap_report(tb_writer, iteration, Lfmap0, Lfmap1, L_dotp_sim_loss, loss, elapsed, testing_iterations, scene : Scene, renderFunc, lerf_model, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/Lfmap0', Lfmap0.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/Lfmap1', Lfmap1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/L_dotp_sim_loss', L_dotp_sim_loss.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                lfmap_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    pipe, background, fmap_resolution, fmap_render_radiithre = renderArgs
                    render_pkg = renderFunc(viewpoint, scene.gaussians, pipe, background, lerfmodel=lerf_model,
                               bvl_feature_precomp=True, fmap_resolution=fmap_resolution, fmap_render_radiithre=fmap_render_radiithre)
                    rendered_image, rendered_featmap_dino, rendered_featmap_clip = (render_pkg["render"], render_pkg["rendered_featmap"], render_pkg["rendered_featmap_ex"])
                    image = torch.clamp(rendered_image, 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        # Get the visualization of the featmap
                        dino_fmap_show = get_shown_feat_map(rendered_featmap_dino, "RenderFeat_scaled").permute(2, 0, 1).contiguous() # (C, H, W)
                        clip_fmap_show = get_shown_feat_map(rendered_featmap_clip, "RenderFeat_scaled").permute(2, 0, 1).contiguous() # (C, H, W)
                        dino_fmap_show = torch.clamp(dino_fmap_show, 0.0, 1.0)
                        clip_fmap_show = torch.clamp(clip_fmap_show, 0.0, 1.0)
                        tb_writer.add_images(config['name'] + "_view_{}/dino".format(viewpoint.image_name),
                                             dino_fmap_show[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/clip".format(viewpoint.image_name),
                                             clip_fmap_show[None], global_step=iteration)
                        del clip_fmap_show; del dino_fmap_show
                        del rendered_featmap_clip; del rendered_featmap_dino

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    # l1_test += l1_loss(image, gt_image).mean().double()
                    # psnr_test += psnr(image, gt_image).mean().double()
                    torch.cuda.empty_cache()

                # psnr_test /= len(config['cameras'])
                # l1_test /= len(config['cameras'])
                # print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                # if tb_writer:
                #     tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                #     tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        # if tb_writer:
        #     tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
        #     tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--opt_vlrenderfeat_from', type=int, default=-1, help="Optimize the rendered VL feature map from the spec iteration. If it is < 0, then never optimize the feature map.")
    parser.add_argument('--dotp_simloss_w', type=float, default=-1.0, help="The wegith for the dotp_sim loss applied on the rendered VL feature maps , If it is < 0, then it is disabled.") # 0.03
    parser.add_argument('--fmap_resolution', type=int, default=-1, help="The scale for raysettings when render feature map. This can be -1, 4, 8 ...")
    parser.add_argument('--fmap_lr', type=float, default=1e-2, help="The learning rate for VL fmap")
    parser.add_argument('--fmap_render_radiithre', type=int, default=2, help="The radii threshold when rendering feature map.")
    parser.add_argument('--dataformat', type=str, default='colmap')
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)
    print("==========================")
    print(args)
    print("==========================")

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.opt_vlrenderfeat_from, args)

    # All done
    print("\nTraining complete.")
