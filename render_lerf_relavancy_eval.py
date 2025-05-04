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
from third_party.scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from third_party.myutils.general_utils import safe_state
from argparse import ArgumentParser
from third_party.arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from third_party.myutils.system_utils import searchForMaxCKPTfile
from lerf.lerf import LERFModel
from lerf.encoders.openclip_encoder import OpenCLIPNetworkConfig, OpenCLIPNetwork
from third_party.myutils.vis_lerf_utils import get_shown_feat_map, get_composited_relevancy_map
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from torchvision.transforms import ToPILImage
import cv2

from typing import Any,Dict, List
import json
from pathlib import Path


lerf_image_encoder_config = OpenCLIPNetworkConfig(clip_model_type="ViT-B-16", clip_model_pretrained="laion2b_s34b_b88k",
                                                  clip_n_dims=512)
# print("for debug, lerf_image_encoder_config.negatives, clip_model_type: ", lerf_image_encoder_config.negatives, lerf_image_encoder_config.clip_model_pretrained, lerf_image_encoder_config.clip_model_type)
lerf_image_encoder = OpenCLIPNetwork(lerf_image_encoder_config)

eval_keyframe_path_filename = None
to_pil = ToPILImage()

global_opt=None


def get_shapes_fromjson(shape_path: str) -> List:
    with open(shape_path, 'r') as f:
        json_data = json.load(f)
        return json_data['shapes']

def get_relavancy(lerf_model, rendered_featmap_clip, rendered_image, img_idx, render_path, results_lines = []):
    '''
    rendered_image: tensor (C, H, W)
    '''
    global eval_keyframe_path_filename
    # Get positive text prompts
    eval_keyframes_dir = Path(eval_keyframe_path_filename).parents[0] # eval_keyframes_filename is "xxxx/keyframes_reversed_transform2colmap.json"

    shapes = get_shapes_fromjson(os.path.join(eval_keyframes_dir, f'{img_idx}_rgb.json'))
    img_path = os.path.join(eval_keyframes_dir, f'{img_idx}_rgb.png')
    queries = [shape['label'] for shape in shapes]

    # Some queries have multiple bounding boxes, finding the boxes for each query
    query_boxes_dict = {}
    for query_id, query in enumerate(queries):
        points = shapes[query_id]['points']
        min_x = min(points[0][0], points[1][0])
        max_x = max(points[0][0], points[1][0])
        min_y = min(points[0][1], points[1][1])
        max_y = max(points[0][1], points[1][1])
        box = (min_x, min_y, max_x, max_y)
        if query in query_boxes_dict:
            query_boxes_dict[query].append(box)
        else:
            query_boxes_dict[query] = [box]

    output_path = f'{render_path}/eval_relevancy'
    Path(output_path).mkdir(parents=True, exist_ok=True)

    # Writing the rendered rgb image, this is only for debug only
    # out_rgb = (rendered_image.permute(1, 2, 0).contiguous().cpu().numpy() * 255).astype(np.uint8)
    # out_rgb = cv2.cvtColor(out_rgb, cv2.COLOR_BGR2RGB)
    # cv2.imwrite(f'{output_path}/{img_idx}_rendered_rgb.png', out_rgb)
    torchvision.utils.save_image(rendered_image, f'{output_path}/{img_idx}_rendered_rgb.png')

    correctquery_in_img = 0
    for query in query_boxes_dict:
        with torch.no_grad():
            # print("----------- query: ", query)
            lerf_image_encoder.set_positives(text_list=[query])
            pos_prob_maps_rendered = lerf_model.get_relevancy_img(rendered_featmap_clip,
                                                                  lerf_image_encoder)  # (n_positive_embeds, H, W, 1)
            # rendered_composited_relevancy_maps = get_composited_relevancy_map(rendered_image.permute(1, 2, 0).contiguous(), pos_prob_maps_rendered)  # n_positive_embeds [(H, W, 1)]

        name = f'{output_path}/{img_idx}_{query}'
        output_relevancy = pos_prob_maps_rendered[0].cpu().numpy()  # Tensor (270, 480, 1)

        plt.matshow(output_relevancy, cmap='turbo')  # 'jet'
        plt.axis('off')
        plt.savefig(f'{name}_relevancy.png')
        plt.close()
        np.save(f'{name}_relevancy.npy', output_relevancy)

        #######################################
        rel_value = output_relevancy
        ind = np.unravel_index(np.argmax(rel_value, axis=None), rel_value.shape)
        max_response = (ind[1], ind[0])

        # image_data = to_pil(rendered_image.clone())
        image_data = Image.open(img_path)
        image_draw = ImageDraw.Draw(image_data)

        m = 0.5
        check = False
        for box_coords in query_boxes_dict[query]:
            min_x, min_y, max_x, max_y = box_coords[0], box_coords[1], box_coords[2], box_coords[3]
            box = [(min_x, min_y), (max_x, max_y)]

            check_x = max_response[0] > min_x - m and (max_response[0] < max_x + m)
            check_y = max_response[1] > min_y - m and (max_response[1] < max_y + m)
            check_this_box = check_x and check_y
            check = check | check_this_box

            # Drawing the boxes
            image_draw.rectangle(box, outline="red", width=3)

        dot_shape = [(max_response[0] - 2, max_response[1] - 2),
                     (max_response[0] + 2, max_response[1] + 2)]
        image_draw.rectangle(dot_shape, outline="blue", width=6)
        if check:
            image_draw.rectangle([(0, 0), (20, 20)], fill="green", width=3)
            correctquery_in_img += 1
        else:
            image_draw.rectangle([(0, 0), (20, 20)], fill="red", width=3)
        image_data.save(f"{name}.png")

        # print(f"Query '{query}' --- in camera {img_idx}, correct {check}")

    results_lines.append([correctquery_in_img, len(set(queries)), img_idx])
    print(f"--- Correctquery_in_img {correctquery_in_img}/{len(set(queries))} for camera_idx {img_idx}")
    return correctquery_in_img, len(set(queries))



def render_lerf_set(model_path, name, iteration, views, gaussians,  lerf_model, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    # gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    # makedirs(gts_path, exist_ok=True)

    results_lines = []
    num_true_positives = 0
    num_labels = 0

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        # ### Render RGB image only
        # rendered_image = render(view, gaussians, pipeline, background)["render"]
        # torchvision.utils.save_image(rendered_image, os.path.join(render_path, '{0:05d}_rgb'.format(idx) + ".png"))
        # # gt = view.original_image[0:3, :, :]
        # # torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

        ### Render RGB image and fmap
        render_pkg = render(view, gaussians, pipeline, background, lerfmodel=lerf_model, bvl_feature_precomp=True)
        rendered_image, rendered_featmap_dino, rendered_featmap_clip = (render_pkg["render"], render_pkg["rendered_featmap"], render_pkg["rendered_featmap_ex"])
        torchvision.utils.save_image(rendered_image, os.path.join(render_path, '{0:05d}_rgb'.format(idx) + ".png"))

        # ## Visualize the dino feature
        # render_fmap_show = get_shown_feat_map(rendered_featmap_dino, "RenderFeat_scaled").cpu().numpy()
        # plt.imsave(os.path.join(render_path, '{0:05d}_dino'.format(idx) + ".png"), render_fmap_show)
        
        # # Display the image using Matplotlib
        # plt.imshow(render_fmap_show)
        # plt.axis('off')  # Hide the axis
        # plt.title('fmap_rendered_dino & fmap_gt')
        # plt.show()

        # ## Visualize the Clip feature
        # render_fmap_ex_show = get_shown_feat_map(rendered_featmap_clip, "RenderFeat_scaled").cpu().numpy()
        # plt.imsave(os.path.join(render_path, '{0:05d}_clip'.format(idx) + ".png"), render_fmap_ex_show)
        
        # # Display the image using Matplotlib
        # plt.imshow(render_fmap_ex_show)
        # plt.axis('off')  # Hide the axis
        # plt.title('fmap_rendered_clip & fmap_gt')
        # plt.show()

        ## Get relavancy map
        num_correctquery, num_query = get_relavancy(lerf_model, rendered_featmap_clip, rendered_image, idx, render_path, results_lines=results_lines)
        num_true_positives += num_correctquery
        num_labels += num_query

    str_eval = f'{num_true_positives / num_labels:5f} {num_true_positives} {num_labels}'
    print("In general, correct_rate: " + str_eval)
    eval_output_path = f'{render_path}/eval_relevancy'
    with open(f'{eval_output_path}/eval_relevancy.txt', 'w') as file1:
        file1.write("correct_query total_query camera_idx \n")
        for result in results_lines:
            file1.write(" ".join(map(str, result)) + "\n")
        file1.write(str_eval)


def render_lerf_sets(dataset : ModelParams, ckpt_filename : str, pipeline : PipelineParams, runon_train : bool, skip_test : bool, dataformat='lerf'):
    with torch.no_grad():
        # Load the saved ckpt
        loaded_statedict = torch.load(ckpt_filename)
        assert isinstance(loaded_statedict, dict), f"The given ckpt {ckpt_filename} is probably not a VL_GS model!"
        gaussians_model_params = loaded_statedict['gaussians']
        iteration_num = loaded_statedict['iteration']
        gaussians = GaussianModel(dataset.sh_degree)

        # Restore the lerf relevant states
        assert 'lerf_model' in loaded_statedict, f"The given ckpt_filename {ckpt_filename} does not contain weights of lerf_model!"
        lerf_model = LERFModel()
        lerf_model.cuda()
        lerf_model.eval()
        lerf_model.load_state_dict(loaded_statedict['lerf_model'], strict=False)

        # scene = Scene(dataset, gaussians, load_iteration=iteration_num, shuffle=False, dataformat=dataformat)

        scene = Scene(dataset, gaussians, shuffle=False, dataformat=dataformat)
        gaussians.restore(gaussians_model_params)

        # import pdb; pdb.set_trace()

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if runon_train:
             render_lerf_set(dataset.model_path, "train", iteration_num, scene.getTrainCameras(), gaussians, lerf_model, pipeline, background)

        if not skip_test:
             render_lerf_set(dataset.model_path, "test", iteration_num, scene.getTestCameras(), gaussians, lerf_model, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument('--dataformat', type=str, default='colmap')
    parser.add_argument("--runon_train", action="store_true") # By default, we won't run on train.
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--eval_keyframe_path_filename",
                        default='/usr/local/google/home/xingxingzuo/Documents/lerf/data/lerf/Datasets/Localization_eval_dataset/bouquet/gs_render_label_gt/keyframes_reversed_transform2colmap.json',
                        metavar="FILE", help="path to eval keyframe file")
    args = get_combined_args(parser)
    print("Rendering: ", args.model_path)
    print("eval_keyframe_path_filename: ",  args.eval_keyframe_path_filename)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    ckpt_filename = None
    if args.iteration < 0:
        # Find the ckpt at max iteration by default
        ckpt_filename = searchForMaxCKPTfile(args.model_path)
        ckpt_filename = os.path.join(args.model_path, ckpt_filename)
    else:
        ckpt_filename = os.path.join(args.model_path, "chkpnt{}.pth".format(args.iteration))

    print(f"Eval on the ckpt: {ckpt_filename}")
    eval_keyframe_path_filename = args.eval_keyframe_path_filename
    render_lerf_sets(model.extract(args), ckpt_filename, pipeline.extract(args), args.runon_train, args.skip_test, dataformat=args.dataformat)