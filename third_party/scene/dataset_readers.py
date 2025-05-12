#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image
from typing import NamedTuple
from third_party.scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from third_party.myutils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from third_party.myutils.sh_utils import SH2RGB
from third_party.scene.gaussian_model import BasicPointCloud
from tqdm import tqdm
class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def rotation_and_translation_to_matrix(rotation, translation_vector):
    # Create a 4x4 transformation matrix
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation
    transformation_matrix[:3, 3] = translation_vector

    return transformation_matrix

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        # # For debug only
        # tmp_matrix = rotation_and_translation_to_matrix(qvec2rotmat(extr.qvec), extr.tvec)
        # print("------------------ key: ", key, "\n")
        # print(tmp_matrix)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    # normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=None)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8):
    # try:
    #     cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
    #     cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
    #     cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
    #     cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    # except:
    #     cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
    #     cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
    #     cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
    #     cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    # Rocky: this is customized for lerf dataset

    sequence_name = os.path.basename(path)
    lerf_sequences = ["bouquet",  "figurines",  "ramen",  "teatime",  "waldo_kitchen"]
    # keyframeposes_filename = os.path.join(os.path.dirname(os.path.dirname(path)), f"Localization_eval_dataset/{sequence_name}/gs_render_label_gt/keyframes_reversed_transform2colmap.json")
    keyframeposes_filename = os.path.join(os.path.dirname(os.path.dirname(path)),
                                          f"Localization_eval_puremycolmap/{sequence_name}/keyframes_reversed_transform2colmap.json")

    print("------- keyframeposes_filename in dataset_readers.py", keyframeposes_filename)
    # if os.path.exists(keyframeposes_filename):
    if sequence_name in lerf_sequences:
        print(f"This is a lerf dataset, overide test_cam_infos according to lerf_sequences names: {lerf_sequences}")
        test_cam_infos = readKeyframesCameras_lerf_mycolmap(keyframeposes_filename, False, extension="")
    else:
        print(f"This is not a lerf dataset!")

    # Rocky: this is customized for 3dovs dataset
    segmentation_dir=os.path.join(path, "segmentations")
    if os.path.exists(segmentation_dir):
        print(f"This is a 3dvos dataset, since the folder exists: {segmentation_dir}")
        print(f"Please double check whether it is 3dovs dataset!")

        # get a list of all the folders in the directory
        seg_folders = [f for f in os.listdir(segmentation_dir) if os.path.isdir(os.path.join(segmentation_dir, f))]
        seg_folders = sorted(seg_folders, key=lambda x: int(x))
        seg_test_id = [int(i) for i in seg_folders]
        print("seg_folders/test_frame_id in the dataset: ", seg_folders)


        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx in seg_test_id]
        print("test_cam_infos: ", test_cam_infos)

        # # TODO: downsample the test images
        # for test_cam in test_cam_infos:
        #     test_cam.width = int(test_cam.width/8)
        #     test_cam.height = int(test_cam.height/8)

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def nerstudio_c2w_to_colmap_w2c(c2w_init):
    w2c = c2w_init.copy()
    w2c[2, :] *= -1
    w2c = w2c[np.array([1, 0, 2, 3]), :]
    w2c[0:3, 1:3] *= -1
    w2c = np.linalg.inv(w2c)
    return w2c


def readCamerasFromTransforms_lerf(path, transformsfile, white_background, extension=".jpg"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])

            # # ## Way 1: implemented in readCamerasFromTransforms() -- GaussainSplatting:scene/dataset_readers.py
            # # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            # c2w[:3, 1:3] *= -1
            # # get the world-to-camera transform and set R, T
            # w2c = np.linalg.inv(c2w)
            # R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
            # T = w2c[:3, 3]

            ### From my test, this way is correct.
            # ## Way 2: implemented by myself.
            w2c = nerstudio_c2w_to_colmap_w2c(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]


            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")
            assert (image.size[0] == frame['w'])
            assert (image.size[1] == frame['h'])

            focal_length_x = frame['fl_x']
            focal_length_y = frame['fl_y']
            FovY = focal2fov(focal_length_y, frame['h'])
            FovX = focal2fov(focal_length_x, frame['w'])

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                        image_path=image_path, image_name=image_name, width=image.size[0],
                                        height=image.size[1]))

    return cam_infos


def three_js_perspective_camera_focal_length(fov: float, image_height: int):
    """Returns the focal length of a three.js perspective camera.

    Args:
        fov: the field of view of the camera in degrees.
        image_height: the height of the image in pixels.
    """
    if fov is None:
        print("Warning: fov is None, using default value")
        return 50
    pp_h = image_height / 2.0
    focal_length = pp_h / np.tan(fov * (np.pi / 180.0) / 2.0)
    return focal_length


def readKeyframesCameras_lerf(path, keyframefile, white_background, extension=".jpg"):
    cam_infos = []

    with open(os.path.join(path, keyframefile)) as json_file:
        contents = json.load(json_file)

        frames = contents["keyframes"]
        img_width = contents['render_width']
        img_height = contents['render_height']

        for idx, frame in enumerate(frames):
            # NeRF 'transform_matrix' is a camera-to-world transform
            # read_matrix = [float(c) for c in frame["matrix"][1:-1].split(',')]
            # c2w = torch.t(torch.tensor(read_matrix).view(4, 4)).numpy()

            c2w = np.asarray(frame["reversed_aligned_matrix"])


            # # ## Way 1: implemented in readCamerasFromTransforms() -- GaussainSplatting:scene/dataset_readers.py
            # # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            # c2w[:3, 1:3] *= -1
            # # get the world-to-camera transform and set R, T
            # w2c = np.linalg.inv(c2w)
            # R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
            # T = w2c[:3, 3]

            ### From my test, this way is correct.
            # ## Way 2: implemented by myself.
            w2c = nerstudio_c2w_to_colmap_w2c(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            # cam_name = os.path.join(path, frame["file_path"] + extension)
            # image_path = os.path.join(path, cam_name)
            # image_name = Path(cam_name).stem
            # image = Image.open(image_path)
            # im_data = np.array(image.convert("RGBA"))
            #
            # bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])
            #
            # norm_data = im_data / 255.0
            # arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            # image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")
            # assert (image.size[0] == frame['w'])
            # assert (image.size[1] == frame['h'])

            image_fake = Image.fromarray(np.array(np.ones((img_height, img_width, 3)) * 255.0, dtype=np.byte), "RGB")

            fov = frame['fov']
            focal_length = three_js_perspective_camera_focal_length(fov, img_height)
            # # focal_length_x = three_js_perspective_camera_focal_length(fov, img_width)
            FovY = focal2fov(focal_length, img_height)
            FovX = focal2fov(focal_length, img_width)
            # print(f"FovX {FovX}, FovY {FovY}, focal_length {focal_length}, image_height {img_height}, image_width {img_width}")


            # focal_length_x = 1158.0337370751618
            # focal_length_y = 1158.0337370751618
            # FovY = focal2fov(focal_length_y,1080)
            # FovX = focal2fov(focal_length_x, 1920)

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image_fake,
                                        image_path='', image_name='', width=img_width,
                                        height=img_height))

    return cam_infos


def readKeyframesCameras_lerf_mycolmap(keyframefile, white_background, extension=".jpg"):
    cam_infos = []

    with open(keyframefile) as json_file:
        contents = json.load(json_file)

        frames = contents["keyframes"]
        img_width = contents['render_width']
        img_height = contents['render_height']

        for idx, frame in enumerate(frames):
            w2c = np.asarray(frame["transformed_mycolmap_w2c_matrix"])
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_fake = Image.fromarray(np.array(np.ones((img_height, img_width, 3)) * 255.0, dtype=np.byte), "RGB")

            fov = frame['fov']
            focal_length = three_js_perspective_camera_focal_length(fov, img_height)
            # # focal_length_x = three_js_perspective_camera_focal_length(fov, img_width)
            FovY = focal2fov(focal_length, img_height)
            FovX = focal2fov(focal_length, img_width)
            # print(f"FovX {FovX}, FovY {FovY}, focal_length {focal_length}, image_height {img_height}, image_width {img_width}")


            # focal_length_x = 1158.0337370751618
            # focal_length_y = 1158.0337370751618
            # FovY = focal2fov(focal_length_y,1080)
            # FovX = focal2fov(focal_length_x,1920)

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image_fake,
                                        image_path='', image_name='', width=img_width,
                                        height=img_height))

    return cam_infos


def readNerfLerfInfo(path, white_background, eval, extension=".jpg"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms_lerf(path, "transforms_train.json", white_background, extension="")
    print("Reading Test Transforms")

    sequence_name = os.path.basename(path)
    test_cam_infos = readKeyframesCameras_lerf(os.path.dirname(os.path.dirname(path)), f"Localization_eval_dataset/{sequence_name}/keyframes_reversed.json", white_background, extension="")

    # try:
    #     # test_cam_infos = readCamerasFromTransforms_lerf(path, "transforms_test.json", white_background, extension="")
    #     sequence_name = os.path.basename(path)
    #     test_cam_infos = readKeyframesCameras_lerf(os.path.dirname(os.path.dirname(path)), f"{sequence_name}/Localization_eval_dataset/keyframes.json", white_background, extension="")
    # except:
    #     test_cam_infos = []

    # if not eval:
    #     train_cam_infos.extend(test_cam_infos)
    #     test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms_nerfstudio(path, transformsfile, depths_folder, white_background, is_test, extension=".png"):
    cam_infos = []

    with open(transformsfile) as json_file:
        contents = json.load(json_file)
        focal_len_x = contents["fl_x"]
        focal_len_y = contents["fl_y"]
        cx = contents["cx"] * 2 
        cy = contents["cy"] * 2 
        fovx = focal2fov(focal_len_x, cx)
        fovy = focal2fov(focal_len_y, cy)

        FovY = fovy 
        FovX = fovx
        frames = contents["frames"]
        for idx, frame in enumerate(tqdm(frames)):
            image_path = path.replace("nerfstudio", "undistorted_images")
            cam_name = frame["file_path"]

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            applied_transform = np.array([
                [0,  1,  0,  0],
                [1,  0,  0,  0],
                [0,  0, -1,  0],
                [0,  0,  0,  1],
            ], dtype=float)
            c2w = np.dot(applied_transform, c2w)
            # get the world-to-camera transform and set R, T
            # w2c = c2w
            w2c = np.linalg.inv(c2w)
            w2c[1:3] *= -1
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]
            image_path = os.path.join(image_path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)
            im_data = np.array(image.convert("RGBA"))
            bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])
            norm_data = im_data / 255.0
            arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")



            depth_path = os.path.join(depths_folder, f"{image_name}.png") if depths_folder != "" else ""

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name,
                            width=cx, height=cy))
            
    return cam_infos

def readCamerasFromTransforms_nerfstudio_opencv(path, transformsfile, depths_folder, white_background, is_test, extension=".jpg"):
    cam_infos = []

    with open(transformsfile) as json_file:
        contents = json.load(json_file)
        focal_len_x = contents["fx"]
        focal_len_y = contents["fy"]
        cx = contents["cx"] * 2 
        cy = contents["cy"] * 2 
        fovx = focal2fov(focal_len_x, cx)
        fovy = focal2fov(focal_len_y, cy)

        FovY = fovy 
        FovX = fovx
        frames = contents["frames"]
        
        for idx, frame in enumerate(tqdm(frames)):
            image_path = path
            cam_name = frame["file_path"]

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]
            image_path = os.path.join(image_path, cam_name) + ".jpg"
            image_name = Path(cam_name).stem
            image = Image.open(image_path)
            im_data = np.array(image.convert("RGBA"))
            bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])
            norm_data = im_data / 255.0
            arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")



            depth_path = os.path.join(depths_folder, f"{image_name}.png") if depths_folder != "" else ""

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name,
                            width=cx, height=cy))
            
    return cam_infos

def readCamerasFromTransforms_nerfstudio_opencv_synthetic(path, transformsfile, depths_folder, white_background, is_test, extension=".jpg"):
    cam_infos = []

    with open(transformsfile) as json_file:
        contents = json.load(json_file)
        frames = contents["frames"]
        
        if contents["share_intrinsics"] == True:
            focal_len_x = contents["fx"]
            focal_len_y = contents["fy"]
            cx = contents["cx"] * 2 
            cy = contents["cy"] * 2 
            fovx = focal2fov(focal_len_x, cx)
            fovy = focal2fov(focal_len_y, cy)

            FovY = fovy 
            FovX = fovx
        
        for idx, frame in enumerate(tqdm(frames)):
            image_path = path
            cam_name = frame["file_path"]

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]
            image_path = os.path.join(image_path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)
            im_data = np.array(image.convert("RGBA"))
            bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])
            norm_data = im_data / 255.0
            arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")

            if contents["share_intrinsics"] == False:
                focal_len_x = frame["fx"]
                focal_len_y = frame["fy"]
                cx = frame["cx"] * 2 
                cy = frame["cy"] * 2 
                fovx = focal2fov(focal_len_x, cx)
                fovy = focal2fov(focal_len_y, cy)

                FovY = fovy 
                FovX = fovx

            depth_path = os.path.join(depths_folder, f"{image_name}.png") if depths_folder != "" else ""

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name,
                            width=cx, height=cy))
            
    return cam_infos

def readScanNetInfo(path, white_background, depths, eval, extension=".JPG"):

    depths_folder=os.path.join(path, depths) if depths != "" else ""
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms_nerfstudio_opencv(path, os.path.join(path, "lang_feat_selected_imgs.json"), depths_folder, white_background, False, extension)
    # print("Reading Test Transforms")
    # test_cam_infos = readCamerasFromTransforms_nerfstudio(path, os.path.join(path, "nerfstudio/transforms_undistorted.json"),  depths_folder, white_background, True, extension)
    
    # if not eval:
        # train_cam_infos.extend(test_cam_infos)
    test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)
    
    ply_path = os.path.join(path, "point3d.ply")
    bin_path = os.path.join(path, "../colmap/colmap/points3D.bin")
    txt_path = os.path.join(path, "../colmap/colmap/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
            storePly(ply_path, xyz, rgb)
        except:
            pass
        
        
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        # storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readScanNetppInfo(path, white_background, depths, eval, extension=".JPG"):

    depths_folder=os.path.join(path, depths) if depths != "" else ""
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms_nerfstudio(path, os.path.join(path, "lang_feat_selected_imgs.json"), depths_folder, white_background, False, extension)
    # print("Reading Test Transforms")
    # test_cam_infos = readCamerasFromTransforms_nerfstudio(path, os.path.join(path, "nerfstudio/transforms_undistorted.json"),  depths_folder, white_background, True, extension)
    
    # if not eval:
        # train_cam_infos.extend(test_cam_infos)
    test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)
    ply_path = os.path.join(path, "points3D.ply")
    bin_path = os.path.join(path, "../colmap/colmap/points3D.bin")
    txt_path = os.path.join(path, "../colmap/colmap/points3D.txt")

    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
            storePly(ply_path, xyz, rgb)
        except:
            pass
        
        
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        # storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readReplicaInfo(path, white_background, depths, eval, lang_path, extension=".JPG"):

    depths_folder=os.path.join(path, depths) if depths != "" else ""
    print("Reading Training Transforms")
    train_cam_infos = readCamerasReplica(path, lang_path, depths_folder, white_background, False, extension)
    # print("Reading Test Transforms")
    # test_cam_infos = readCamerasReplica(path, lang_path, depths_folder, white_background, True, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        # storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=False)
    return scene_info

def readCamerasReplica(path, transformsfile, depths_folder, white_background, is_test, extension=".png"):
    cam_infos = []
    print(transformsfile)
    with open(transformsfile) as json_file:
        contents = json.load(json_file)
        focal_len_x = contents["fl_x"]
        focal_len_y = contents["fl_y"]
        cx = contents["w"] 
        cy = contents["h"]
        fovx = focal2fov(focal_len_x, cx)
        fovy = focal2fov(focal_len_y, cy)
        print(cx, cy)
        FovY = fovy 
        FovX = fovx
        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            # print(path)
            image_path = path.replace("nerfstudio", "undistorted_images")
            cam_name = frame["file_path"]

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            applied_transform = np.array([
                [0,  1,  0,  0],
                [1,  0,  0,  0],
                [0,  0, -1,  0],
                [0,  0,  0,  1],
            ], dtype=float)
            # c2w = np.dot(applied_transform, c2w)
            # get the world-to-camera transform and set R, T
            # w2c = c2w
            w2c = np.linalg.inv(c2w)
            # w2c[1:3] *= -1
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]
            image_path = os.path.join(image_path, cam_name)
            image_name = Path(cam_name).stem
    
            depth_path = os.path.join(depths_folder, f"{image_name}.png") if depths_folder != "" else ""

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX,
                            image_path=image_path, image_name=image_name,
                            width=cx, height=cy, depth_path=depth_path, depth_params=None, is_test=is_test))
            
    return cam_infos




def readGenericFrameInfo(path, white_background, depths, eval, extension=".jpg"):

    depths_folder=os.path.join(path, depths) if depths != "" else ""
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms_nerfstudio_opencv_synthetic(path, os.path.join(path, "transforms_train.json"), depths_folder, white_background, False, extension)
    # print("Reading Test Transforms")
    # test_cam_infos = readCamerasFromTransforms_nerfstudio_opencv(path, depths_folder, white_background, True, extension)
    
    # if not eval:
    #     train_cam_infos.extend(test_cam_infos)
    #     test_cam_infos = []

    test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "point3d_fused.ply")
    bin_path = os.path.join(path, "../colmap/colmap/points3D.bin")
    txt_path = os.path.join(path, "../colmap/colmap/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
            storePly(ply_path, xyz, rgb)
        except:
            pass
        
        
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        # storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None
    # import pdb; pdb.set_trace()
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info



sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "ScanNetpp": readScanNetppInfo,
    "Replica": readReplicaInfo,
    "ScanNet": readScanNetInfo,
    "Matterport": readGenericFrameInfo,
    "Holicity": readGenericFrameInfo,
}