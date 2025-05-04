/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_CONFIG_H_INCLUDED
#define CUDA_RASTERIZER_CONFIG_H_INCLUDED

#define NUM_CHANNELS 3 // Default 3, RGB
#define VL_FEATURE_NUM_CHANNELS 384 // 384 for Dino feature, 512 for Clip feature
#define VL_FEATURE_EX_NUM_CHANNELS 512 // 512 for Clip feature
#define BLOCK_X 16
#define BLOCK_Y 16

#endif