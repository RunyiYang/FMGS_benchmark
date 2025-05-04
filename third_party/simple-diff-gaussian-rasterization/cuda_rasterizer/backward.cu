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

#include "backward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Backward version of the rendering procedure.
template <uint32_t CH, uint32_t CH_EX>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float* __restrict__ bg_color,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ conic_opacity,
	const float* __restrict__ colors,
    const float* __restrict__ colors_ex,
	const float* __restrict__ final_Ts,
	const uint32_t* __restrict__ n_contrib,
	const float* __restrict__ dL_dpixels, // dL_dout_color
    const float* __restrict__ dL_dpixels_ex, // dL_dout_color_ex
	float* __restrict__ dL_dcolors, // dL_dcolor, Gradient of loss w.r.t. colors_precomp
    float* __restrict__ dL_dcolors_ex // dL_dcolor_ex, Gradient of loss w.r.t. colors_ex_precomp
    )
{
	// We rasterize again. Compute necessary block info.
	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = { (float)pix.x, (float)pix.y };

	const bool inside = pix.x < W&& pix.y < H;
	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

	bool done = !inside;
	int toDo = range.y - range.x;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
//	__shared__ float collected_colors[CH * BLOCK_SIZE];

	// In the forward, we stored the final value for T, the
	// product of all (1 - alpha) factors.
	const float T_final = inside ? final_Ts[pix_id] : 0;
	float T = T_final;

	// We start from the back. The ID of the last contributing
	// Gaussian is known from each pixel from the forward.
	uint32_t contributor = toDo;
	const int last_contributor = inside ? n_contrib[pix_id] : 0;

	float accum_rec[CH] = { 0 };
    float accum_rec_ex[CH_EX] = { 0 };
    float dL_dpixel[CH];
	float dL_dpixel_ex[CH_EX];
	if (inside)
		for (int i = 0; i < CH; i++)
			dL_dpixel[i] = dL_dpixels[i * H * W + pix_id];
    if (inside)
        for (int i = 0; i < CH_EX; i++)
            dL_dpixel_ex[i] = dL_dpixels_ex[i * H * W + pix_id];

	float last_alpha = 0;
	float last_color[CH] = { 0 };
    float last_color_ex[CH_EX] = { 0 };

	// Gradient of pixel coordinate w.r.t. normalized
	// screen-space viewport corrdinates (-1 to 1)
	const float ddelx_dx = 0.5 * W;
	const float ddely_dy = 0.5 * H;

	// Traverse all Gaussians
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// Load auxiliary data into shared memory, start in the BACK
		// and load them in revers order.
		block.sync();
		const int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.y - progress - 1];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
//			for (int i = 0; i < CH; i++)
//				collected_colors[i * BLOCK_SIZE + block.thread_rank()] = colors[coll_id * CH + i];
		}
		block.sync();

		// Iterate over Gaussians
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current Gaussian ID. Skip, if this one
			// is behind the last contributor for this pixel.
			contributor--;
			if (contributor >= last_contributor)
				continue;

			// Compute blending values, as before.
			const float2 xy = collected_xy[j];
			const float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			const float4 con_o = collected_conic_opacity[j];
			const float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			const float G = exp(power);
			const float alpha = min(0.99f, con_o.w * G);
			if (alpha < 1.0f / 255.0f)
				continue;

			T = T / (1.f - alpha);
			const float dchannel_dcolor = alpha * T;

			// Propagate gradients to per-Gaussian colors and keep
			// gradients w.r.t. alpha (blending factor for a Gaussian/pixel
			// pair).
//			float dL_dalpha = 0.0f;
			const int global_id = collected_id[j];
			for (int ch = 0; ch < CH; ch++)
			{
//				const float c = collected_colors[ch * BLOCK_SIZE + j];
                const float c = colors[global_id * CH + ch];

				// Update last color (to be used in the next iteration)
				accum_rec[ch] = last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
				last_color[ch] = c;

				const float dL_dchannel = dL_dpixel[ch];
//				dL_dalpha += (c - accum_rec[ch]) * dL_dchannel;
				// Update the gradients w.r.t. color of the Gaussian.
				// Atomic, since this pixel is just one of potentially
				// many that were affected by this Gaussian.
				atomicAdd(&(dL_dcolors[global_id * CH + ch]), dchannel_dcolor * dL_dchannel);
			}

            for (int ch = 0; ch < CH_EX; ch++)
            {
                const float c = colors_ex[global_id * CH_EX + ch];

                // Update last color (to be used in the next iteration)
                accum_rec_ex[ch] = last_alpha * last_color_ex[ch] + (1.f - last_alpha) * accum_rec_ex[ch];
                last_color_ex[ch] = c;

                const float dL_dchannel_ex = dL_dpixel_ex[ch];
                atomicAdd(&(dL_dcolors_ex[global_id * CH_EX + ch]), dchannel_dcolor * dL_dchannel_ex);
            }

			// Update last alpha (to be used in the next iteration)
			last_alpha = alpha;
		}
	}
}



void BACKWARD::preprocess(
	int P, int D, int M,
	const float3* means3D,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* cov3Ds,
	const float* viewmatrix,
	const float* projmatrix,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	const glm::vec3* campos,
	float* dL_dcolor, // dL_dcolor, Gradient of loss w.r.t. colors_precomp
    float* dL_dcolor_ex
)
{
    return;
}

void BACKWARD::render(
	const dim3 grid, const dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H, int FEATURES_CH, int FEATURES_EX_CH,
	const float* bg_color,
	const float2* means2D,
	const float4* conic_opacity,
	const float* colors,
    const float* colors_ex,
	const float* final_Ts,
	const uint32_t* n_contrib,
	const float* dL_dpixels, // dL_dout_color
    const float* dL_dpixels_ex, // dL_dout_color_ex
	float* dL_dcolors, // dL_dcolor, Gradient of loss w.r.t. colors_precomp
    float* dL_dcolors_ex) // dL_dcolor_ex, Gradient of loss w.r.t. colors_ex_precomp
{
    if (FEATURES_CH != VL_FEATURE_NUM_CHANNELS){
        printf("FEATURES_CH != VL_FEATURE_NUM_CHANNELS:  %d %d \n", FEATURES_CH, VL_FEATURE_NUM_CHANNELS);
    }
    assert(FEATURES_CH==VL_FEATURE_NUM_CHANNELS);
    if (FEATURES_EX_CH != VL_FEATURE_EX_NUM_CHANNELS){
        printf("FEATURES_EX_CH != VL_FEATURE_EX_NUM_CHANNELS:  %d %d \n", FEATURES_EX_CH, VL_FEATURE_EX_NUM_CHANNELS);
    }
    assert(FEATURES_EX_CH==VL_FEATURE_EX_NUM_CHANNELS);

    // renderCUDA
    renderCUDA<VL_FEATURE_NUM_CHANNELS, VL_FEATURE_EX_NUM_CHANNELS> << <grid, block >> >(
		ranges,
		point_list,
		W, H,
		bg_color,
		means2D,
		conic_opacity,
		colors,
        colors_ex,
		final_Ts,
		n_contrib,
		dL_dpixels, // dL_dout_color
        dL_dpixels_ex,
		dL_dcolors, // dL_dcolor, Gradient of loss w.r.t. colors_precomp
        dL_dcolors_ex
		);
}