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

#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include <fstream>
#include <algorithm>


// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	// clamped[3 * idx + 0] = (result.x < 0);
	// clamped[3 * idx + 1] = (result.y < 0);
	// clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

// Forward version of 2D covariance matrix computation
__device__ float3 computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, glm::mat4 viewmatrix)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	float3 t = transformPoint4x3(mean, (float*)&viewmatrix);

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		((float*)&viewmatrix)[0], ((float*)&viewmatrix)[4], ((float*)&viewmatrix)[8],
		((float*)&viewmatrix)[1], ((float*)&viewmatrix)[5], ((float*)&viewmatrix)[9],
		((float*)&viewmatrix)[2], ((float*)&viewmatrix)[6], ((float*)&viewmatrix)[10]);

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Apply low-pass filter: every Gaussian should be at least
	// one pixel wide/high. Discard 3rd row and column.
	cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;
	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

// Perform initial steps for each Gaussian prior to rasterization.
//计算每一个3d高斯球在每个平面上的投影，以及投影与平面tile的交集
template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
	const float* orig_points,
	const float* opacities,
	const float* shs,
	glm::mat4 viewmatrix,
	glm::mat4 projmatrix,
	glm::vec3 cam_pos,
	const int W, int H,
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	int* radii,
	float2* points_xy_image,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	int* tiles_touched)
{
	//处理高斯球
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	int my_radii = 0;
	int my_tiles_touched = 0;

	do {
		// Perform near culling, quit if outside.
		float3 p_view;
		if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, false, p_view))
			break;

		if (255.0f * opacities[idx] < 1.0f)
			break;
		
		//3d高斯球的中心点投影
		// Transform point by projecting
		float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
		float4 p_hom = transformPoint4x4(p_orig, (float*)&projmatrix);
		float p_w = 1.0f / (p_hom.w + 0.0000001f);
		float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

		//3d高斯投影在平面上的椭圆
		// Compute 2D screen-space covariance matrix
		float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3Ds + idx * 6, viewmatrix);

		// Invert covariance (EWA algorithm)
		float det = (cov.x * cov.z - cov.y * cov.y);
		// if (det == 0.0f)
		// 	return;
		float det_inv = 1.f / det;
		float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

		float power = __logf(255.0f * opacities[idx]);
		int width = (int)(1.414214f * __fsqrt_ru(cov.x * power) + 1.0f);
		int height = (int)(1.414214f * __fsqrt_ru(cov.z * power) + 1.0f);

		//圆覆盖了哪些tile，要用的值存储在中间结果中
		uint2 rect_min, rect_max;
		float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
		getRect(point_image, width, height, rect_min, rect_max, grid);
		if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
			break;

		// If colors have been precomputed, use them, otherwise convert
		// spherical harmonics coefficients to RGB color.
		if(shs != nullptr)
		{
			glm::vec3 color = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, cam_pos, shs, nullptr);
			rgb[idx * C + 0] = color.x;
			rgb[idx * C + 1] = color.y;
			rgb[idx * C + 2] = color.z;
		}


		//计算深度值 depth id
		// Store some useful helper data for the next steps.
		depths[idx] = p_view.z;
		
		points_xy_image[idx] = point_image;
		// Inverse 2D covariance and opacity neatly pack into one float4
		conic_opacity[idx] = { conic.x, conic.y, conic.z, opacities[idx] };

		my_radii = width | height << 16;
		my_tiles_touched = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
	} while (false);

	radii[idx] = my_radii;
	tiles_touched[idx] = my_tiles_touched;
	// printf("proprecessend\n");
}

__device__ bool block_intersect_ellipse(int2 pix_min, int2 pix_max, float2 center, float4 con_o, bool check_x, bool check_y)
{
	float a, b, c, delta, dx, dy;
	int lambda1, lambda2;
	float w = 2.0f * __logf(256.0f * con_o.w);
	
	if (check_y)
	{
		dx = center.x - pix_min.x;
		a = con_o.z;
		b = -2.0f * con_o.y * dx;
		c = con_o.x * dx * dx - w;
		delta = b * b - 4.0f * a * c;
		if (delta >= 0.0f)
		{
			float sqrt_delta = __fsqrt_ru(delta);
			lambda1 = (-b + sqrt_delta) / (2.0f * a) + center.y;
			lambda2 = (-b - sqrt_delta) / (2.0f * a) + center.y;
			if (pix_min.y <= lambda1 && pix_max.y >= lambda2)
			{
				return true;
			}
		}
		
		dx = center.x - pix_max.x;
		a = con_o.z;
		b = -2.0f * con_o.y * dx;
		c = con_o.x * dx * dx - w;
		delta = b * b - 4.0f * a * c;
		if (delta >= 0.0f)
		{
			float sqrt_delta = __fsqrt_ru(delta);
			lambda1 = (-b + sqrt_delta) / (2.0f * a) + center.y;
			lambda2 = (-b - sqrt_delta) / (2.0f * a) + center.y;
			if (pix_min.y <= lambda1 && pix_max.y >= lambda2)
			{
				return true;
			}
		}			
	}

	if (check_x)
	{
		dy = center.y - pix_min.y;
		a = con_o.x;
		b = -2.0f * con_o.y * dy;
		c = con_o.z * dy * dy - w;
		delta = b * b - 4.0f * a * c;
		if (delta >= 0.0f)
		{
			float sqrt_delta = __fsqrt_ru(delta);
			lambda1 = (-b + sqrt_delta) / (2.0f * a) + center.x;
			lambda2 = (-b - sqrt_delta) / (2.0f * a) + center.x;
			if (pix_min.x <= lambda1 && pix_max.x >= lambda2)
			{
				return true;
			}
		}
		
		dy = center.y - pix_max.y;
		a = con_o.x;
		b = -2.0f * con_o.y * dy;
		c = con_o.z * dy * dy - w;
		delta = b * b - 4.0f * a * c;
		if (delta >= 0.0f)
		{
			float sqrt_delta = __fsqrt_ru(delta);
			lambda1 = (-b + sqrt_delta) / (2.0f * a) + center.x;
			lambda2 = (-b - sqrt_delta) / (2.0f * a) + center.x;
			if (pix_min.x <= lambda1 && pix_max.x >= lambda2)
			{
				return true;
			}
		}
	}
	return false;
}

__device__ bool block_contains_center(int2 pix_min, int2 pix_max, float2 center)
{
	return center.x >= pix_min.x && center.x <= pix_max.x && center.y >= pix_min.y && center.y <= pix_max.y;
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float* __restrict__ depths,
	const float4* __restrict__ conic_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
	float* __restrict__ out_depth)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	int2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	int2 pix_max = { min(pix_min.x + BLOCK_X - 1, W), min(pix_min.y + BLOCK_Y - 1, H) };
	int2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	// Check if this thread is associated with a valid pixel or outside.
	bool inside;

	inside = pix.x < W && pix.y < H;
	
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ int is_valid[BLOCK_SIZE];
	
	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };
	float D = { 0 };

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			float2 my_points_xy_image = points_xy_image[coll_id];
			collected_xy[block.thread_rank()] = my_points_xy_image;
			float4 my_conic_opacity = conic_opacity[coll_id];
			collected_conic_opacity[block.thread_rank()] = my_conic_opacity;
			int valid_mask = 0;
			for (int k = 0; k < 16; k += 2)
			{
				int2 pix_min_warp = { pix_min.x , pix_min.y + k};
				int2 pix_max_warp = { pix_max.x , pix_min.y + k + 1};
				bool valid0 = block_contains_center(pix_min_warp, pix_max_warp, my_points_xy_image) || 
					block_intersect_ellipse(pix_min_warp, pix_max_warp, my_points_xy_image, my_conic_opacity, true, false);
				valid_mask |= valid0 ? (3 << k) : 0;
			}
			is_valid[block.thread_rank()] = valid_mask;
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;
			if (!(is_valid[j] & (1 << block.thread_index().y)))
				continue;

			// Resample using conic matrix (cf. "Surface
			// Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 con_o = collected_conic_opacity[j];
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;
			
			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix).
			float alpha = min(0.99f, con_o.w * exp(power));
			if (alpha < 1.0f / 255.0f)
				continue;

			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += features[collected_id[j] * CHANNELS + ch] * (alpha * T);
			D += depths[collected_id[j]] * (alpha * T);
			
			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
		out_depth[pix_id] = D;
	}
}

void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	const float* colors,
	const float* depths,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color,
	float* out_depth)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		means2D,
		colors,
		depths,
		conic_opacity,
		final_T,
		n_contrib,
		bg_color,
		out_color,
		out_depth);
}

// void FORWARD::preprocess(int P, int D, int M,
// 	const float* means3D,
// 	const glm::vec3* scales,
// 	const float scale_modifier,
// 	const glm::vec4* rotations,
// 	const float* opacities,
// 	const float* shs,
// 	bool* clamped,
// 	const float* cov3D_precomp,
// 	const float* colors_precomp,
// 	const float* viewmatrix,
// 	const float* projmatrix,
// 	const glm::vec3* cam_pos,
// 	const int W, int H,
// 	const float focal_x, float focal_y,
// 	const float tan_fovx, float tan_fovy,
// 	int* radii,
// 	float2* means2D,
// 	float* depths,
// 	float* cov3Ds,
// 	float* rgb,
// 	float4* conic_opacity,
// 	const dim3 grid,
// 	uint32_t* tiles_touched,
// 	bool prefiltered)
// {
// 	glm::mat4 viewmatrix_host;
// 	glm::mat4 projmatrix_host;
// 	glm::vec3 cam_pos_host;
// 	cudaMemcpy(&viewmatrix_host, viewmatrix, sizeof viewmatrix_host, cudaMemcpyDeviceToHost);
// 	cudaMemcpy(&projmatrix_host, projmatrix, sizeof projmatrix_host, cudaMemcpyDeviceToHost);
// 	cudaMemcpy(&cam_pos_host, cam_pos, sizeof cam_pos_host, cudaMemcpyDeviceToHost);

// 	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
// 		P, D, M,
// 		means3D,
// 		opacities,
// 		shs,
// 		viewmatrix_host, 
// 		projmatrix_host,
// 		cam_pos_host,
// 		W, H,
// 		tan_fovx, tan_fovy,
// 		focal_x, focal_y,
// 		radii,
// 		means2D,
// 		depths,
// 		cov3Ds,
// 		rgb,
// 		conic_opacity,
// 		grid,
// 		tiles_touched
// 		);
// }
void FORWARD::preprocess(
	int P, int D, int M,
	const float* orig_points,
	const float* opacities,
	const float* shs,
	glm::mat4 viewmatrix,
	glm::mat4 projmatrix,
	glm::vec3 cam_pos,
	const int W, int H,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	int* radii,
	float2* means2D,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	int* tiles_touched)
{
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, D, M,
		orig_points,
		opacities,
		shs,
		viewmatrix, 
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		means2D,
		depths,
		cov3Ds,
		rgb,
		conic_opacity,
		grid,
		tiles_touched);
}
// Forward version of 2D covariance matrix computation
__device__ float3 origin_computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	float3 t = transformPoint4x3(mean, viewmatrix);

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Apply low-pass filter: every Gaussian should be at least
	// one pixel wide/high. Discard 3rd row and column.
	cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;
	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}


// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void origin_computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Covariance is symmetric, only store upper right
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}

// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void filter_preprocessCUDA(int P, int M,
	const float* orig_points,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const int W, int H,
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	int* radii,
	float* cov3Ds,
	const dim3 grid,
	bool prefiltered)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;

	// Perform near culling, quit if outside.
	float3 p_view;
	if (!origin_in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;

	// Transform point by projecting
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

	// If 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters. 
	const float* cov3D;
	if (cov3D_precomp != nullptr)
	{
		cov3D = cov3D_precomp + idx * 6;
	}
	else
	{
		origin_computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
		cov3D = cov3Ds + idx * 6;
	}

	// Compute 2D screen-space covariance matrix
	float3 cov = origin_computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

	// Invert covariance (EWA algorithm)
	float det = (cov.x * cov.z - cov.y * cov.y);
	if (det == 0.0f)
		return;


	// Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles. 
	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
	float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
	uint2 rect_min, rect_max;
	origin_getRect(point_image, my_radius, rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;


	radii[idx] = my_radius;
}


void FORWARD::filter_preprocess(int P, int M,
	const float* means3D,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const int W, int H,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	int* radii,
	float* cov3Ds,
	const dim3 grid,
	bool prefiltered)
{

	filter_preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		cov3D_precomp,
		viewmatrix, 
		projmatrix,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		cov3Ds,
		grid,
		prefiltered
		);
}
