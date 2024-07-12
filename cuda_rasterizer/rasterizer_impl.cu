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

#include "rasterizer_impl.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"
#include <torch/torch.h>
#include <cuda_runtime.h>
#include <nvToolsExt.h>


// Helper function to find the next-highest bit of the MSB
// on the CPU.
int max_allocated_memory_before = 0;

uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}

// Wrapper method to call auxiliary coarse frustum containment test.
// Mark all Gaussians that pass it.
__global__ void checkFrustum(int P,    //P 点的数量
	const float* orig_points,  //点坐标
	const float* viewmatrix,  //视图矩阵 
	const float* projmatrix,  //投影矩阵
	bool* present)  //是否在视角内
{
	auto idx = cg::this_grid().thread_rank();  //每个线程处理一个数据点
	if (idx >= P)
		return;      //每个线程只在P范围内索引

	float3 p_view;   //存储点在视图空间中的坐标
	present[idx] = origin_in_frustum(idx, orig_points, viewmatrix, projmatrix, false, p_view);  //in_frustum函数，检查当前索引处的点是否在视锥体内
}
__device__ static inline bool block_intersect_ellipse(int2 pix_min, int2 pix_max, float2 center, float4 con_o)
{
	float a, b, c, delta, dx, dy;
	int lambda1, lambda2;
	float w = 2.0f * __logf(256.0f * con_o.w);
	
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
	return false;
}

__device__ static inline bool block_contains_center(int2 pix_min, int2 pix_max, float2 center)
{
	return center.x >= pix_min.x && center.x <= pix_max.x && center.y >= pix_min.y && center.y <= pix_max.y;
}

// Generates one key/value pair for all Gaussian / tile overlaps. 
// Run once per Gaussian (1:N mapping).
//键值绑定
__global__ void duplicateWithKeys(
	int P,
	const float2* points_xy,
	const float* depths,
	int* curr_offset,
	uint64_t* gaussian_keys_unsorted,   //用于存储键值对的数组 /64
	uint32_t* gaussian_values_unsorted,  //用于存储键值对的数组
	int* radii,
	const float4* conic_opacity,
	dim3 grid)  //计算键值对的二维网格
{
	int lane = threadIdx.y * blockDim.x + threadIdx.x;
	int warp_id = blockIdx.x * blockDim.z + threadIdx.z;
	int idx_vec = warp_id * 32 + lane;  //每个warp处理一个数据点
	// float amplification_factor = 0.05;
	// const float max_depth_value = 14.0f;  // Assuming the maximum depth value is 50.0
	// const int quantization_levels = 900000000;
	// Generate no key/value pair for invisible Gaussians
	int wh_vec = idx_vec < P ? radii[idx_vec] : 0;
	for (int i = 0; i < 32; i++)
	{
		int wh = __shfl_sync(~0, wh_vec, i);
		if (wh > 0)  //P范围内
		{
			// Find this Gaussian's offset in buffer for writing keys/values.
			int idx = warp_id * 32 + i;
			uint2 rect_min, rect_max;
			//通过3d高斯球投影到2d平面的中心点以及半径，得到投影覆盖了哪些tile
			int width = wh & 0xffff;
			int height = wh >> 16 & 0xffff;
			float2 my_points_xy_image = points_xy[idx];
			float4 my_conic_opacity = conic_opacity[idx];
			getRect(my_points_xy_image, width, height, rect_min, rect_max, grid);

			// For each tile that the bounding rect overlaps, emit a 
			// key/value pair. The key is |  tile ID  |      depth      |,
			// and the value is the ID of the Gaussian. Sorting the values 
			// with this key yields Gaussian IDs in a list, such that they
			// are first sorted by tile and then by depth. 
			__syncwarp();
			for (int y0 = rect_min.y; y0 < rect_max.y; y0 += blockDim.y)   //循环迭代tile范围，为每个tile生成键值对
			{
				int y = y0 + threadIdx.y;
				for (int x0 = rect_min.x; x0 < rect_max.x; x0 += blockDim.x)
				{
					int x = x0 + threadIdx.x;
					bool valid = y < rect_max.y && x < rect_max.x;

					if (valid)
					{
						int2 pix_min = { x * BLOCK_X, y * BLOCK_Y };
						int2 pix_max = { pix_min.x + BLOCK_X - 1, pix_min.y + BLOCK_Y - 1 };
						valid = block_contains_center(pix_min, pix_max, my_points_xy_image) || 
							block_intersect_ellipse(pix_min, pix_max, my_points_xy_image, my_conic_opacity);
					}

					int mask = __ballot_sync(~0, valid);
					if (mask == 0)
					{
						continue;
					}
					int my_offset;
					if (lane == 0)
					{
						my_offset = atomicAdd(curr_offset, __popc(mask));
					}
					my_offset = __shfl_sync(~0, my_offset, 0);
					if (valid)
					{
						int count = __popc(mask & ((1 << lane) - 1));
						uint64_t key = y * grid.x + x;   //得到tile对应本张图片的tile_id 64
						key <<= 32; //32
						// key |= *((uint16_t*)&(__float2half(depths[idx])));  //该高斯球对应的深度 depth_id  32
						// key |= static_cast<uint16_t>((static_cast<int>((depths[idx] / max_depth_value) * quantization_levels) / static_cast<float>(quantization_levels)) * 65535);
						key |= *((uint32_t*)&depths[idx]);
						gaussian_keys_unsorted[my_offset + count] = key;    //key
						gaussian_values_unsorted[my_offset + count] = idx;  //高斯id
					}
				}
			}
		}
	}
}

// Check keys to see if it is at the start/end of one tile's range in 
// the full sorted list. If yes, write start/end of this tile. 
// Run once per instanced (duplicated) Gaussian ID.
//识别排序后的键列表中每个tile范围的起始和结束位置
__global__ void identifyTileRanges(int L, uint64_t* point_list_keys, uint2* ranges)  //L：排序后的键列表的长度  point_list_keys：key列表 输出ranges：存储每个tile范围开始和结束的位置 //64
{
	auto idx = cg::this_grid().thread_rank();  //每个线程处理一个idx，在L范围内
	if (idx >= L)
		return;

	// Read tile ID from key. Update start/end of tile range if at limit.
	uint64_t key = point_list_keys[idx];  //读取当前key，从总提取tile id 64
	uint32_t currtile = key >> 32; //32
	if (idx == 0)
		ranges[currtile].x = 0;  //如果当前索引是列表的第一个元素，将当前tile范围的起始位置设置为 0
	else   //否则，对于其他索引，检查当前tile是否与前一个tile相同
	{
		uint32_t prevtile = point_list_keys[idx - 1] >> 32; //32
		if (currtile != prevtile)    //如果当前tile与前一个tile不同，则更新前一个tile范围的结束位置和当前tile范围的起始位置
		{
			ranges[prevtile].y = idx;
			ranges[currtile].x = idx;
		}
	}
	if (idx == L - 1)
		ranges[currtile].y = L;
}

// Mark Gaussians as visible/invisible, based on view frustum testing
void CudaRasterizer::Rasterizer::markVisible(
	int P,
	float* means3D,
	float* viewmatrix,
	float* projmatrix,
	bool* present)
{
	checkFrustum << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		viewmatrix, projmatrix,
		present);
}

CudaRasterizer::GeometryState CudaRasterizer::GeometryState::fromChunk(char*& chunk, size_t P) //从输入的二进制数据块中解析出geometrystate，获取数组返回一个geometrystate对象
{
	GeometryState geom;
	obtain(chunk, geom.depths, P, 128);
	obtain(chunk, geom.clamped, P * 3, 128);
	obtain(chunk, geom.internal_radii, P, 128);
	obtain(chunk, geom.means2D, P, 128);
	obtain(chunk, geom.cov3D, P * 6, 128);
	obtain(chunk, geom.conic_opacity, P, 128);
	obtain(chunk, geom.rgb, P * 3, 128);
	obtain(chunk, geom.tiles_touched, P, 128);
	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
	obtain(chunk, geom.scanning_space, geom.scan_size, 128);
	obtain(chunk, geom.point_offsets, P, 128);
	return geom;
}

CudaRasterizer::ImageState CudaRasterizer::ImageState::fromChunk(char*& chunk, size_t N)
{
	ImageState img;
	obtain(chunk, img.accum_alpha, N, 128);
	obtain(chunk, img.n_contrib, N, 128);
	obtain(chunk, img.ranges, N, 128);
	return img;
}

CudaRasterizer::BinningState CudaRasterizer::BinningState::fromChunk(char*& chunk, size_t P)
{
	BinningState binning;
	obtain(chunk, binning.point_list, P, 128);
	obtain(chunk, binning.point_list_unsorted, P, 128);
	obtain(chunk, binning.point_list_keys, P, 128);
	obtain(chunk, binning.point_list_keys_unsorted, P, 128);
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.point_list_keys_unsorted, binning.point_list_keys,
		binning.point_list_unsorted, binning.point_list, P);
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	return binning;
}


void printSHS(const float* shs, int size) {
    for (int i = 0; i < 100; ++i) {
        for (int j = 0; j < 16; ++j) {
            for (int k = 0; k < 3; ++k) {
                std::cout << shs[i * 16 * 3 + j * 3 + k] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
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

__global__ void precomputeCov3D(int P, const glm::vec3* scales, float scale_modifier, const glm::vec4* rotations, float* cov3Ds)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;
	computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
}


// Forward rendering procedure for differentiable rasterization
// of Gaussians.
int CudaRasterizer::Rasterizer::forward(
	std::function<char* (size_t)> geometryBuffer,
	std::function<char* (size_t)> binningBuffer,
	std::function<char* (size_t)> imageBuffer,
	const int P, int D, int M,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* opacities,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* cam_pos,
	const float tan_fovx, float tan_fovy,
	const bool prefiltered,
	float* out_color,
	float* out_depth,
	int* radii,
	bool debug)
{
	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);
	size_t chunk_size = required<GeometryState>(P); //计算GeometryState对象所需要的内存空间的大小 
	char* chunkptr = geometryBuffer(chunk_size);  //获取相应大小的数据块
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P);  //从数据块中解析geometrystate

	
	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}
	// std::cout << "\nsupport CudaRasterizer::Rasterizer::forward\n" << std::endl;
	// std::cout << "grid 数量" << ((width + BLOCK_X - 1) / BLOCK_X) << " " <<((height + BLOCK_Y - 1) / BLOCK_Y) << std::endl;
	// std::cout << "block 数量" << BLOCK_X << " " << BLOCK_Y << std::endl;
	// std::cout << "\nsupport CudaRasterizer::Rasterizer::forward\n" << std::endl;

	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1); //网格和线程快的数量
	dim3 block(BLOCK_X, BLOCK_Y, 1);

	size_t img_chunk_size = required<ImageState>(width * height);
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);


	if (NUM_CHANNELS != 3 && colors_precomp == nullptr)
	{
		throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
	}


    // // Copy the result from GPU to host
    // cudaMemcpy(h_shs, shs_, P * 16 * 3 * sizeof(float), cudaMemcpyDeviceToHost);
	// // printf("jisuanhou:\n");
    // // printSHS(h_shs, P);
	// float* h_shs_ = new float[100 * 16 * 3];
	// cudaMemcpy(h_shs_, shs, 100 * 16 * 3 * sizeof(float), cudaMemcpyDeviceToHost);
	// printf("shs:\n");
	// printSHS(h_shs_, P);
	// delete[] h_shs_;

    
    cudaError_t cudaErr = cudaGetLastError();
    if (cudaErr != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(cudaErr));
        // Handle the error as needed
    }

	if (cov3D_precomp == nullptr)
	{
		precomputeCov3D<<<(P + 255) / 256, 256>>>(P, (const glm::vec3*)scales, scale_modifier, (const glm::vec4*)rotations, geomState.cov3D);
		cov3D_precomp = geomState.cov3D;
	}
	glm::mat4 viewmatrix_host;
	glm::mat4 projmatrix_host;
	glm::vec3 cam_pos_host;
	cudaMemcpy(&viewmatrix_host, viewmatrix, sizeof viewmatrix_host, cudaMemcpyDeviceToHost);
	cudaMemcpy(&projmatrix_host, projmatrix, sizeof projmatrix_host, cudaMemcpyDeviceToHost);
	cudaMemcpy(&cam_pos_host, cam_pos, sizeof cam_pos_host, cudaMemcpyDeviceToHost);
	CHECK_CUDA(FORWARD::preprocess(     
		P, D, M,
		means3D,
		// (glm::vec3*)scales,
		// scale_modifier,
		// (glm::vec4*)rotations,
		opacities,
		shs,
		// geomState.clamped,
		// cov3D_precomp,
		// colors_precomp,
		viewmatrix_host,
		projmatrix_host,
		cam_pos_host,
		width, height,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		radii,
		geomState.means2D,
		geomState.depths,
		geomState.cov3D,
		geomState.rgb,
		geomState.conic_opacity,
		tile_grid,
		(int*)geomState.tiles_touched
		// prefiltered
	), debug)

	// Compute prefix sum over full list of touched tile counts by Gaussians
	// E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
	//计算所有高斯点包含的tile的数量的总和

	CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size, geomState.tiles_touched, geomState.point_offsets, P), debug)


	// Retrieve total number of Gaussian instances to launch and resize aux buffers

	int num_rendered;
	//获取在渲染过程中产生的高斯数量

	CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug); 
	//将 geomState.point_offsets 中存储的最后一个元素（即 P - 1 处）的整数值传送到num_rendered 。这个整数表示渲染的高斯数量。


	size_t binning_chunk_size = required<BinningState>(num_rendered);
	char* binning_chunkptr = binningBuffer(binning_chunk_size);
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

	// For each instance to be rendered, produce adequate [ tile | depth ] key 
	// and corresponding dublicated Gaussian indices to be sorted
	int* curr_offset;
	cudaMalloc(&curr_offset, sizeof(int));
	cudaMemset(curr_offset, 0, sizeof(int));
	duplicateWithKeys << <(P + 127) / 128, dim3(8, 4, 4) >> > (
		P,
		geomState.means2D,
		geomState.depths,
		// geomState.point_offsets,
		curr_offset,
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted,
		radii,
		geomState.conic_opacity,
		tile_grid)
	CHECK_CUDA(, debug)
	CHECK_CUDA(cudaMemcpy(&num_rendered, curr_offset, sizeof(int), cudaMemcpyDeviceToHost), debug);
	cudaFree(curr_offset);

	// CHECK_CUDA(cudaDeviceSynchronize(), debug);

	int bit = getHigherMsb(tile_grid.x * tile_grid.y);

	// Sort complete list of (duplicated) Gaussian indices by keys

	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
		binningState.list_sorting_space,
		binningState.sorting_size,
		binningState.point_list_keys_unsorted, binningState.point_list_keys,
		binningState.point_list_unsorted, binningState.point_list,
		num_rendered, 0, 32 + bit), debug) //32


	CHECK_CUDA(cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)), debug);


	// nvtxRangePushA("identifyTileRanges");
	// Identify start and end of per-tile workloads in sorted list
	if (num_rendered > 0)
		identifyTileRanges << <(num_rendered + 255) / 256, 256 >> > (
			num_rendered,
			binningState.point_list_keys,
			imgState.ranges);

	// CHECK_CUDA(cudaDeviceSynchronize(), debug);
	// Let each tile blend its range of Gaussians independently in parallel
	const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;
	
	CHECK_CUDA(FORWARD::render(
		tile_grid, block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		geomState.means2D,
		feature_ptr,
		geomState.depths,
		geomState.conic_opacity,
		imgState.accum_alpha,
		imgState.n_contrib,
		background,
		out_color,
		out_depth), debug)
	// cudaFree(shs);
	CHECK_CUDA(cudaDeviceSynchronize(), debug);

	// printf("num_rendered = %d\n", num_rendered);
	return num_rendered;
}


// Produce necessary gradients for optimization, corresponding
// to forward render pass

void CudaRasterizer::Rasterizer::backward(
	const int P, int D, int M, int R,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* campos,
	const float tan_fovx, float tan_fovy,
	const int* radii,
	char* geom_buffer,
	char* binning_buffer,
	char* img_buffer,
	const float* dL_dpix,
	float* dL_dmean2D,
	float* dL_dconic,
	float* dL_dopacity,
	float* dL_dcolor,
	float* dL_dmean3D,
	float* dL_dcov3D,
	float* dL_dsh,
	float* dL_dscale,
	float* dL_drot,
	bool debug)
{

	GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);
	BinningState binningState = BinningState::fromChunk(binning_buffer, R);
	ImageState imgState = ImageState::fromChunk(img_buffer, width * height);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	const dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Compute loss gradients w.r.t. 2D mean position, conic matrix,
	// opacity and RGB of Gaussians from per-pixel loss gradients.
	// If we were given precomputed colors and not SHs, use them.
	const float* color_ptr = (colors_precomp != nullptr) ? colors_precomp : geomState.rgb;
	const float* depth_ptr = geomState.depths;
	CHECK_CUDA(BACKWARD::render(
		tile_grid,
		block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		background,
		geomState.means2D,
		geomState.conic_opacity,
		color_ptr,
		imgState.accum_alpha,
		imgState.n_contrib,
		dL_dpix,
		(float3*)dL_dmean2D,
		(float4*)dL_dconic,
		dL_dopacity,
		dL_dcolor), debug)

	// Take care of the rest of preprocessing. Was the precomputed covariance
	// given to us or a scales/rot pair? If precomputed, pass that. If not,
	// use the one we computed ourselves.
	const float* cov3D_ptr = (cov3D_precomp != nullptr) ? cov3D_precomp : geomState.cov3D;
	CHECK_CUDA(BACKWARD::preprocess(P, D, M,
		(float3*)means3D,
		radii,
		shs,
		geomState.clamped,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		cov3D_ptr,
		viewmatrix,
		projmatrix,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		(glm::vec3*)campos,
		(float3*)dL_dmean2D,
		dL_dconic,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_dcov3D,
		dL_dsh,
		(glm::vec3*)dL_dscale,
		(glm::vec4*)dL_drot), debug)

}

void CudaRasterizer::Rasterizer::visible_filter(
	std::function<char* (size_t)> geometryBuffer,
	std::function<char* (size_t)> binningBuffer,
	std::function<char* (size_t)> imageBuffer,
	const int P, int M,
	const int width, int height,
	const float* means3D,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float tan_fovx, float tan_fovy,
	const bool prefiltered,
	int* radii,
	bool debug)
{
	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	size_t chunk_size = required<GeometryState>(P);
	char* chunkptr = geometryBuffer(chunk_size);
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	// dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Dynamically resize image-based auxiliary buffers during training
	size_t img_chunk_size = required<ImageState>(width * height);
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

	// Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)
	CHECK_CUDA(FORWARD::filter_preprocess(
		P, M,
		means3D,
		(glm::vec3*)scales,
		scale_modifier,
		(glm::vec4*)rotations,
		cov3D_precomp,
		viewmatrix, projmatrix,
		width, height,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		radii,
		geomState.cov3D,
		tile_grid,
		prefiltered
	), debug)

}
