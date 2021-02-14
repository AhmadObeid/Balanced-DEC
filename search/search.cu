/*
* Copyright 2020 Ahmad Obeid - Khalifa University.  Supervised by Dr. Ibrahim Elfadel.
* All rights reserved.
*/

#include <iostream>
#include <fstream>
#include <assert.h>
#include <cooperative_groups.h>
#include <helper_cuda.h>
#include "search_common.h"
#include <stdlib.h>
#include <stdio.h>
#include <curand_kernel.h>
#define K 9 // change this if needed
#define serial_iteration 1 //change this if needed

using namespace std;

////////////////////////////////////////////////////////////////////////////////
// CUDA kernels
////////////////////////////////////////////////////////////////////////////////

__device__ unsigned int WangHash(unsigned int a) {
	a = (a ^ 61) ^ (a >> 16);
	a = a + (a << 3);
	a = a ^ (a >> 4);
	a = a * 0x27d4eb2d;
	a = a ^ (a >> 15);
	return a;
}


__global__ void max_reduction_final(
	float *  d_distance,
	int * ultimate_idx
	)
{
	////////////// Thread identification ////////////
	int tx = threadIdx.x;

	__shared__ float local_distance[serial_iteration];
	__shared__ int maximal_idx[serial_iteration];
	maximal_idx[threadIdx.x] = threadIdx.x;
	local_distance[threadIdx.x] = d_distance[tx];

	////////////// Starting max reduction ////////////
	for (int stride = 1; stride < blockDim.x; stride *= 2) {
		__syncthreads();
		if ((threadIdx.x % (2 * stride)) == 0) {
			if (local_distance[threadIdx.x]>local_distance[threadIdx.x + stride]) {
				local_distance[threadIdx.x] = local_distance[threadIdx.x];
				maximal_idx[threadIdx.x] = maximal_idx[threadIdx.x];
			}
			else {
				local_distance[threadIdx.x] = local_distance[threadIdx.x + stride];
				maximal_idx[threadIdx.x] = maximal_idx[threadIdx.x + stride];
			}
		}
	}
	__syncthreads();
	////////////// First thread of block to submit the maximum smallest separation ////////////
	if (threadIdx.x == 0) {
		d_distance[threadIdx.x] = local_distance[threadIdx.x];
		ultimate_idx[threadIdx.x] = maximal_idx[threadIdx.x];
		
		

		}
		
	
	
}



__global__ void max_reduction_short(
	float *  elected_distance,
	int * elected_groups
	)
{
	////////////// Thread identification ////////////
	int tx = threadIdx.x + blockIdx.x*blockDim.x;
	int ty = threadIdx.y + blockIdx.y*blockDim.y;
	int num_threads_x = blockDim.x*gridDim.x;
	int idx = ty*num_threads_x + tx;

	__shared__ float local_distance[32];
	__shared__ int maximal_idx[32];
	maximal_idx[threadIdx.x] = threadIdx.x;
	local_distance[threadIdx.x] = elected_distance[idx];

	////////////// Starting max reduction ////////////
	for (int stride = 1; stride < blockDim.x; stride *= 2) {
		__syncthreads();
		if ((threadIdx.x % (2 * stride)) == 0) {
			if (local_distance[threadIdx.x]>local_distance[threadIdx.x + stride]) {
				local_distance[threadIdx.x] = local_distance[threadIdx.x];
				maximal_idx[threadIdx.x] = maximal_idx[threadIdx.x];
			}
			else {
				local_distance[threadIdx.x] = local_distance[threadIdx.x + stride];
				maximal_idx[threadIdx.x] = maximal_idx[threadIdx.x + stride];
			}
		}
	}
	__syncthreads();
	////////////// First thread of block to submit the maximum smallest separation ////////////
	if (threadIdx.x == 0) {
		elected_distance[blockIdx.y*gridDim.x + blockIdx.x] = local_distance[threadIdx.x];
		
		for (int i = 0; i<K; i++) {
			elected_groups[(blockIdx.x)*K + i] =
				elected_groups[blockIdx.x*K*blockDim.x + maximal_idx[0] * K + i];

		}
		
	}
	
}




__global__ void max_reduction_long(
	float *  elected_distance,
	int * elected_groups
	)
{
	////////////// Thread identification ////////////
	int tx = threadIdx.x + blockIdx.x*blockDim.x;
	int ty = threadIdx.y + blockIdx.y*blockDim.y;
	int num_threads_x = blockDim.x*gridDim.x;
	int idx = ty*num_threads_x + tx;

	__shared__ float local_distance[1024];
	__shared__ int maximal_idx[1024];
	maximal_idx[threadIdx.x] = threadIdx.x;
	local_distance[threadIdx.x] = elected_distance[idx];
	
	////////////// Starting max reduction ////////////
	for (int stride = 1; stride < blockDim.x; stride *= 2) {
		__syncthreads();
		if ((threadIdx.x % (2 * stride)) == 0) {
			if (local_distance[threadIdx.x]>local_distance[threadIdx.x + stride]) {
				local_distance[threadIdx.x] = local_distance[threadIdx.x];
				maximal_idx[threadIdx.x] = maximal_idx[threadIdx.x];
			}
			else {
				local_distance[threadIdx.x] = local_distance[threadIdx.x + stride];
				maximal_idx[threadIdx.x] = maximal_idx[threadIdx.x + stride];
			}
		}
	}
	__syncthreads();
	////////////// First thread of block to submit the maximum smallest separation ////////////
	if (threadIdx.x == 0) {
		elected_distance[blockIdx.y*gridDim.x + blockIdx.x] = local_distance[threadIdx.x];
		
		for (int i = 0; i<K; i++) {
			elected_groups[(blockIdx.y*gridDim.x + blockIdx.x)*K + i] =
				elected_groups[(blockIdx.y*gridDim.x + blockIdx.x)*K*blockDim.x + maximal_idx[0] * K + i];

		}
	}
	
}


__global__ void search_kernel(
	int N,
	float *d_data,
	curandState *states,
	float *  elected_distance,
	int * elected_groups,	
	int d)
{
	////////////// Thread identification ////////////

	int tx = threadIdx.x + blockIdx.x*blockDim.x;
	int ty = threadIdx.y + blockIdx.y*blockDim.y;
	int tz = threadIdx.z + blockIdx.z*blockDim.z;
	int num_threads_x = blockDim.x*gridDim.x;
	int num_threads_y = blockDim.y*gridDim.y;
	int num_all_threads = num_threads_x*num_threads_y;
	int idx = ty*num_threads_x + tz*num_all_threads + tx;
	
	////////////// Randomizing the choice of groups via cuRand ////////////
	__shared__ int group[K][512]; //shared mem is used to avoid register spilling.

								   
	unsigned int seed = (unsigned int)clock64();
	curand_init(WangHash(seed) + idx, 0, 0, &states[idx]);

	for (int i = 0; i<K; i++) {
		group[i][threadIdx.x] = (int)(curand_uniform(&states[idx])*N) * d;
	}
	__syncthreads();


	__shared__ float min_distance[512]; //put in shared mem so that min_distance of each thread is seen over the whole block,
										//in order to start the max-reduction from the granularity of a thread block.
	min_distance[threadIdx.x] = 1e3;

	////////////// Get the minimum separation of each group ////////////
	float distance = 0;
	float idx1, idx2;
	for (int i = 0; i<K - 1; i++) {
		for (int ii = i + 1; ii<K; ii++) {
			for (int j = 0; j<d; j++) {
				idx1 = d_data[group[i][threadIdx.x] + j];
				idx2 = d_data[group[ii][threadIdx.x] + j];
				distance += (idx1 - idx2)*(idx1 - idx2);
				
			}
			distance = sqrt(distance);
					min_distance[threadIdx.x] = (distance<min_distance[threadIdx.x]) ? distance : min_distance[threadIdx.x];
			distance = 0;
		}
	}
	__syncthreads();


	__shared__ int maximal_idx[512]; //The index of maximally separated group
	maximal_idx[threadIdx.x] = threadIdx.x;
	////////////// Starting local max reduction ////////////
	for (int stride = 1; stride < blockDim.x; stride *= 2) {
		__syncthreads();
		if ((threadIdx.x % (2 * stride)) == 0) {
			if (min_distance[threadIdx.x]>min_distance[threadIdx.x + stride]) {
				min_distance[threadIdx.x] = min_distance[threadIdx.x];
				maximal_idx[threadIdx.x] = maximal_idx[threadIdx.x];
			}
			else {
				min_distance[threadIdx.x] = min_distance[threadIdx.x + stride];
				maximal_idx[threadIdx.x] = maximal_idx[threadIdx.x + stride];
			}
		}
	}
	__syncthreads();
	////////////// First thread of block to submit the maximum smallest separation ////////////
	if (threadIdx.x == 0) {
		elected_distance[blockIdx.z*gridDim.x*gridDim.y + blockIdx.y*gridDim.x + blockIdx.x] = min_distance[threadIdx.x];


		for (int i = 0; i<K; i++) {
			elected_groups[(blockIdx.z*gridDim.x*gridDim.y + blockIdx.y*gridDim.x + blockIdx.x)*K + i] =
				group[i][maximal_idx[0]];

		}
	}
	

}
/*

extern "C" void cuda_err(
	cudaError_t error,
	string mission,
	ofstream file)
{
	if (error != cudaSuccess)
		//printf("Task : %s-- Failure..ERROR %d!\n", mission.c_str(), error);
		file << "Task : " << mission.c_str() << "-- Failure..ERROR " << cuda_stat << "!\n";
	else
		//printf("Task : %s-- Success!\n", mission.c_str());
		file << "Task : " << mission.c_str() << "-- Success!\n";
}
*/

extern "C" void search_func(
	dim3 g,
	dim3 g_long,
	dim3 g_short,
	dim3 b,
	dim3 b_long,
	dim3 b_short,
	int N,
	float *d_data,
	int iterations,
	float *h_data,
	string wanted_file,	
	int d
	)
{
	
	float *elected_distance;
	int *elected_groups;
	static curandState *states = NULL;
	float *h_distances;
	int *h_groups;	
	cudaError_t cuda_stat;
	ofstream logfile;
	string out_file = "./logs/" + wanted_file + "_log.txt";
	logfile.open (out_file.c_str(), ios::out | ios::app);

	h_distances = (float *)malloc(sizeof(float)*iterations);
	h_groups = (int *)malloc(sizeof(int)*K*iterations);
	for (int i = 0; i< iterations; i++) {

		logfile << "\n........Starting iteration " << i << ".........\n";
		printf("........Starting iteration %d.........\n", i);
		string mission = "Allocating memory for randomizor";
		cuda_stat = cudaMalloc((void **)&states, sizeof(curandState) *
			g.x*g.y*g.z*b.x*b.y*b.z);
		if (cuda_stat != cudaSuccess)
		//printf("Task : %s-- Failure..ERROR %d!\n", mission.c_str(), error);
		logfile << "Task : " << mission.c_str() << "-- Failure..ERROR " << cuda_stat << "!\n";
		else
		//printf("Task : %s-- Success!\n", mission.c_str());
		logfile << "Task : " << mission.c_str() << "-- Success!\n";


		mission = "Allocating memory for elected distances";
		cuda_stat = cudaMalloc((void **)&elected_distance, sizeof(float) *
			g.x*g.y*g.z);
		if (cuda_stat != cudaSuccess)
		//printf("Task : %s-- Failure..ERROR %d!\n", mission.c_str(), error);
		logfile << "Task : " << mission.c_str() << "-- Failure..ERROR " << cuda_stat << "!\n";
		else
		//printf("Task : %s-- Success!\n", mission.c_str());
		logfile << "Task : " << mission.c_str() << "-- Success!\n";


		mission = "Allocating memory for elected groups";
		cuda_stat = cudaMalloc((void **)&elected_groups, sizeof(int) *
			g.x*g.y*g.z*K);
		if (cuda_stat != cudaSuccess)
		//printf("Task : %s-- Failure..ERROR %d!\n", mission.c_str(), error);
		logfile << "Task : " << mission.c_str() << "-- Failure..ERROR " << cuda_stat << "!\n";
		else
		//printf("Task : %s-- Success!\n", mission.c_str());
		logfile << "Task : " << mission.c_str() << "-- Success!\n";

		search_kernel << <g, b >> >(N, d_data, states, elected_distance, elected_groups,d);
		mission = "Synchorinze device - 1";
		cuda_stat = cudaDeviceSynchronize();
		if (cuda_stat != cudaSuccess)
		//printf("Task : %s-- Failure..ERROR %d!\n", mission.c_str(), error);
		logfile << "Task : " << mission.c_str() << "-- Failure..ERROR " << cuda_stat << "!\n";
		else
		//printf("Task : %s-- Success!\n", mission.c_str());
		logfile << "Task : " << mission.c_str() << "-- Success!\n";

		
		//printf("\nInitial search is done; now starting the max reduction....\n");
		//printf("[||||||||||||||                                           ]\n\n");

		max_reduction_long << <g_long, b_long >> >(elected_distance, elected_groups);
		mission = "Synchorinze device - 2";
		cuda_stat = cudaDeviceSynchronize();
		if (cuda_stat != cudaSuccess)
		//printf("Task : %s-- Failure..ERROR %d!\n", mission.c_str(), error);
		logfile << "Task : " << mission.c_str() << "-- Failure..ERROR " << cuda_stat << "!\n";
		else

		//printf("Task : %s-- Success!\n", mission.c_str());
		logfile << "Task : " << mission.c_str() << "-- Success!\n";

		//printf("max reduction phase 1 is done; now starting phase 2....\n");
		//printf("[|||||||||||||||||||||||||||||||||||                      ]\n\n");

		max_reduction_short << <g_short, b_short >> >(elected_distance, elected_groups);
		mission = "Synchorinze device - 3";
		cuda_stat = cudaDeviceSynchronize();
		if (cuda_stat != cudaSuccess)
		//printf("Task : %s-- Failure..ERROR %d!\n", mission.c_str(), error);
		logfile << "Task : " << mission.c_str() << "-- Failure..ERROR " << cuda_stat << "!\n";
		else
		//printf("Task : %s-- Success!\n", mission.c_str());
		logfile << "Task : " << mission.c_str() << "-- Success!\n";

		//printf("max reduction phase 2 is done. Fetching Results....\n");
		//printf("[|||||||||||||||||||||||||||||||||||||||||||||||||||||||| ]\n\n");

		mission = "Saving maximum distance";
		cuda_stat = cudaMemcpy(h_distances + i, elected_distance, sizeof(float), cudaMemcpyDeviceToHost);
		if (cuda_stat != cudaSuccess)
		//printf("Task : %s-- Failure..ERROR %d!\n", mission.c_str(), error);
		logfile << "Task : " << mission.c_str() << "-- Failure..ERROR " << cuda_stat << "!\n";
		else
		//printf("Task : %s-- Success!\n", mission.c_str());
		logfile << "Task : " << mission.c_str() << "-- Success!\n";


		mission = "Saving maximal group";
		cuda_stat = cudaMemcpy(h_groups + i*K, elected_groups, sizeof(int)*K,
			cudaMemcpyDeviceToHost);	
	
		if (cuda_stat != cudaSuccess)
		//printf("Task : %s-- Failure..ERROR %d!\n", mission.c_str(), error);
		logfile << "Task : " << mission.c_str() << "-- Failure..ERROR " << cuda_stat << "!\n";
		else
		//printf("Task : %s-- Success!\n", mission.c_str());
		logfile << "Task : " << mission.c_str() << "-- Success!\n";

		
		
		logfile << "\n\n{{{{{{{-----------RESULTS------------}}}}}}}\n\n";
		logfile << "Distance is :" << h_distances[i] << "\n\n";
		logfile << "The group is:\n\n";

		logfile << "[";
		for (int j = 0; j<K; j++) {
			logfile << h_groups[i*K + j]/d + 1 << ", "; 
			if (j == K / 2) logfile << "\n";
		}
		logfile << "]\n\n";

		cudaFree(elected_distance);
		cudaFree(elected_groups);
		cudaFree(states);
	}
if (iterations != 1){
	logfile << "-------------------------------------------\n";
	logfile << "\n Getting results of final election";
	float *d_distances;
	int *ultimate_idx, *ultimate_idx_h = (int *)malloc(sizeof(int));

	string mission = "Allocating memory for final election";
	cuda_stat = cudaMalloc((void **)&d_distances, sizeof(float) *iterations);
	if (cuda_stat != cudaSuccess)
	//printf("Task : %s-- Failure..ERROR %d!\n", mission.c_str(), error);
	logfile << "Task : " << mission.c_str() << "-- Failure..ERROR " << cuda_stat << "!\n";
	else
	//printf("Task : %s-- Success!\n", mission.c_str());
	logfile << "Task : " << mission.c_str() << "-- Success!\n";
	

	mission = "Copying data from host to device";
	cuda_stat = cudaMemcpy(d_distances, h_distances, sizeof(float)*iterations, cudaMemcpyHostToDevice);
	if (cuda_stat != cudaSuccess)
		//printf("Task : %s-- Failure..ERROR %d!\n", mission.c_str(), error);
		logfile << "Task : " << mission.c_str() << "-- Failure..ERROR " << cuda_stat << 	"!\n";
	else
		//printf("Task : %s-- Success!\n", mission.c_str());
		logfile << "Task : " << mission.c_str() << "-- Success!\n";


	mission = "Allocating memory for best index";
	cuda_stat = cudaMalloc((void **)&ultimate_idx, sizeof(int));
	if (cuda_stat != cudaSuccess)
	//printf("Task : %s-- Failure..ERROR %d!\n", mission.c_str(), error);
	logfile << "Task : " << mission.c_str() << "-- Failure..ERROR " << cuda_stat << "!\n";
	else
	//printf("Task : %s-- Success!\n", mission.c_str());
	logfile << "Task : " << mission.c_str() << "-- Success!\n";
	max_reduction_final<<<1,iterations>>>(d_distances, ultimate_idx);

	mission = "Saving ultimate distance";
	cuda_stat = cudaMemcpy(h_distances, d_distances, sizeof(float), cudaMemcpyDeviceToHost);
	if (cuda_stat != cudaSuccess)
	//printf("Task : %s-- Failure..ERROR %d!\n", mission.c_str(), error);
	logfile << "Task : " << mission.c_str() << "-- Failure..ERROR " << cuda_stat << "!\n";
	else
	//printf("Task : %s-- Success!\n", mission.c_str());
	logfile << "Task : " << mission.c_str() << "-- Success!\n";


	mission = "Saving ultimate index";
	cuda_stat = cudaMemcpy(ultimate_idx_h, ultimate_idx, sizeof(int),
		cudaMemcpyDeviceToHost);
	if (cuda_stat != cudaSuccess)
	//printf("Task : %s-- Failure..ERROR %d!\n", mission.c_str(), error);
	logfile << "Task : " << mission.c_str() << "-- Failure..ERROR " << cuda_stat << "!\n";
	else
	//printf("Task : %s-- Success!\n", mission.c_str());
	logfile << "Task : " << mission.c_str() << "-- Success!\n";
	logfile << "Max distance is: " << h_distances[0] << "\n";
	logfile << "Which is obtained by the group: ";
	logfile << "[";
		for (int j = 0; j<K; j++) {
			logfile << h_groups[ultimate_idx_h[0]*K + j]/d + 1 << ", "; 
			if (j == K / 2) logfile << "\n";
		}
		logfile << "]\n\n";
cudaFree(d_distances);
}
	cudaFree(d_data);
	
	free(h_distances);
	free(h_groups);
	free(h_data);


}

















