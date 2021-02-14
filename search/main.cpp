/*
* Copyright 2020 Ahmad Obeid - Khalifa University.  Supervised by Dr. Ibrahim Elfadel.
* All rights reserved.
*/
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <helper_cuda.h>
#include <iostream>
#include <fstream>
#include <memory>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "search_common.h"
#include <chrono> 
using namespace std;
using namespace std::chrono;

int main(int argc, char **argv)
{
	string wanted_file = argv[1];
	int d = (int)strtol(argv[2], NULL, 10); 



	float *h_data;
	long filelen1;
	FILE *fileptr1;
	string data_file = "./datasets/" + wanted_file + ".data";
	fileptr1 = fopen(data_file.c_str(), "rb");
	if (!fileptr1)
		printf("Unable to open file...");
	fseek(fileptr1, 0, SEEK_END); // Jump to the end of the file
	filelen1 = ftell(fileptr1); // Get the current byte offset in the file
								//printf("File length is: %d\n",filelen1);
	rewind(fileptr1); // Jump back to the beginning of the file
	int N = filelen1 / (d * 4);
	printf("\nThe number of samples is: %d\n", N);
	h_data = (float *)malloc(filelen1);
	fread(h_data, 4, filelen1, fileptr1); // Read in the entire file


	float *d_data;
	string out_file = "./logs/" + wanted_file + "_log.txt";
	ofstream logfile (out_file.c_str());
	cudaError_t cuda_stat;

	string mission = "Allocating CUDA memory for the data";
	cuda_stat = cudaMalloc((void **)&d_data, filelen1);
	if (cuda_stat != cudaSuccess)
		//printf("Task : %s-- Failure..ERROR %d!\n", mission.c_str(), error);
		logfile << "Task : " << mission.c_str() << "-- Failure..ERROR " << cuda_stat << "!\n";
	else
		//printf("Task : %s-- Success!\n", mission.c_str());
		logfile << "Task : " << mission.c_str() << "-- Success!\n";

	mission = "Copying data from host to device";
	cuda_stat = cudaMemcpy(d_data, h_data, filelen1, cudaMemcpyHostToDevice);
	if (cuda_stat != cudaSuccess)
		//printf("Task : %s-- Failure..ERROR %d!\n", mission.c_str(), error);
		logfile << "Task : " << mission.c_str() << "-- Failure..ERROR " << cuda_stat << "!\n";
	else
		//printf("Task : %s-- Success!\n", mission.c_str());
		logfile << "Task : " << mission.c_str() << "-- Success!\n";

	logfile.close();
	int iterations =  (int)strtol(argv[3], NULL, 10);
	dim3 g(64, 64, 8), b(512, 1, 1);
	dim3 g_long(8, 4, 1), b_long(1024, 1, 1);
	dim3 g_short(1, 1, 1), b_short(32, 1, 1);
	printf("Running %d searches in parallel...\n\n\n", g.x*g.y*g.z*b.x);
	auto start = high_resolution_clock::now();
	search_func(g, g_long, g_short, b, b_long, b_short, N, d_data, iterations, h_data, wanted_file, d);
	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop - start);
	cout << "\n\nThe search is Done\nTotal time is: " << duration.count() / (1e6) << " seconds...\n";
	cout << "Number of searches is: " << iterations * g.x*g.y*g.z*b.x << "\n\n";

	return 0;
}

