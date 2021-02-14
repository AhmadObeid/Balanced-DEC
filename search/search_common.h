/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#ifndef SEARCH_COMMON_H
#define SEARCH_COMMON_H

#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <stdio.h>
#include <stdlib.h>


using namespace std;
////////////////////////////////////////////////////////////////////////////////
// Shortcut typename
////////////////////////////////////////////////////////////////////////////////
typedef unsigned int uint;


////////////////////////////////////////////////////////////////////////////////
// CUDA SEARCH
////////////////////////////////////////////////////////////////////////////////


extern "C" void search_func(
    dim3 g, dim3 g_long, dim3 g_short, dim3 b, dim3 b_long, dim3 b_short, int N, float *d_data, int iterations, float *h_data, string wanted_file, int d
);
/*
extern "C" void cuda_err(cudaError_t error, string, ofstream file);
*/
////////////////////////////////////////////////////////////////////////////////



#endif
