/*
* Copyright (c) 2012 Fredrik Mellbin, Austin Dworaczyk Wiltshire
*
* This file is part of VapourSynth.
*
* VapourSynth is free software; you can redistribute it and/or
* modify it under the terms of the GNU Lesser General Public
* License as published by the Free Software Foundation; either
* version 2.1 of the License, or (at your option) any later version.
*
* VapourSynth is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
* Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public
* License along with VapourSynth; if not, write to the Free Software
* Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
*/

// Contains the CUDA implementations of the simplefilters operations.
#include "VapourSynth.h"
#include "VSCuda.h"

///////////////////////
// Merge


//This kernel operates by each thread fetching a stretch of 4 8-bit pixels,
//operating on them, and then sending them back to the destination.
//This is done to achieve coalesced memory accesses, which are crucial for
//high performance in CUDA.
static __global__ void mergeKernel(uint8_t *dstp, const uint8_t *srcp1, const uint8_t *srcp2, const int stride, const int width, const int height, const int weight, const int round, const int MergeShift){
    const int column = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
    const int row = IMAD(blockDim.y, blockIdx.y, threadIdx.y);

    if (column >= width || row >= height)
        return;

   uint32_t src1_data = ((uint32_t *)srcp1)[(stride / sizeof(uint32_t)) * row + column];
   uint32_t src2_data = ((uint32_t *)srcp2)[(stride / sizeof(uint32_t)) * row + column];
   uint32_t dst_data = 0;

   for (int i = 0; i < sizeof(uint32_t); i++) {
      ((uint8_t *)&dst_data)[i] = ((uint8_t *)&src1_data)[i] + (((((uint8_t *)&src2_data)[i] - ((uint8_t *)&src1_data)[i]) * weight + round) >> MergeShift);
   }

   //dstp[x] = srcp1[x] + (((srcp2[x] - srcp1[x]) * weight + round) >> MergeShift);
   ((uint32_t *)dstp)[(stride / sizeof(uint32_t)) * row + column] = dst_data;
}

VS_EXTERN_C void VS_CC mergeProcessCUDA(uint8_t *dstp, const uint8_t *srcp1, const uint8_t *srcp2, const int stride, const int width, const int height, const int weight, const int round, const int MergeShift) {
   cudaDeviceProp * deviceProp = VSCUDAGetDefaultDeviceProperties();

   int blockSize = (deviceProp->major < 2) ? 16 : 32;

   dim3 threads(blockSize, blockSize);
   dim3 grid(ceil((float)width / (threads.x * sizeof(uint32_t))), ceil((float)height / threads.y));

   mergeKernel<<<grid, threads>>>(dstp, srcp1, srcp2, stride, width / sizeof(uint32_t), height, weight, round, MergeShift);
}

