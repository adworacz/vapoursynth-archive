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
static __global__ void mergeKernel(uint8_t * __restrict__ dstp, const uint8_t * __restrict__ srcp1, const uint8_t * __restrict__ srcp2, const int stride, const int width, const int height, const int weight, const int round, const int MergeShift){
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

typedef struct {
    VSNodeRef *node1;
    VSNodeRef *node2;
    const VSVideoInfo *vi;
    int weight[3];
    float fweight[3];
    int process[3];
} MergeData;

VS_EXTERN_C int VS_CC mergeProcessCUDA(const VSFrameRef *src1, const VSFrameRef *src2, VSFrameRef *dst, const int *pl, const VSFrameRef **fr, const MergeData *d, const int MergeShift, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) {
    int blockSize = VSCUDAGetBasicBlocksize();
    dim3 threads(blockSize, blockSize);

    cudaStream_t stream = vsapi->getStreamForFrame(src2, frameCtx, core);

    if (stream == 0) {
        return 0;
    }

    for (int plane = 0; plane < d->vi->format->numPlanes; plane++) {
        if (d->process[plane] == 0) {
            int weight = d->weight[plane];
            // float fweight = d->fweight[plane];
            int height = vsapi->getFrameHeight(src1, plane);
            int width = vsapi->getFrameWidth(src2, plane);
            int stride = vsapi->getStride(src1, plane);
            const uint8_t *srcp1 = vsapi->getReadPtr(src1, plane);
            const uint8_t *srcp2 = vsapi->getReadPtr(src2, plane);
            uint8_t *dstp = vsapi->getWritePtr(dst, plane);

            if (d->vi->format->sampleType == stInteger) {
                const int round = 1 << (MergeShift - 1);
                if (d->vi->format->bytesPerSample == 1) {
                    dim3 grid(ceil((float)width / (threads.x * sizeof(uint32_t))), ceil((float)height / threads.y));

                    mergeKernel<<<grid, threads, 0, stream>>>(dstp, srcp1, srcp2, stride, width / sizeof(uint32_t), height, weight, round, MergeShift);
                } else if (d->vi->format->bytesPerSample == 2) {
                  // const int round = 1 << (MergeShift - 1);
                  // for (y = 0; y < h; y++) {
                  //     for (x = 0; x < w; x++)
                  //         ((uint16_t *)dstp)[x] = ((const uint16_t *)srcp1)[x] + (((((const uint16_t *)srcp2)[x] - ((const uint16_t *)srcp1)[x]) * weight + round) >> MergeShift);
                  //     srcp1 += stride;
                  //     srcp2 += stride;
                  //     dstp += stride;
                  // }
                }
            } else if (d->vi->format->sampleType == stFloat) {
              // if (d->vi->format->bytesPerSample == 4) {
              //     for (y = 0; y < h; y++) {
              //         for (x = 0; x < w; x++)
              //             ((float *)dstp)[x] = (((const float *)srcp1)[x] + (((const float *)srcp2)[x] - ((const float *)srcp1)[x]) * fweight);
              //         srcp1 += stride;
              //         srcp2 += stride;
              //         dstp += stride;
              //     }
              // }
            }
        }
    }

    return 1;
}
