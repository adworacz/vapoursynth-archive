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
// Lut

typedef struct {
    VSNodeRef *node;
    const VSVideoInfo *vi;
    void *lut;
    int process[3];
} LutData;

//Each thread perfoms a LUT on a block of 32 / 8 = 4 pixels.
static __global__ void lutKernel8(const uint8_t * __restrict__ srcp, uint8_t * __restrict__ dstp,
                                  const int stride, const int width, const int height,
                                  const uint8_t * __restrict__ lut){
    const int column = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
    const int row = IMAD(blockDim.y, blockIdx.y, threadIdx.y);

    if (column >= width || row >= height)
        return;

    uint32_t src_data = ((uint32_t *)srcp)[(stride / sizeof(uint32_t)) * row + column];
    uint32_t dst_data = 0;

    #pragma unroll 4
    for (int i = 0; i < sizeof(uint32_t); i++) {
       ((uint8_t *)&dst_data)[i] = lut[((uint8_t *)&src_data)[i]];
    }

    ((uint32_t *)dstp)[(stride / sizeof(uint32_t)) * row + column] = dst_data;
}

VS_EXTERN_C int VS_CC lutProcessCUDA(const VSFrameRef *src, VSFrameRef *dst, const VSFormat *fi,
                                     const LutData *d, VSFrameContext *frameCtx, VSCore *core,
                                     const VSAPI *vsapi) {
    int blockSize = VSCUDAGetBasicBlocksize();
    dim3 threads(blockSize, blockSize);

    cudaStream_t stream = vsapi->getStreamForFrame(src, frameCtx, core);

    if (stream == 0) {
        return 0;
    }

    uint8_t *d_lut;
    int numElements = (1 << fi->bitsPerSample);
    CHECKCUDA(cudaMalloc(&d_lut, numElements * sizeof(uint8_t)));
    CHECKCUDA(cudaMemcpy(d_lut, d->lut, numElements, cudaMemcpyHostToDevice));

    for (int plane = 0; plane < fi->numPlanes; plane++) {
        const uint8_t *srcp = vsapi->getReadPtr(src, plane);
        uint8_t *dstp = vsapi->getWritePtr(dst, plane);
        int dst_stride = vsapi->getStride(dst, plane);
        int h = vsapi->getFrameHeight(src, plane);

        if (d->process[plane]) {
            int w = vsapi->getFrameWidth(src, plane);

            if (fi->bytesPerSample == 1) {
                dim3 grid(ceil((float)w / (threads.x * sizeof(uint32_t))), ceil((float)h / threads.y));

                lutKernel8<<<grid, threads, 0, stream>>>(srcp, dstp, dst_stride, w, h, d_lut);
            } else {
                // const uint16_t *lut = (uint16_t *)d->lut;

                // for (hl = 0; hl < h; hl++) {
                //     for (x = 0; x < w; x++)
                //         ((uint16_t *)dstp)[x] =  lut[srcp[x]];

                //     dstp += dst_stride;
                //     srcp += src_stride;
                // }
            }
        }
    }

    CHECKCUDA(cudaFree(d_lut));

    return 1;
}



///////////////////////
// Merge
typedef struct {
    VSNodeRef *node1;
    VSNodeRef *node2;
    const VSVideoInfo *vi;
    int weight[3];
    float fweight[3];
    int process[3];
} MergeData;

//This kernel operates by each thread fetching a stretch of 4 8-bit pixels,
//operating on them, and then sending them back to the destination.
//This is done to achieve coalesced memory accesses, which are crucial for
//high performance in CUDA.
static __global__ void mergeKernel(const uint8_t * __restrict__ srcp1, const uint8_t * __restrict__ srcp2, uint8_t * __restrict__ dstp, const int stride, const int width, const int height, const int weight, const int round, const int MergeShift){
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

VS_EXTERN_C int VS_CC mergeProcessCUDA(const VSFrameRef *src1, const VSFrameRef *src2, VSFrameRef *dst, const MergeData *d, const int MergeShift, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) {
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

                    mergeKernel<<<grid, threads, 0, stream>>>(srcp1, srcp2, dstp, stride, width / sizeof(uint32_t), height, weight, round, MergeShift);
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


//////////////////////////////////////////
// MaskedMerge

typedef struct {
    const VSVideoInfo *vi;
    VSNodeRef *node1;
    VSNodeRef *node2;
    VSNodeRef *mask;
    VSNodeRef *mask23;
    int first_plane;
    int process[3];
} MaskedMergeData;

//This kernel operates by each thread fetching a stretch of 4 8-bit pixels,
//operating on them, and then sending them back to the destination.
//This is done to achieve coalesced memory accesses, which are crucial for
//high performance in CUDA.
static __global__ void maskedMergeKernel(const uint8_t * __restrict__ srcp1, const uint8_t * __restrict__ srcp2, uint8_t * __restrict__ dstp, const int stride, const int width, const int height, const uint8_t * __restrict__ maskp){
    const int column = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
    const int row = IMAD(blockDim.y, blockIdx.y, threadIdx.y);

    if (column >= width || row >= height)
        return;

    const uint32_t src1_data = ((uint32_t *)srcp1)[(stride / sizeof(uint32_t)) * row + column];
    const uint32_t src2_data = ((uint32_t *)srcp2)[(stride / sizeof(uint32_t)) * row + column];
    const uint32_t mask_data = ((uint32_t *)maskp)[(stride / sizeof(uint32_t)) * row + column];
    uint32_t dst_data = 0;

    for (int i = 0; i < sizeof(uint32_t); i++) {
       ((uint8_t *)&dst_data)[i] = ((uint8_t *)&src1_data)[i] + (((((uint8_t *)&src2_data)[i] - ((uint8_t *)&src1_data)[i]) * (((uint8_t *)&mask_data)[i] > 2 ? ((uint8_t *)&mask_data)[i] + 1 : ((uint8_t *)&mask_data)[i]) + 128) >> 8);
    }

    //dstp[x] = srcp1[x] + (((srcp2[x] - srcp1[x]) * (maskp[x] > 2 ? maskp[x] + 1 : maskp[x]) + 128) >> 8);
    ((uint32_t *)dstp)[(stride / sizeof(uint32_t)) * row + column] = dst_data;
}

VS_EXTERN_C int VS_CC maskedMergeProcessCUDA(const VSFrameRef *src1, const VSFrameRef *src2, VSFrameRef *dst, const VSFrameRef *mask, const VSFrameRef *mask23, const MaskedMergeData *d, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) {
    int blockSize = VSCUDAGetBasicBlocksize();
    dim3 threads(blockSize, blockSize);

    cudaStream_t stream = vsapi->getStreamForFrame(src2, frameCtx, core);

    if (stream == 0) {
        return 0;
    }

    for (int plane = 0; plane < d->vi->format->numPlanes; plane++) {
        if (d->process[plane]) {
            int height = vsapi->getFrameHeight(src1, plane);
            int width = vsapi->getFrameWidth(src2, plane);
            int stride = vsapi->getStride(src1, plane);
            const uint8_t *srcp1 = vsapi->getReadPtr(src1, plane);
            const uint8_t *srcp2 = vsapi->getReadPtr(src2, plane);
            const uint8_t *maskp = vsapi->getReadPtr((plane && mask23) ? mask23 : mask, d->first_plane ? 0 : plane);
            uint8_t *dstp = vsapi->getWritePtr(dst, plane);

            if (d->vi->format->sampleType == stInteger) {
                if (d->vi->format->bytesPerSample == 1) {
                    dim3 grid(ceil((float)width / (threads.x * sizeof(uint32_t))), ceil((float)height / threads.y));

                    maskedMergeKernel<<<grid, threads, 0, stream>>>(srcp1, srcp2, dstp, stride, width / sizeof(uint32_t), height, maskp);
                } else if (d->vi->format->bytesPerSample == 2) {
                    // int shift = d->vi->format->bitsPerSample;
                    // int round = 1 << (shift - 1);
                    // for (y = 0; y < h; y++) {
                    //     for (x = 0; x < w; x++)
                    //         ((uint16_t *)dstp)[x] = ((const uint16_t *)srcp1)[x] + (((((const uint16_t *)srcp2)[x]
                    //             - ((const uint16_t *)srcp1)[x]) * (((const uint16_t *)maskp)[x] > 2 ? ((const uint16_t *)maskp)[x] + 1 : ((const uint16_t *)maskp)[x]) + round) >> shift);
                    //     srcp1 += stride;
                    //     srcp2 += stride;
                    //     maskp += stride;
                    //     dstp += stride;
                    // }
                }
            } else if (d->vi->format->sampleType == stFloat) {
                // if (d->vi->format->bytesPerSample == 4) {
                //     for (y = 0; y < h; y++) {
                //         for (x = 0; x < w; x++)
                //             ((float *)dstp)[x] = ((const float *)srcp1)[x] + ((((const float *)srcp2)[x] - ((const float *)srcp1)[x]) * ((const float *)maskp)[x]);
                //         srcp1 += stride;
                //         srcp2 += stride;
                //         maskp += stride;
                //         dstp += stride;
                //     }
                // }
            }
        }
    }

    return 1;
}
