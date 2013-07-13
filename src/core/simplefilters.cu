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


//////////////////////////////////////////
// AddBorders
typedef struct {
    VSNodeRef *node;
    const VSVideoInfo *vi;
    int left;
    int right;
    int top;
    int bottom;
    union {
        uint32_t i[3];
        float f[3];
    } color;
} AddBordersData;


VS_EXTERN_C void VS_CC addBordersProcessCUDA(const VSFrameRef *src, VSFrameRef *dst, const AddBordersData *d,
                                            VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) {

    for (int plane = 0; plane < d->vi->format->numPlanes; plane++) {
        int rowsize = vsapi->getFrameWidth(src, plane) * d->vi->format->bytesPerSample;
        int srcstride = vsapi->getStride(src, plane);
        int dststride = vsapi->getStride(dst, plane);
        int srcheight = vsapi->getFrameHeight(src, plane);
        const uint8_t *srcdata = vsapi->getReadPtr(src, plane);
        uint8_t *dstdata = vsapi->getWritePtr(dst, plane);
        int padt = d->top >> (plane ? d->vi->format->subSamplingH : 0);
        int padb = d->bottom >> (plane ? d->vi->format->subSamplingH : 0);
        int padl = (d->left >> (plane ? d->vi->format->subSamplingW : 0)) * d->vi->format->bytesPerSample;
        int padr = (d->right >> (plane ? d->vi->format->subSamplingW : 0)) * d->vi->format->bytesPerSample;
        int color = d->color.i[plane];
        const VSCUDAStream *stream = vsapi->getStream(dst, plane);

        // Pad TOP
        switch (d->vi->format->bytesPerSample) {
        case 1:
            CHECKCUDA(cudaMemset2DAsync(dstdata, dststride, color, rowsize + padl + padr, padt, stream->stream));
            break;
        // case 2:
        //     vs_memset16(dstdata, color, padt * dststride / 2);
        //     break;
        // case 4:
        //     vs_memset32(dstdata, color, padt * dststride / 4);
        //     break;
        }
        dstdata += padt * dststride;

        // Pad LEFT/RIGHT
        switch (d->vi->format->bytesPerSample) {
        case 1:
            CHECKCUDA(cudaMemset2DAsync(dstdata, dststride, color, padl, srcheight, stream->stream)); //Maybe remove the Async, have had problems in the past.
            CHECKCUDA(cudaMemcpy2DAsync(dstdata + padl, dststride, srcdata, srcstride, rowsize, srcheight, cudaMemcpyDeviceToDevice, stream->stream));
            CHECKCUDA(cudaMemset2DAsync(dstdata + padl + rowsize, dststride, color, padr, srcheight, stream->stream));
            // vs_memset8(dstdata, color, padl);
            // memcpy(dstdata + padl, srcdata, rowsize);
            // vs_memset8(dstdata + padl + rowsize, color, padr);
            break;
        // case 2:
        //     vs_memset16(dstdata, color, padl / 2);
        //     memcpy(dstdata + padl, srcdata, rowsize);
        //     vs_memset16(dstdata + padl + rowsize, color, padr / 2);
        //     break;
        // case 4:
        //     vs_memset32(dstdata, color, padl / 4);
        //     memcpy(dstdata + padl, srcdata, rowsize);
        //     vs_memset32(dstdata + padl + rowsize, color, padr / 4);
        //     break;
        }
        dstdata += srcheight * dststride;

        // Pad BOTTOM
        switch (d->vi->format->bytesPerSample) {
        case 1:
            CHECKCUDA(cudaMemset2DAsync(dstdata, dststride, color, rowsize + padl + padr, padb, stream->stream));
            // vs_memset8(dstdata, color, padb * dststride);
            break;
        // case 2:
        //     vs_memset16(dstdata, color, padb * dststride / 2);
        //     break;
        // case 4:
        //     vs_memset32(dstdata, color, padb * dststride / 4);
        //     break;
        }
    }
}


//////////////////////////////////////////
// BlankClip

typedef struct {
    VSFrameRef *f;
    VSVideoInfo vi;
} BlankClipData;

union color{
    uint32_t i[3];
    float f[3];
};

template<typename T>
static __global__ void blankClipKernel(uint8_t *dstp, int stride, int width, int height, uint32_t color) {
    const int column = blockDim.x * blockIdx.x + threadIdx.x;
    const int row = blockDim.y * blockIdx.y + threadIdx.y;

    stride >>= 2;

    if (column >= width || row >= height)
        return;

    uint32_t dst_data = 0;

    for (int i = 0; i < (sizeof(uint32_t) / sizeof(T)); i++) {
       ((T *)&dst_data)[i] = (T)color;
    }

    ((uint32_t *)dstp)[stride * row + column] = dst_data;
}

VS_EXTERN_C void VS_CC blankClipProcessCUDA(void *color, const BlankClipData *d, VSCore *core, const VSAPI *vsapi) {
    int blockSize = VSCUDAGetBasicBlocksize();
    dim3 threads(blockSize, blockSize);

    for (int plane = 0; plane < d->vi.format->numPlanes; plane++) {
        uint8_t *dst = vsapi->getWritePtr(d->f, plane);
        int stride = vsapi->getStride(d->f, plane);
        uint32_t c = ((union color *)color)->i[plane];
        int width = vsapi->getFrameWidth(d->f, plane);
        int height = vsapi->getFrameHeight(d->f, plane);
        const VSCUDAStream *stream = vsapi->getStream(d->f, plane);
        dim3 grid(ceil((float)width / (threads.x * (sizeof(uint32_t) / d->vi.format->bytesPerSample))), ceil((float)height / threads.y));
        switch (d->vi.format->bytesPerSample) {
        case 1:
            blankClipKernel<uint8_t><<<grid, threads, 0, stream->stream>>>(dst, stride, width, height, c);
            break;
        case 2:
            blankClipKernel<uint16_t><<<grid, threads, 0, stream->stream>>>(dst, stride, width, height, c);
            break;
        case 4:
            blankClipKernel<uint32_t><<<grid, threads, 0, stream->stream>>>(dst, stride, width, height, c);
            break;
        }
    }
}


///////////////////////
// Lut

typedef struct {
    VSNodeRef *node;
    const VSVideoInfo *vi;
    void *lut;
    int process[3];
} LutData;

//Each thread perfoms a LUT on a block of 32 / 8 = 4 pixels.
template<typename T>
static __global__ void lutKernel(const uint8_t * __restrict__ srcp, uint8_t * __restrict__ dstp,
                                  int stride, const int width, const int height,
                                  const T * __restrict__ lut){
    const int column = blockDim.x * blockIdx.x + threadIdx.x;
    const int row = blockDim.y * blockIdx.y + threadIdx.y;

    stride >>= 2;

    if (column >= width || row >= height)
        return;

    uint32_t src_data = ((uint32_t *)srcp)[stride * row + column];
    uint32_t dst_data = 0;

    for (int i = 0; i < (sizeof(uint32_t) / sizeof(T)); i++) {
       ((T *)&dst_data)[i] = lut[((T *)&src_data)[i]];
    }

    ((uint32_t *)dstp)[stride * row + column] = dst_data;
}

VS_EXTERN_C void VS_CC lutProcessCUDA(const VSFrameRef *src, VSFrameRef *dst, const LutData *d,
                                    VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) {
    int blockSize = VSCUDAGetBasicBlocksize();
    dim3 threads(blockSize, blockSize);

    uint8_t *d_lut;
    int numElements = (1 << d->vi->format->bitsPerSample);
    CHECKCUDA(cudaMalloc(&d_lut, numElements * d->vi->format->bytesPerSample));
    CHECKCUDA(cudaMemcpy(d_lut, d->lut, numElements * d->vi->format->bytesPerSample, cudaMemcpyHostToDevice));

    for (int plane = 0; plane < d->vi->format->numPlanes; plane++) {
        const uint8_t *srcp = vsapi->getReadPtr(src, plane);
        uint8_t *dstp = vsapi->getWritePtr(dst, plane);
        int dst_stride = vsapi->getStride(dst, plane);
        int h = vsapi->getFrameHeight(src, plane);

        if (d->process[plane]) {
            int w = vsapi->getFrameWidth(src, plane);
            const VSCUDAStream *stream = vsapi->getStream(dst, plane);
            dim3 grid(ceil((float)w / (threads.x * (sizeof(uint32_t) / d->vi->format->bytesPerSample))), ceil((float)h / threads.y));

            if (d->vi->format->bytesPerSample == 1) {
                lutKernel<uint8_t><<<grid, threads, 0, stream->stream>>>(srcp, dstp, dst_stride, w, h, d_lut);
            } else {
                lutKernel<uint16_t><<<grid, threads, 0, stream->stream>>>(srcp, dstp, dst_stride, w, h, (uint16_t *)d_lut);
            }
        }
    }

    CHECKCUDA(cudaFree(d_lut));
}

//////////////////////////////////////////
// Transpose

typedef struct {
    VSNodeRef *node;
    VSVideoInfo vi;
} TransposeData;

#define TILE_DIM 128

static __global__ void alignedTransposeKernel(const uint8_t * __restrict__ src, uint8_t * __restrict__ dst, int src_stride, int dst_stride, int width, int height){
    __shared__ uint8_t tile[TILE_DIM][TILE_DIM + 1];

    //Adjust strides to account for using 4-byte data to work on 1-byte data.
    src_stride >>= 2;
    dst_stride >>= 2;

    int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
    int index_in = xIndex + (yIndex) * src_stride;

    for (int j = 0; j < TILE_DIM; j += blockDim.y) {
        if (yIndex + j < height) {
            uint32_t input = ((uint32_t *)src)[index_in + (j * src_stride)];

            #pragma unroll
            for (int i = 0; i < 4; i++) {
                tile[threadIdx.y + j][(threadIdx.x * 4) + i] = ((uint8_t *)&input)[i];
            }
        }
    }

    __syncthreads();

    xIndex = blockIdx.y * blockDim.x + threadIdx.x;
    yIndex = blockIdx.x * TILE_DIM + threadIdx.y;
    int index_out = xIndex + (yIndex) * dst_stride;

    for (int j = 0; j < TILE_DIM; j += blockDim.y) {
        if (yIndex + j < width) {
            int output = 0;

            #pragma unroll
            for (int i = 0; i < 4; i++) {
                ((uint8_t *)&output)[i] = tile[(threadIdx.x * 4) + i][threadIdx.y + j];
            }
            ((uint32_t *)dst)[index_out + (j * dst_stride)] = output;
        }
    }
}

VS_EXTERN_C void VS_CC transposeProcessCUDA(const VSFrameRef *src, VSFrameRef *dst, const TransposeData *d,
                                           VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) {
    int blockSize = VSCUDAGetBasicBlocksize();
    dim3 threads(blockSize, blockSize);

    for (int plane = 0; plane < d->vi.format->numPlanes; plane++) {
        int width = vsapi->getFrameWidth(src, plane);
        int height = vsapi->getFrameHeight(src, plane);
        const uint8_t *srcp = vsapi->getReadPtr(src, plane);
        int src_stride = vsapi->getStride(src, plane);
        uint8_t *dstp = vsapi->getWritePtr(dst, plane);
        int dst_stride = vsapi->getStride(dst, plane);
        const VSCUDAStream *stream = vsapi->getStream(dst, plane);

        switch (d->vi.format->bytesPerSample) {
            case 1:
                dim3 grid(ceil((float)width / TILE_DIM), ceil((float)height / TILE_DIM));
                alignedTransposeKernel<<<grid, threads, 0, stream->stream>>>(srcp, dstp, src_stride, dst_stride, width, height);
                // for (y = 0; y < height; y++)
                //     for (x = 0; x < width; x++)
                //         dstp[dst_stride * x + y] = srcp[src_stride * y + x];
                break;
            // case 2:
            //     src_stride /= 2;
            //     dst_stride /= 2;
            //     for (y = 0; y < height; y++)
            //         for (x = 0; x < width; x++)
            //             ((uint16_t *)dstp)[dst_stride * x + y] = ((const uint16_t *)srcp)[src_stride * y + x];
            //     break;
            // case 4:
            //     src_stride /= 4;
            //     dst_stride /= 4;
            //     for (y = 0; y < height; y++)
            //         for (x = 0; x < width; x++)
            //             ((uint32_t *)dstp)[dst_stride * x + y] = ((const uint32_t *)srcp)[src_stride * y + x];
            //     break;
        }
    }
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
static __global__ void mergeKernel(const uint8_t * __restrict__ srcp1, const uint8_t * __restrict__ srcp2, uint8_t * __restrict__ dstp, int stride, const int width, const int height, const int weight, const int round, const int MergeShift){
    const int column = blockDim.x * blockIdx.x + threadIdx.x;
    const int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (column >= width || row >= height)
        return;

    stride >>= 2;

    uint32_t src1_data = ((uint32_t *)srcp1)[stride * row + column];
    uint32_t src2_data = ((uint32_t *)srcp2)[stride * row + column];
    uint32_t dst_data = 0;

    for (int i = 0; i < sizeof(uint32_t); i++) {
       ((uint8_t *)&dst_data)[i] = ((uint8_t *)&src1_data)[i] + (((((uint8_t *)&src2_data)[i] - ((uint8_t *)&src1_data)[i]) * weight + round) >> MergeShift);
    }

    ((uint32_t *)dstp)[stride * row + column] = dst_data;
}

VS_EXTERN_C void VS_CC mergeProcessCUDA(const VSFrameRef *src1, const VSFrameRef *src2, VSFrameRef *dst, const MergeData *d, const int MergeShift, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) {
    int blockSize = VSCUDAGetBasicBlocksize();
    dim3 threads(blockSize, blockSize);

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
            const VSCUDAStream *stream = vsapi->getStream(dst, plane);

            if (d->vi->format->sampleType == stInteger) {
                const int round = 1 << (MergeShift - 1);
                if (d->vi->format->bytesPerSample == 1) {
                    dim3 grid(ceil((float)width / (threads.x * sizeof(uint32_t))), ceil((float)height / threads.y));

                    mergeKernel<<<grid, threads, 0, stream->stream>>>(srcp1, srcp2, dstp, stride, width / sizeof(uint32_t), height, weight, round, MergeShift);
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
static __global__ void maskedMergeKernel(const uint8_t * __restrict__ srcp1, const uint8_t * __restrict__ srcp2, uint8_t * __restrict__ dstp, int stride, const int width, const int height, const uint8_t * __restrict__ maskp){
    const int column = blockDim.x * blockIdx.x + threadIdx.x;
    const int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (column >= width || row >= height)
        return;

    stride >>= 2;

    const uint32_t src1_data = ((uint32_t *)srcp1)[stride * row + column];
    const uint32_t src2_data = ((uint32_t *)srcp2)[stride * row + column];
    const uint32_t mask_data = ((uint32_t *)maskp)[stride * row + column];
    uint32_t dst_data = 0;

    for (int i = 0; i < sizeof(uint32_t); i++) {
       ((uint8_t *)&dst_data)[i] = ((uint8_t *)&src1_data)[i] + (((((uint8_t *)&src2_data)[i] - ((uint8_t *)&src1_data)[i]) * (((uint8_t *)&mask_data)[i] > 2 ? ((uint8_t *)&mask_data)[i] + 1 : ((uint8_t *)&mask_data)[i]) + 128) >> 8);
    }

    ((uint32_t *)dstp)[stride * row + column] = dst_data;
}

VS_EXTERN_C void VS_CC maskedMergeProcessCUDA(const VSFrameRef *src1, const VSFrameRef *src2, VSFrameRef *dst, const VSFrameRef *mask, const VSFrameRef *mask23, const MaskedMergeData *d, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) {
    int blockSize = VSCUDAGetBasicBlocksize();
    dim3 threads(blockSize, blockSize);

    for (int plane = 0; plane < d->vi->format->numPlanes; plane++) {
        if (d->process[plane]) {
            int height = vsapi->getFrameHeight(src1, plane);
            int width = vsapi->getFrameWidth(src2, plane);
            int stride = vsapi->getStride(src1, plane);
            const uint8_t *srcp1 = vsapi->getReadPtr(src1, plane);
            const uint8_t *srcp2 = vsapi->getReadPtr(src2, plane);
            const uint8_t *maskp = vsapi->getReadPtr((plane && mask23) ? mask23 : mask, d->first_plane ? 0 : plane);
            uint8_t *dstp = vsapi->getWritePtr(dst, plane);
            const VSCUDAStream *stream = vsapi->getStream(dst, plane);

            if (d->vi->format->sampleType == stInteger) {
                if (d->vi->format->bytesPerSample == 1) {
                    dim3 grid(ceil((float)width / (threads.x * sizeof(uint32_t))), ceil((float)height / threads.y));

                    maskedMergeKernel<<<grid, threads, 0, stream->stream>>>(srcp1, srcp2, dstp, stride, width / sizeof(uint32_t), height, maskp);
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
}
