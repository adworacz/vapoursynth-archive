/*
* Copyright (c) 2012 Austin Dworaczyk Wiltshire
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

// Contains the CUDA implementation of the Expr filter.
#include <vector>
#include <stdexcept>
#include "VapourSynth.h"
#include "VSCuda.h"

typedef enum {
    opLoadSrc8, opLoadSrc16, opLoadSrcF, opLoadConst,
    opStore8, opStore16, opStoreF,
    opDup, opSwap,
    opAdd, opSub, opMul, opDiv, opMax, opMin, opSqrt, opAbs,
    opGt, opLt, opEq, opLE, opGE, opTernary,
    opAnd, opOr, opXor, opNeg,
    opExp, opLog, opPow
} SOperation;

typedef union {
    float fval;
    int32_t ival;
} ExprUnion;

struct ExprOp {
    ExprUnion e;
    uint32_t op;
    ExprOp() {
    }
    ExprOp(SOperation op, float val) : op(op) {
        e.fval = val;
    }
    ExprOp(SOperation op, int32_t val = 0) : op(op) {
        e.ival = val;
    }
};

enum PlaneOp {
    poProcess, poCopy, poUndefined
};

typedef struct {
    VSNodeRef *node[3];
    VSVideoInfo vi;
    ExprOp *gpu_ops[3];
    std::vector<ExprOp> ops[3];
    int plane[3];
#ifdef VS_X86
    void *stack;
#else
    std::vector<float> stack;
#endif
} JitExprData;

//Since we don't support the allocation of variable size arrays in
//each thread's local memory, we will just have to work with a large
//buffer unfortunately. This will really effect performance unfortunately.
#define MAX_STACK_SIZE 10

template <typename T>
__device__ float performOp(float * __restrict__ stack, uint32_t * __restrict__ input, const int index, const ExprOp * __restrict__ d_vops);

static __global__ void exprKernel(uint8_t * __restrict__ dstp, int stride, const uint8_t * __restrict__ srcp0, const uint8_t * __restrict__ srcp1, const uint8_t * __restrict__ srcp2,
                                const int width, const int height, const ExprOp * __restrict__ d_vops) {
    const int column = blockDim.x * blockIdx.x + threadIdx.x;
    const int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (column >= width || row >= height)
        return;

    float stack[MAX_STACK_SIZE];

    //uint8_t case.
    uint32_t input[3];
    uint32_t output;

    if (srcp0 != NULL)
        input[0] = ((uint32_t *)srcp0)[(stride >> 2) * row + column];
    if (srcp1 != NULL)
        input[1] = ((uint32_t *)srcp1)[(stride >> 2) * row + column];
    if (srcp2 != NULL)
        input[2] = ((uint32_t *)srcp2)[(stride >> 2) * row + column];

    for (int i = 0; i < 4; i++) {
        ((uint8_t *)&output)[i] = performOp<uint8_t>(stack, input, i, d_vops);
    }

    ((uint32_t *)dstp)[(stride >> 2) * row + column] = output;
}

template <typename T>
__device__ float performOp(float * __restrict__ stack, uint32_t * __restrict__ input, const int index, const ExprOp * __restrict__ d_vops) {
    float stacktop = 0;
    float tmp;

    int si = 0;
    int i = -1;
    while (true) {
        i++;
        switch (d_vops[i].op) {
        case opLoadSrc8:
        case opLoadSrc16:
        case opLoadSrcF:
            stack[si] = stacktop;
            stacktop = ((T *)&input[d_vops[i].e.ival])[index];
            ++si;
            break;
        case opLoadConst:
            stack[si] = stacktop;
            stacktop = d_vops[i].e.fval;
            ++si;
            break;
        case opDup:
            stack[si] = stacktop;
            ++si;
            break;
        case opSwap:
            tmp = stacktop;
            stacktop = stack[si];
            stack[si] = tmp;
            break;
        case opAdd:
            --si;
            stacktop += stack[si];
            break;
        case opSub:
            --si;
            stacktop = stack[si] - stacktop;
            break;
        case opMul:
            --si;
            stacktop *= stack[si];
            break;
        case opDiv:
            --si;
            stacktop = stack[si] / stacktop;
            break;
        case opMax:
            --si;
            stacktop = fmaxf(stacktop, stack[si]);
            break;
        case opMin:
            --si;
            stacktop = fminf(stacktop, stack[si]);
            break;
        case opExp:
            stacktop = expf(stacktop);
            break;
        case opLog:
            stacktop = logf(stacktop);
            break;
        case opPow:
            --si;
            stacktop = powf(stack[si], stacktop);
            break;
        case opSqrt:
            stacktop = sqrtf(stacktop);
            break;
        case opAbs:
            stacktop = fabsf(stacktop);
            break;
        case opGt:
            --si;
            stacktop = (stack[si] > stacktop) ? 1.0f : 0.0f;
            break;
        case opLt:
            --si;
            stacktop = (stack[si] < stacktop) ? 1.0f : 0.0f;
            break;
        case opEq:
            --si;
            stacktop = (stack[si] == stacktop) ? 1.0f : 0.0f;
            break;
        case opLE:
            --si;
            stacktop = (stack[si] <= stacktop) ? 1.0f : 0.0f;
            break;
        case opGE:
            --si;
            stacktop = (stack[si] >= stacktop) ? 1.0f : 0.0f;
            break;
        case opTernary:
            si -= 2;
            stacktop = (stack[si] > 0) ? stack[si + 1] : stacktop;
            break;
        case opAnd:
            --si;
            stacktop = (stacktop > 0 && stack[si] > 0) ? 1.0f : 0.0f;
            break;
        case opOr:
            --si;
            stacktop = (stacktop > 0 || stack[si] > 0) ? 1.0f : 0.0f;
            break;
        case opXor:
            --si;
            stacktop = ((stacktop > 0) != (stack[si] > 0)) ? 1.0f : 0.0f;
            break;
        case opNeg:
            stacktop = (stacktop > 0) ? 0.0f : 1.0f;
            break;
        case opStore8:
            return fmaxf(0.0f, fminf(stacktop, 255.0f)) + 0.5f;
        case opStore16:
            return fmaxf(0.0f, fminf(stacktop, 256*255.0f)) + 0.5f;
        case opStoreF:
            return stacktop;
        }
    }
}

ExprOp * VS_CC copyExprOps(const ExprOp *vops, int numOps) {
    ExprOp *gpu_ops;
    CHECKCUDA(cudaMalloc(&gpu_ops, numOps * sizeof(ExprOp)));
    CHECKCUDA(cudaMemcpy(gpu_ops, vops, numOps * sizeof(ExprOp), cudaMemcpyHostToDevice));

    return gpu_ops;
}

void VS_CC freeExprOps(ExprOp *gpu_ops) {
    CHECKCUDA(cudaFree(gpu_ops));
}

void VS_CC exprProcessCUDA(const VSFrameRef **src, VSFrameRef *dst, const JitExprData *d,
                                           VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) {
    int blockSize = VSCUDAGetBasicBlocksize();
    dim3 threads(blockSize, blockSize);

    //Change the preferred cache config. Shows significant speedup in our case.
    CHECKCUDA(cudaFuncSetCacheConfig(exprKernel, cudaFuncCachePreferL1));

    const uint8_t *srcp[3];

    for (int plane = 0; plane < d->vi.format->numPlanes; plane++) {
        if (d->plane[plane] == poProcess) {
            for (int i = 0; i < 3; i++) {
                if (d->node[i]) {
                    srcp[i] = vsapi->getReadPtr(src[i], plane);
                } else {
                    srcp[i] = NULL;
                }
            }

            uint8_t *dstp = vsapi->getWritePtr(dst, plane);
            int dst_stride = vsapi->getStride(dst, plane);
            int height = vsapi->getFrameHeight(src[0], plane);
            int width = vsapi->getFrameWidth(src[0], plane);
            const VSCUDAStream *stream = vsapi->getStream(dst, plane);

            dim3 grid(ceil((float)width / (threads.x * sizeof(uint32_t))), ceil((float)height / threads.y));
            exprKernel<<<grid, threads, 0, stream->stream>>>(dstp, dst_stride, srcp[0], srcp[1], srcp[2], width, height, d->gpu_ops[plane]);
        }
    }
}
