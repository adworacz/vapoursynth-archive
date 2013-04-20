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

// Contains the CUDA implementation of the Expr filter.
#include <vector>
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
    ExprOp *d_ops[3];
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
#define MAX_EXPR_OPS 64

__constant__ uint8_t *d_srcp[3];
__constant__ int d_src_stride[3];

static __global__ void exprKernel(uint8_t * __restrict__ dstp, int dst_stride, const int width,
                                  const int height, const ExprOp * __restrict__ vops) {
    const int column = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
    const int row = IMAD(blockDim.y, blockIdx.y, threadIdx.y);

    if (column >= width || row >= height)
        return;

    float stack[MAX_EXPR_OPS];
    float stacktop = 0;
    float tmp;

    int si = 0;
    int i = -1;
    while (true) {
        i++;
        switch (vops[i].op) {
        case opLoadSrc8:
            stack[si] = stacktop;
            stacktop = d_srcp[vops[i].e.ival][d_src_stride[vops[i].e.ival] * row + column];
            ++si;
            break;
        case opLoadSrc16:
            stack[si] = stacktop;
            stacktop = ((const uint16_t *)d_srcp[vops[i].e.ival])[(d_src_stride[vops[i].e.ival] >> 1) * row + column];
            ++si;
            break;
        case opLoadSrcF:
            stack[si] = stacktop;
            stacktop = ((const float *)d_srcp[vops[i].e.ival])[(d_src_stride[vops[i].e.ival] >> 2) * row + column];
            ++si;
            break;
        case opLoadConst:
            stack[si] = stacktop;
            stacktop = vops[i].e.fval;
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
            dstp[dst_stride * row + column] = fmaxf(0.0f, fminf(stacktop, 255.0f)) + 0.5f;
            goto loopend;
        case opStore16:
            ((uint16_t *)dstp)[(dst_stride >> 1) * row + column] = fmaxf(0.0f, fminf(stacktop, 256*255.0f)) + 0.5f;
            goto loopend;
        case opStoreF:
            ((float *)dstp)[(dst_stride >> 2) * row + column] = stacktop;
            goto loopend;
        }
    }
    loopend:;
}

ExprOp * VS_CC copyExprOps(const ExprOp *vops, int numOps) {
    ExprOp *d_ops;
    CHECKCUDA(cudaMalloc(&d_ops, numOps * sizeof(ExprOp)));
    CHECKCUDA(cudaMemcpy(d_ops, vops, numOps * sizeof(ExprOp), cudaMemcpyHostToDevice));

    return d_ops;
}

void VS_CC freeExprOps(ExprOp *d_ops) {
    CHECKCUDA(cudaFree(d_ops));
}

int VS_CC exprProcessCUDA(const VSFrameRef **src, VSFrameRef *dst, const JitExprData *d,
                                           VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) {
    int blockSize = VSCUDAGetBasicBlocksize();
    dim3 threads(blockSize, blockSize);

    cudaStream_t stream = vsapi->getStreamForFrame(src[0], frameCtx, core);

    if (stream == 0) {
        return 0;
    }

    const uint8_t *srcp[3];
    int src_stride[3];

    for (int plane = 0; plane < d->vi.format->numPlanes; plane++) {
        if (d->plane[plane] == poProcess) {
            for (int i = 0; i < 3; i++) {
                if (d->node[i]) {
                    srcp[i] = vsapi->getReadPtr(src[i], plane);
                    src_stride[i] = vsapi->getStride(src[i], plane);
                } else {
                    srcp[i] = NULL;
                    src_stride[i] = 0;
                }
            }

            uint8_t *dstp = vsapi->getWritePtr(dst, plane);
            int dst_stride = vsapi->getStride(dst, plane);
            int height = vsapi->getFrameHeight(src[0], plane);
            int width = vsapi->getFrameWidth(src[0], plane);

            CHECKCUDA(cudaMemcpyToSymbol(d_srcp, srcp, 3 * sizeof(uint8_t *)));
            CHECKCUDA(cudaMemcpyToSymbol(d_src_stride, src_stride, 3 * sizeof(int)));

            dim3 grid(ceil((float)width / threads.x), ceil((float)height / threads.y));
            exprKernel<<<grid, threads, 0, stream>>>(dstp, dst_stride, width, height, d->d_ops[plane]);
        }
    }

    return 1;
}
