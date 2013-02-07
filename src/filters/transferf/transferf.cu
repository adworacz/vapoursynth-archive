//////////////////////////////////////////
// This file contains a simple filter
// skeleton you can use to get started.
// With no changes it simply passes
// frames through.

#include "VapourSynth.h"
#include "VSHelper.h"
#include "VSCuda.h"

typedef struct {
    VSNodeRef *node;
    const VSVideoInfo *vi;
    int direction; // 0 = to Host, 1 = to GPU.
} TransferFData;

static void VS_CC transferFInit(VSMap *in, VSMap *out, void **instanceData, VSNode *node, VSCore *core, const VSAPI *vsapi) {
    TransferFData *d = (TransferFData *) * instanceData;
    vsapi->setVideoInfo(d->vi, 1, node);
}

static const VSFrameRef *VS_CC transferFGetFrame(int n, int activationReason, void **instanceData, void **frameData, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) {
    TransferFData *d = (TransferFData *) * instanceData;

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->node, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        const VSFrameRef *src = vsapi->getFrameFilter(n, d->node, frameCtx);
        const VSFormat *fi = d->vi->format;
        int height = vsapi->getFrameHeight(src, 0);
        int width = vsapi->getFrameWidth(src, 0);

        if (d->direction == 0) {
            //Create a new CPU/Host frame.
            if (vsapi->getFrameLocation(src) == flLocal) {
                vsapi->setFilterError("TransferFrame: Attempted to transfer a CPU frame to a CPU frame. Check your direction.", frameCtx);
                vsapi->freeNode(d->node);
                return 0;
            }

            VSFrameRef *src_cpu = vsapi->newVideoFrame(fi, width, height, src, core);
            vsapi->transferVideoFrame(src, src_cpu, ftdGPUtoCPU, core);
            vsapi->freeFrame(src);

            return src_cpu;

        } else {
            //Create a new GPU/Device frame.
            if (vsapi->getFrameLocation(src) == flGPU) {
                vsapi->setFilterError("TransferFrame: Attempted to transfer a GPU frame to a GPU frame. Check your direction.", frameCtx);
                vsapi->freeNode(d->node);
                return 0;
            }

            VSFrameRef *src_gpu = vsapi->newVideoFrame3(fi, width, height, src, core, flGPU);
            vsapi->transferVideoFrame(src, src_gpu, ftdCPUtoGPU, core);
            vsapi->freeFrame(src);

            return src_gpu;
        }
    }

    return 0;
}

static void VS_CC transferFFree(void *instanceData, VSCore *core, const VSAPI *vsapi) {
    TransferFData *d = (TransferFData *)instanceData;
    vsapi->freeNode(d->node);
    free(d);
}

static void VS_CC transferFCreate(const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi) {
    TransferFData d;
    TransferFData *data;
    int err;

    d.node = vsapi->propGetNode(in, "clip", 0, 0);
    d.vi = vsapi->getVideoInfo(d.node);

    d.direction = vsapi->propGetInt(in, "direction", 0, &err);
    if(err || (d.direction < 0 || d.direction > 1)) {
        vsapi->setError(out, "TransferFrame: Direction must be specified and must be either 0 or 1.");
        vsapi->freeNode(d.node);
        return;
    }

    data = (TransferFData *)malloc(sizeof(d));
    *data = d;

    vsapi->createFilter(in, out, "Filter", transferFInit, transferFGetFrame, transferFFree, fmParallel, 0, data, core);
    return;
}

//////////////////////////////////////////
// Init

VS_EXTERNAL_API(void) VapourSynthPluginInit(VSConfigPlugin configFunc, VSRegisterFunction registerFunc, VSPlugin *plugin) {
    configFunc("com.cuda.transferf", "transferf", "A filter to transfer frames between gpu and host.", VAPOURSYNTH_API_VERSION, 1, plugin);
    registerFunc("TransferFrame", "clip:clip;direction:int;", transferFCreate, 0, plugin);
}
