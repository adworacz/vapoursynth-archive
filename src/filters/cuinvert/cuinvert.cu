//////////////////////////////////////////
// This file contains a simple invert
// filter that's commented to show
// the basics of the filter api.
// This file may make more sense when
// read from the bottom and up.

#include <stdlib.h>
#include "VapourSynth.h"
#include "VSHelper.h"
#include "VSCuda.h"

typedef struct {
    VSNodeRef *node;
    const VSVideoInfo *vi;
    int enabled;
} InvertData;

// This function is called immediately after vsapi->createFilter(). This is the only place where the video
// properties may be set. In this case we simply use the same as the input clip. You may pass an array
// of VSVideoInfo if the filter has more than one output, like rgb+alpha as two separate clips.
static void VS_CC invertInit(VSMap *in, VSMap *out, void **instanceData, VSNode *node, VSCore *core, const VSAPI *vsapi) {
    InvertData *d = (InvertData *) * instanceData;
    vsapi->setVideoInfo(d->vi, 1, node);
}

// This is the main function that gets called when a frame should be produced. It will, in most cases, get
// called several times to produce one frame. This state is being kept track of by the value of
// activationReason. The first call to produce a certain frame n is always arInitial. In this state
// you should request all the input frames you need. Always do it in ascending order to play nice with the
// upstream filters.
// Once all frames are ready, the the filter will be called with arAllFramesReady. It is now time to
// do the actual processing.
static const VSFrameRef *VS_CC invertGetFrame(int n, int activationReason, void **instanceData, void **frameData, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) {
    InvertData *d = (InvertData *) * instanceData;

    if (activationReason == arInitial) {
        // Request the source frame on the first call
        vsapi->requestFrameFilter(n, d->node, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        const VSFrameRef *src = vsapi->getFrameFilter(n, d->node, frameCtx);
        // The reason we query this on a per frame basis is because we want our filter
        // to accept clips with varying dimensions. If we reject such content using d->vi
        // would be better.
        const VSFormat *fi = d->vi->format;
        int height = vsapi->getFrameHeight(src, 0);
        int width = vsapi->getFrameWidth(src, 0);


        // When creating a new frame for output it is VERY EXTREMELY SUPER IMPORTANT to
        // supply the "dominant" source frame to copy properties from. Frame props
        // are an essential part of the filter chain and you should NEVER break it.
        VSFrameRef *dst = vsapi->newVideoFrame(fi, width, height, src, core);

        // It's processing loop time!
        // Loop over all the planes
        int plane;
        for (plane = 0; plane < fi->numPlanes; plane++) {
            const uint8_t *srcp = vsapi->getReadPtr(src, plane);
            int src_stride = vsapi->getStride(src, plane);
            uint8_t *dstp = vsapi->getWritePtr(dst, plane);
            int dst_stride = vsapi->getStride(dst, plane); // note that if a frame has the same dimensions and format, the stride is guaranteed to be the same. int dst_stride = src_stride would be fine too in this filter.
            // Since planes may be subsampled you have to query the height of them individually
            int h = vsapi->getFrameHeight(src, plane);
            int y;
            int w = vsapi->getFrameWidth(src, plane);
            int x;

            for (y = 0; y < h; y++) {
                for (x = 0; x < w; x++)
                    dstp[x] = ~srcp[x];

                dstp += dst_stride;
                srcp += src_stride;
            }
        }

        // Release the source frame
        vsapi->freeFrame(src);

        // A reference is consumed when it is returned, so saving the dst reference somewhere
        // and reusing it is not allowed.
        return dst;
    }

    return 0;
}

static __global__ void invertKernel(uint8_t *d_srcdata, uint8_t *d_dstdata, int width, int height) {
    const int ix = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
    const int iy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);

    if (ix > width || iy > height)
        return;

    d_dstdata[width * iy + ix] = ~d_srcdata[width * iy + ix];
}

static void invertWithCuda(VSFrameRef *src, VSFrameRef *dst, const VSFormat *fi, const VSAPI *vsapi){
    cudaPitchedPtr d_srcp;
    cudaPitchedPtr d_dstp;


    int deviceID = 0;
    cudaDeviceProp deviceProp;
    CHECKCUDA(cudaGetDeviceProperties(&deviceProp, deviceID));

    //CUDA Compute Capability < 2.0 only supports a maximum of 512 threads,
    //while CUDA Compute Capability >= 2.0 supports 1024 threads.
    int blockSize = (deviceProp.major < 2) ? 16 : 32;

    int plane;
    for (plane = 0; plane < fi->numPlanes; plane++) {
        int h = vsapi->getFrameHeight(src, plane);
        int w = vsapi->getFrameWidth(src, plane);

        const uint8_t *srcp = vsapi->getReadPtr(src, plane);
        uint8_t *dstp = vsapi->getWritePtr(dst, plane);
        int src_stride = vsapi->getStride(src, plane);
        int dst_stride = vsapi->getStride(dst, plane);

        if (plane > 0)
            h = h >> fi->subSamplingH; //Account for subsampling in non-luma planes

        //Allocate GPU memory for src frame and dst frame, then ship over src data.
        CHECKCUDA(cudaMalloc3D(&d_srcp, make_cudaExtent(w * fi->bytesPerSample, h, 1)));
        CHECKCUDA(cudaMalloc3D(&d_dstp, make_cudaExtent(w * fi->bytesPerSample, h, 1)));

        CHECKCUDA(cudaMemcpy2D(d_srcp.ptr, d_srcp.pitch, srcp, src_stride, w * fi->bytesPerSample, h, cudaMemcpyHostToDevice));

        //Do processing.
        dim3 threads(blockSize, blockSize);
        dim3 grid(ceil(w / threads.x), ceil(h / threads.y));

        invertKernel<<<grid, threads>>>((uint8_t *)d_srcp.ptr, (uint8_t *)d_dstp.ptr, w, h);

        //Free up GPU memory.
        CHECKCUDA(cudaFree(d_srcp.ptr));
        CHECKCUDA(cudaFree(d_dstp.ptr));
    }
}

// Free all allocated data on filter destruction
static void VS_CC invertFree(void *instanceData, VSCore *core, const VSAPI *vsapi) {
    InvertData *d = (InvertData *)instanceData;
    vsapi->freeNode(d->node);
    free(d);
}

// This function is responsible for validating arguments and creating a new filter
static void VS_CC invertCreate(const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi) {
    InvertData d;
    InvertData *data;
    VSNodeRef *cref;
    int err;

    // Get a clip reference from the input arguments. This must be freed later.
    d.node = vsapi->propGetNode(in, "clip", 0, 0);
    d.vi = vsapi->getVideoInfo(d.node);

    // In this first version we only want to handle 8bit integer formats. Note that
    // vi->format can be 0 if the input clip can change format midstream.
    if (!isConstantFormat(d.vi) || d.vi->format->sampleType != stInteger || d.vi->format->bitsPerSample != 8) {
        vsapi->setError(out, "cuInvert: only constant format 8bit integer input supported");
        vsapi->freeNode(d.node);
        return;
    }

    // If a property read fails for some reason (index out of bounds/wrong type)
    // then err will have flags set to indicate why and 0 will be returned. This
    // can be very useful to know when having optional arguments. Since we have
    // strict checking because of what we wrote in the argument string, the only
    // reason this could fail is when the value wasn't set by the user.
    // And when it's not set we want it to default to enabled.
    d.enabled = !!vsapi->propGetInt(in, "enable", 0, &err);
    if (err)
        d.enabled = 1;

    // Let's pretend the only allowed values are 1 or 0...
    if (d.enabled < 0 || d.enabled > 1) {
        vsapi->setError(out, "cuInvert: enabled must be 0 or 1");
        vsapi->freeNode(d.node);
        return;
    }

    // I usually keep the filter data struct on the stack and don't allocate it
    // until all the input validation is done.
    data = (InvertData *)malloc(sizeof(d));
    *data = d;

    // Creates a new filter and returns a reference to it. Always pass on the in and out
    // arguments or unexpected things may happen. The name should be something that's
    // easy to connect to the filter, like its function name.
    // The three function pointers handle initialization, frame processing and filter destruction.
    // The filtermode is very important to get right as it controls how threading of the filter
    // is handled. In general you should only use fmParallel whenever possible. This is if you
    // need to modify no shared data at all when the filter is running.
    // For more complicated filters, fmParallelRequests is usually easier to achieve as it can
    // be prefetched in parallel but the actual processing is serialized.
    // The others can be considered special cases where fmSerial is useful to source filters and
    // fmUnordered is useful when a filter's state may change even when deciding which frames to
    // prefetch (such as a cache filter).
    // If your filter is really fast (such as a filter that only resorts frames) you should set the
    // nfNoCache flag to make the caching work smoother.
    vsapi->createFilter(in, out, "cuInvert", invertInit, invertGetFrame, invertFree, fmParallel, 0, data, core);
    return;
}

//////////////////////////////////////////
// Init

// This is the entry point that is called when a plugin is loaded. You are only supposed
// to call the two provided functions here.
// configFunc sets the id, namespace, and long name of the plugin (the last 3 arguments
// never need to be changed for a normal plugin).
//
// id: Needs to be a "reverse" url and unique among all plugins.
//   It is inspired by how android packages identify themselves.
//   If you don't own a domain then make one up that's related
//   to the plugin name.
//
// namespace: Should only use [a-z_] and not be too long.
//
// full name: Any name that describes the plugin nicely.
//
// registerFunc is called once for each function you want to register. Function names
// should be PascalCase. The argument string has this format:
// name:type; or name:type:flag1:flag2....;
// All argument name should be lowercase and only use [a-z_].
// The valid types are int,float,data,clip,frame,func. [] can be appended to allow arrays
// of type to be passed (numbers:int[])
// The available flags are opt, to make an argument optional, empty, which controls whether
// or not empty arrays are accepted and link which will not be explained here.

VS_EXTERNAL_API(void) VapourSynthPluginInit(VSConfigPlugin configFunc, VSRegisterFunction registerFunc, VSPlugin *plugin) {
    configFunc("com.example.invert", "cuinvert", "VapourSynth Invert Example", VAPOURSYNTH_API_VERSION, 1, plugin);
    registerFunc("Filter", "clip:clip;enabled:int:opt;", invertCreate, 0, plugin);
}
