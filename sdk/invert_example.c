//////////////////////////////////////////
// This file contains a simple invert
// filter that's commented to show
// the basics of the filter api.
// This file may make more sense when
// read from the bottom and up.

#include <stdlib.h>
#include "VapourSynth.h"

typedef struct {
    const VSNodeRef *node;
    const VSVideoInfo *vi;
    int enabled;
} InvertData;

// This function is called immediated after vsapi->createFilter(). This is the only place where the video
// properties may be set. In this case we simply use the same as the input clip.
static void VS_CC invertInit(VSMap *in, VSMap *out, void **instanceData, VSNode *node, VSCore *core, const VSAPI *vsapi) {
    InvertData *d = (InvertData *) * instanceData;
    vsapi->setVideoInfo(d->vi, node);
}

// This is the main function that gets called when a frame should be produced. It will in most cases get
// called several times to produce one frame. This state is being kept track of by the value of
// activationReason. The first call to produce a certain frame n is always arInitial. In this state
// you should request all input frames you need. Always do it i ascending order to play nice with the
// upstream filters.
// Once all frames are ready the the filter will be called with arAllFramesReady. It is now time to
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
            int dst_stride = vsapi->getStride(dst, plane); // note that if a frame has the same dimensions and format the stride is guaranteed to be the same, int dst_stride = src_stride would be fin too in this filter
            // Since planes may be subsampled you have to query the height of them individually
            int h = vsapi->getFrameHeight(src, plane);
            int y;
            int w = vsapi->getFrameWidth(src, plane) - 1;
            int x;
    
            for (y = 0; y < h; y++) {
                for (x = 0; x <= w; x++)
                    dstp[x] = ~srcp[x];

                dstp += dst_stride;
                srcp += src_stride;
            }
        }

        // Release the source frame
        vsapi->freeFrame(src);

        // A reference is consumed when it is returned so saving the dst ref somewhere
        // and reusing it is not allowed.
        return dst;
    }

    return 0;
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
    const VSNodeRef *cref;
    int err;

    // Get a clip reference from the input arguments. This must be freed later.
    d.node = vsapi->propGetNode(in, "clip", 0, 0);
    d.vi = vsapi->getVideoInfo(d.node);

    // In this first version we only want to handle 8bit integer formats. Note that
    // vi->format can be 0 if the input clip can change format midstream.
    if (!d.vi->format || d.vi->format->sampleType != stInteger || d.vi->format->bitsPerSample != 8) {
        vsapi->setError(out, "Invert: only constant format 8bit integer input supported");
        vsapi->freeNode(d.node);
        return;
    }

    // If a property read fails for some reason (index out of bounds/wrong type)
    // then err will have flags set to indicate why and 0 will be returned. This
    // can be very useful to know when having optional arguments. Since we have
    // strict checking because of what we wrote in the argument string the only reason
    // this could fail is when the value wasn't set by the user.
    // And when it's not set we want it to default to enabled.
    d.enabled = vsapi->propGetInt(in, "enable", 0, &err);
    if (err)
        d.enabled = 1;

    // Let's pretend the only allowed values are 1 or 0...
    if (d.enabled < 0 || d.enabled > 1) {
        vsapi->setError(out, "Invert: enabled must be 0 or 1");
        vsapi->freeNode(d.node);
        return;
    }

    // I usually keep the filter data struct on the stack and don't allocate it
    // until all the input validation is done.
    data = malloc(sizeof(d));
    *data = d;

    // Create a new filter and returns a reference to it. Always pass on the in an out
    // arguments or unexpected thigns may happen. The name should be something that's
    // easy to connect to the filter, like its function name.
    // The three function pointers handle initialization, frame processing and filter destruction.
    // The filtermode is very important to get right as it controls how threading of the filter
    // is handled. In general you should only use fmParallel whenever possible. This is if you
    // need to modify no shared data at all when the filter is running.
    // For more complicated filters fmParallelRequests is usually easier to achieve as an
    // be prefetched in parallel but no
    // The others can be considered special cases where fmSerial is useful to source filters and
    // fmUnordered is useful when a filter's state may change even when deciding which frames to
    // prefetch (such as a cache filter).
    // If you filter is really fast (such as a filter that only resorts frames) you should set the
    // nfNoCache flag to make the caching work smoother.
    cref = vsapi->createFilter(in, out, "Invert", invertInit, invertGetFrame, invertFree, fmParallel, 0, data, core);
    vsapi->propSetNode(out, "clip", cref, 0);
    vsapi->freeNode(cref);
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
//   It is inspired by how android packages dientify themselves.
//   If you don't own a domain then make one up that's related
//   to the plugin name.
//
// namespace: Should only use [a-z_] and not be too long.
//
// full name: Any name that describes the plugin nicely.
//
// registerFunc is called once for each function you want to register. Function names
// should be CamelCase. The argument string has this format:
// name:type; or name:type:flag1:flag2....;
// All argument name should be lowercase and only use [a-z_].
// The valid types are int,float,data,clip,frame,func. [] can be appended to allow arrays
// of tyhe type to be passed (numbers:int[])
// The available flags are opt, to make an argument optional, and link which will not be
// explained here.

void VS_CC VapourSynthPluginInit(VSConfigPlugin configFunc, VSRegisterFunction registerFunc, VSPlugin *plugin) {
    configFunc("com.example.invert", "invert", "VapourSynth Invert Example", VAPOURSYNTH_API_VERSION, 1, plugin);
    registerFunc("Filter", "clip:clip;enabled:int:opt;", invertCreate, 0, plugin);
}
