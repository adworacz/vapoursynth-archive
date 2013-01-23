// A CUDA specific file for all CUDA-aimed functions of the VSCore.

#include "vscore.h"
#include "VSCuda.h"
#include "VSHelper.h"

//Note: the FrameLocation may be unnecessary, especially if only VSFrame calls VSFrameData.
VSFrameData::VSFrameData(int width, int height, int *stride, int bytesPerSample, MemoryUse * mem,
                         FrameLocation fLocation) : frameLocation(fLocation) {
    cudaPitchedPtr d_ptr;

    CHECKCUDA(cudaMalloc3D(&d_ptr, make_cudaExtent(width * bytesPerSample, height, 1)));
    data = (uint8_t *) d_ptr.ptr;
    *stride = d_ptr.pitch;
    mem->add(*stride * height);
}

VSFrameData::~VSFrameData() {
    if (frameLocation == flLocal)
        vs_aligned_free(data);
    else
        CHECKCUDA(cudaFree(data));

    mem->subtract(size);
}

VSFrame::VSFrame(const VSFormat * f, int width, int height, const VSFrame * propSrc, VSCore * core,
                 FrameLocation fLocation) : format(f), width(width), height(height),
                 frameLocation(fLocation) {
    if (fLocation == flLocal) {
        VSFrame(f, width, height, propSrc, core);
    } else {
        if (!f || width <= 0 || height <= 0)
            qFatal("Invalid new frame");

        if (propSrc)
            properties = propSrc->properties;

        for (int plane = 0; plane < f->numPlanes; plane++) {
            int compensatedWidth  = (plane ? width  >> f->subSamplingW : width);
            int compensatedHeight = (plane ? height >> f->subSamplingH : height);

            data[plane] =
                new VSFrameData(compensatedWidth, compensatedHeight, &stride[plane], f->bytesPerSample,
                                core->gpuMemory, fLocation);
        }
    }
}
