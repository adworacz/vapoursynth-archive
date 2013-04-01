// A CUDA specific file for all CUDA-aimed functions of the VSCore.

#include "vscore.h"
#include "VSCuda.h"
#include "VSHelper.h"

//Note: FrameLocation is necessary in order to manage memory correctly in the VSFrameData destructor.
VSFrameData::VSFrameData(int width, int height, int *stride, int bytesPerSample, MemoryUse * mem,
                         FrameLocation fLocation) : mem(mem), frameLocation(fLocation) {
    cudaPitchedPtr d_ptr;

    if (fLocation != flGPU) {
        qFatal("Only GPU memory allocation is currently supported by this function. This needs to be fixed.");
    }

    CHECKCUDA(cudaMalloc3D(&d_ptr, make_cudaExtent(width * bytesPerSample, height, 1)));
    data = (uint8_t *) d_ptr.ptr;
    *stride = d_ptr.pitch;
    size = *stride * height;
    mem->add(size);
}

VSFrameData::VSFrameData(const VSFrameData &d) : QSharedData(d) {
    size = d.size;
    mem = d.mem;
    frameLocation = d.frameLocation;

    if (frameLocation == flLocal) {
        data = vs_aligned_malloc<uint8_t>(size, VSFrame::alignment);
        Q_CHECK_PTR(data);
        memcpy(data, d.data, size);
    } else {
        CHECKCUDA(cudaMalloc(&data, size));
        CHECKCUDA(cudaMemcpy(data, d.data, size, cudaMemcpyDeviceToDevice));
    }

    mem->add(size);
}

VSFrameData::~VSFrameData() {
    if (frameLocation == flLocal)
        vs_aligned_free(data);
    else
        CHECKCUDA(cudaFree(data));

    mem->subtract(size);
}

void VSFrameData::transferData(VSFrameData *dst, int dstStride,
                               int srcStride, int width, int height, int bytesPerSample,
                               FrameTransferDirection direction) const {
    cudaMemcpyKind transferKind = (direction == ftdCPUtoGPU ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost);

    CHECKCUDA(cudaMemcpy2D(dst->data, dstStride, data, srcStride, width * bytesPerSample, height, transferKind));
}

//Note: future integration can use default parameters to prevent code duplication.
VSFrame::VSFrame(const VSFormat * f, int width, int height, const VSFrame * propSrc, VSCore * core,
                 FrameLocation fLocation) : format(f), width(width), height(height),
                 frameLocation(fLocation) {
    if (!f || width <= 0 || height <= 0)
        qFatal("Invalid new frame");

    if (propSrc)
        properties = propSrc->properties;

    if (frameLocation != flLocal && frameLocation != flGPU)
        qFatal("Invalid frame location. Please use flLocal or flGPU. Specified: %d", frameLocation);

    if (frameLocation == flLocal) {
        //Handle CPU implementation.
        //This is a simple copy and paste of the vscore.cpp VSFrame constructor.
        stride[0] = (width * (f->bytesPerSample) + (alignment - 1)) & ~(alignment - 1);

        if (f->numPlanes == 3) {
            int plane23 = ((width >> f->subSamplingW) * (f->bytesPerSample) + (alignment - 1)) & ~(alignment - 1);
            stride[1] = plane23;
            stride[2] = plane23;
        } else {
            stride[1] = 0;
            stride[2] = 0;
        }

        data[0] = new VSFrameData(stride[0] * height, core->memory);
        if (f->numPlanes == 3) {
            int size23 = stride[1] * (height >> f->subSamplingH);
            data[1] = new VSFrameData(size23, core->memory);
            data[2] = new VSFrameData(size23, core->memory);
        }
    } else {
        //Handle GPU implementation.
        for (int plane = 0; plane < f->numPlanes; plane++) {
            int compensatedWidth  = (plane ? width  >> f->subSamplingW : width);
            int compensatedHeight = (plane ? height >> f->subSamplingH : height);

            data[plane] =
                new VSFrameData(compensatedWidth, compensatedHeight, &stride[plane], f->bytesPerSample,
                                core->gpuMemory, fLocation);
        }
    }
}

void VSFrame::transferFrame(VSFrame &dstFrame, FrameTransferDirection direction) const {
    if(dstFrame.width != width || dstFrame.height != height)
        qFatal("The source frame and destination frame dimensions do not match.");

    //Double check the strides, just to make sure cuda isn't pulling any nasty shit.
    //This can probably safely be removed.
    if(dstFrame.format->numPlanes != format->numPlanes)
        qFatal("The source frame and destination frame do not have the same number of planes.");

    for(int plane = 0; plane < format->numPlanes; plane++) {
        data[plane].data()->transferData(dstFrame.data[plane].data(), dstFrame.stride[plane],
                                                  stride[plane], getWidth(plane),
                                                  getHeight(plane), format->bytesPerSample, direction);
    }
}

PVideoFrame VSCore::newVideoFrame(const VSFormat *f, int width, int height, const VSFrame *propSrc, FrameLocation fLocation) {
    return PVideoFrame(new VSFrame(f, width, height, propSrc, this, fLocation));
}

void VSCore::transferVideoFrame(const PVideoFrame &srcf, PVideoFrame &dstf, FrameTransferDirection direction){
    srcf->transferFrame(*dstf.data(), direction);
}