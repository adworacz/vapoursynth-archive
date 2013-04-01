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

//Transfer video frame data asynchronously using the given cudaStream.
void VSFrameData::transferData(VSFrameData *dst, int dstStride,
                               int srcStride, int width, int height, int bytesPerSample,
                               FrameTransferDirection direction) const {
    cudaMemcpyKind transferKind = (direction == ftdCPUtoGPU ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost);

    //We are not using cudaMemcpy2DAsync for a reason here.
    //Strangely, it offers no performance benefit, and tests showed a strange glitch error
    //occured on the first few frames of a clip with certain combinations of the Merge filter.
    //No idea why the error occurred, but using cudaMemcpy2D (no async) is safe (glitch-free)
    //and offers the same performance.
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

    if (format->numPlanes != 3) {
        stride[1] = 0;
        stride[2] = 0;
    }

    if (frameLocation == flLocal) {
        //Handle CPU implementation.
        //This is a simple copy and paste of the vscore.cpp VSFrame constructor.
        stride[0] = (width * (f->bytesPerSample) + (alignment - 1)) & ~(alignment - 1);

        if (f->numPlanes == 3) {
            int plane23 = ((width >> f->subSamplingW) * (f->bytesPerSample) + (alignment - 1)) & ~(alignment - 1);
            stride[1] = plane23;
            stride[2] = plane23;
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
                                core->gpuMemory, frameLocation);
        }
    }
}

VSFrame::VSFrame(const VSFormat *f, int width, int height, const VSFrame * const *planeSrc, const int *plane, const VSFrame *propSrc, VSCore *core, FrameLocation fLocation) : format(f), width(width), height(height), frameLocation(fLocation) {
    if (!f || width <= 0 || height <= 0)
        qFatal("Invalid new frame");

    if (propSrc)
        properties = propSrc->properties;

    if (format->numPlanes != 3) {
        stride[1] = 0;
        stride[2] = 0;
    }

    //Calculate the stride.
    //WARNING: This stride gets over written when allocating on the GPU, in order to meet
    //GPU memory alignment requirements.
    stride[0] = (width * (f->bytesPerSample) + (alignment - 1)) & ~(alignment - 1);

    if (f->numPlanes == 3) {
        int plane23 = ((width >> f->subSamplingW) * (f->bytesPerSample) + (alignment - 1)) & ~(alignment - 1);
        stride[1] = plane23;
        stride[2] = plane23;
    }

    for (int i = 0; i < format->numPlanes; i++) {
        if (planeSrc[i]) {
            if (plane[i] < 0 || plane[i] >= planeSrc[i]->format->numPlanes)
                qFatal("Plane does no exist, error in frame creation");
            if (planeSrc[i]->getHeight(plane[i]) != getHeight(i) || planeSrc[i]->getWidth(plane[i]) != getWidth(i))
                qFatal("Copied plane dimensions do not match, error in frame creation");
            data[i] = planeSrc[i]->data[plane[i]];
            stride[i] = planeSrc[i]->stride[plane[i]];
        } else {
            int compensatedWidth  = (i ? width  >> f->subSamplingW : width);
            int compensatedHeight = (i ? height >> f->subSamplingH : height);

            if (frameLocation == flLocal)
                data[i] = new VSFrameData(stride[i] * compensatedHeight, core->memory);
            else
                data[i] = new VSFrameData(compensatedWidth, compensatedHeight, &stride[i], f->bytesPerSample,
                            core->gpuMemory, frameLocation);
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

PVideoFrame VSCore::newVideoFrame(const VSFormat *f, int width, int height, const VSFrame * const *planeSrc, const int *planes, const VSFrame *propSrc, FrameLocation fLocation) {
    return PVideoFrame(new VSFrame(f, width, height, planeSrc, planes, propSrc, this, fLocation));
}

void VSCore::transferVideoFrame(const PVideoFrame &srcf, PVideoFrame &dstf, FrameTransferDirection direction){
    srcf->transferFrame(*dstf.data(), direction);
}

VSGPUManager *VSCore::getGPUManager() const {
    return gpuManager;
}