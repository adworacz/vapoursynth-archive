// A CUDA specific file for all CUDA-aimed functions of the VSCore.

#include "vscore.h"
#include "VSCuda.h"

VSFrame::VSFrame(const VSFormat *f, int width, int height, const VSFrame *propSrc, VSCore *core, FrameLocation fLocation) : format(f), width(width), height(height), frameLocation(fLocation){
    if(fLocation == flLocal){
        VSFrame(f, width, height, propSrc, core);
    } else {
        if (!f || width <= 0 || height <= 0)
            qFatal("Invalid new frame");

        if (propSrc)
            properties = propSrc->properties;
    }
}