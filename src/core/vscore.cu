// A CUDA specific file for all CUDA-aimed functions of the VSCore.

#include "vscore.h"
#include "VSHelper.h"
#include "x86utils.h"
#include "version.h"

#include "VSCuda.h"

// Filter headers
extern "C" {
#include "simplefilters.h"
#include "vsresize.h"
}
#include "cachefilter.h"
#include "exprfilter.h"

VSFrame::VSFrame(const VSFormat *f, int width, int height, const VSFrame *propSrc, VSCore *core, FrameLocation fLocation) : frameLocation(fLocation){
    if(fLocation == flLocal){
        VSFrame(f, width, height, propSrc, core);
    } else {
        if (!f || width <= 0 || height <= 0)
            qFatal("Invalid new frame");

        if (propSrc)
            properties = propSrc->properties;
    }
}