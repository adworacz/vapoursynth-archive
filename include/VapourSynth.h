// This header is provided for preview purposes but no big changes are planned

#ifndef VAPOURSYNTH_H
#define VAPOURSYNTH_H

#include <stdint.h>

#define VAPOURSYNTH_API_VERSION 1

// Convenience for C++ users.
#ifdef __cplusplus
#	define VS_EXTERN_C extern "C"
#else
#	define VS_EXTERN_C
#endif

#ifdef _WIN32
#	define VS_CC __stdcall
#else
#	define VS_CC
#endif

// And now for some symbol hide-and-seek...
#if defined(_MSC_VER) // MSVC
#	if defined(VSCORE_EXPORTS) // building the VSCore library itself, with visible API symbols
#		define VS_API(ret) VS_EXTERN_C __declspec(dllexport) ret VS_CC
#	else // building something that depends on VS
#		define VS_API(ret) VS_EXTERN_C __declspec(dllimport) ret VS_CC
#	endif // defined(VSCORE_EXPORTS)
// GCC 4 or later: export API symbols only. Some GCC 3.x versions support the visibility attribute too,
// but we don't really care enough about that to add compatibility defines for it.
#elif defined(__GNUC__) && __GNUC__ >= 4
#	define VS_API(ret) VS_EXTERN_C __attribute__((visibility("default"))) ret VS_CC
#else // fallback for everything else
#	define VS_API(ret) VS_EXTERN_C ret VS_CC
#endif // defined(_MSC_VER)

typedef struct VSFrameRef VSFrameRef;
typedef struct VSNodeRef VSNodeRef;
typedef struct VSCore VSCore;
typedef struct VSPlugin VSPlugin;
typedef struct VSNode VSNode;
typedef struct VSFuncRef VSFuncRef;
typedef struct VSMap VSMap;
typedef struct VSAPI VSAPI;
typedef struct VSFrameContext VSFrameContext;

typedef enum VSColorFamily {
    // all planar formats
    cmGray		= 1000000,
    cmRGB		= 2000000,
    cmYUV		= 3000000,
    cmYCoCg		= 4000000,
    // special for compatibility, if you implement these in new filters I'll personally kill you
    cmCompat	= 9000000
} VSColorFamily;

typedef enum VSSampleType {
    stInteger = 0,
    stFloat = 1
} VSSampleType;

// The +10 is so people won't be using the constants interchangably "by accident"
typedef enum VSPresetFormat {
    pfNone = 0,

    pfGray8 = cmGray + 10,
    pfGray16,

    pfYUV420P8 = cmYUV + 10,
    pfYUV422P8,
    pfYUV444P8,
    pfYUV410P8,
    pfYUV411P8,
    pfYUV440P8,

    pfYUV420P9,
    pfYUV422P9,
    pfYUV444P9,

    pfYUV420P10,
    pfYUV422P10,
    pfYUV444P10,

    pfYUV420P16,
    pfYUV422P16,
    pfYUV444P16,

    pfRGB24 = cmRGB + 10,
    pfRGB27,
    pfRGB30,
    pfRGB48,

    // special for compatibility, if you implement these in any filter I'll personally kill you
    // I'll also change their ids around to break your stuff regularly
    pfCompatBGR32 = cmCompat + 10,
    pfCompatYUY2
} VSPresetFormat;

typedef enum VSFilterMode {
    fmParallel = 100, // completely parallel execution
    fmParallelRequests = 200, // for filters that are serial in nature but can request one or more frames they need in advance
    fmUnordered = 300, // for filters that modify their internal state every request
    fmSerial = 400 // for source filters and compatibility with other filtering architectures
} VSFilterMode;

// possibly add d3d storage on gpu, since the gpu memory can be mapped into normal address space
typedef struct VSFormat {
    char name[32];
    int id;
    int colorFamily; // gray/rgb/yuv/ycocg
    int sampleType; // int/float
    int bitsPerSample; // number of significant bits
    int bytesPerSample; // actual storage is always in a power of 2 and the smallest possible that can fit the number of bits used per sample

    int subSamplingW; // log2 subsampling factor, applied to second and third plane
    int subSamplingH;

    int numPlanes; // implicit from colorFamily, 1 or 3 in the currently specified ones
} VSFormat;

typedef enum NodeFlags {
    nfNoCache	= 1,
} NodeFlags;

typedef enum GetPropErrors {
    peUnset	= 1,
    peType	= 2,
    peIndex	= 4
} GetPropErrors;

typedef struct VSVersion {
    int core;
    int api;
    const char *versionString;
} VSVersion;

typedef struct VSVideoInfo {
    // add node name?
    const VSFormat *format;
    int64_t fpsNum;
    int64_t fpsDen;
    int width;
    int height;
    int numFrames;
    int flags; // expose in some other way?
} VSVideoInfo;

typedef enum ActivationReason {
    arInitial = 0,
    arFrameReady = 1,
    arAllFramesReady = 2,
    arError = -1
} ActivationReason;

// function typedefs
typedef	VSCore *(VS_CC *VSCreateVSCore)(int threads);
typedef	void (VS_CC *VSFreeVSCore)(VSCore *core);

// function/filter typedefs
typedef void (VS_CC *VSPublicFunction)(const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi);
typedef void (VS_CC *VSFreeFuncData)(void *userData);
typedef void (VS_CC *VSFilterInit)(VSMap *in, VSMap *out, void **instanceData, VSNode *node, VSCore *core, const VSAPI *vsapi);
typedef const VSFrameRef *(VS_CC *VSFilterGetFrame)(int n, int activationReason, void **instanceData, void **frameData, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi);
typedef void (VS_CC *VSFilterFree)(void *instanceData, VSCore *core, const VSAPI *vsapi);
typedef void (VS_CC *VSRegisterFunction)(const char *name, const char *args, VSPublicFunction argsFunc, void *functionData, VSPlugin *plugin);
typedef const VSNodeRef *(VS_CC *VSCreateFilter)(const VSMap *in, VSMap *out, const char *name, VSFilterInit init, VSFilterGetFrame getFrame, VSFilterFree free, int filterMode, int flags, void *instanceData, VSCore *core);
typedef VSMap *(VS_CC *VSInvoke)(VSPlugin *plugin, const char *name, const VSMap *args);
typedef void (VS_CC *VSSetError)(VSMap *map, const char *errorMessage);
typedef const char *(VS_CC *VSGetError)(const VSMap *map);
typedef void (VS_CC *VSSetFilterError)(const char *errorMessage, VSFrameContext *frameCtx);

typedef const VSFormat *(VS_CC *VSGetFormatPreset)(int id, VSCore *core);
typedef const VSFormat *(VS_CC *VSRegisterFormat)(int colorFamily, int sampleType, int bitsPerSample, int subSamplingW, int subSamplingH, VSCore *core);

// frame and clip handling
typedef void (VS_CC *VSFrameDoneCallback)(void *userData, const VSFrameRef *f, int n, const VSNodeRef *, const char *errorMsg);
typedef void (VS_CC *VSGetFrameAsync)(int n, const VSNodeRef *node, VSFrameDoneCallback callback, void *userData);
typedef const VSFrameRef *(VS_CC *VSGetFrame)(int n, const VSNodeRef *node, char *errorMsg, int bufSize);
typedef void (VS_CC *VSRequestFrameFilter)(int n, const VSNodeRef *node, VSFrameContext *frameCtx);
typedef const VSFrameRef *(VS_CC *VSGetFrameFilter)(int n, const VSNodeRef *node, VSFrameContext *frameCtx);
typedef const VSFrameRef *(VS_CC *VSCloneFrameRef)(const VSFrameRef *f);
typedef const VSNodeRef *(VS_CC *VSCloneNodeRef)(const VSNodeRef *node);
typedef void (VS_CC *VSFreeFrame)(const VSFrameRef *f);
typedef void (VS_CC *VSFreeNode)(const VSNodeRef *node);
typedef VSFrameRef *(VS_CC *VSNewVideoFrame)(const VSFormat *format, int width, int height, const VSFrameRef *propSrc, VSCore *core);
typedef VSFrameRef *(VS_CC *VSCopyFrame)(const VSFrameRef *f, VSCore *core);
typedef void (VS_CC *VSCopyFrameProps)(const VSFrameRef *src, VSFrameRef *dst, VSCore *core);

typedef int (VS_CC *VSGetStride)(const VSFrameRef *f, int plane);
typedef const uint8_t *(VS_CC *VSGetReadPtr)(const VSFrameRef *f, int plane);
typedef uint8_t *(VS_CC *VSGetWritePtr)(VSFrameRef *f, int plane);

// property access
typedef const VSVideoInfo *(VS_CC *VSGetVideoInfo)(const VSNodeRef *node);
typedef void (VS_CC *VSSetVideoInfo)(const VSVideoInfo *vi, VSNode *node);
typedef const VSFormat *(VS_CC *VSGetFrameFormat)(const VSFrameRef *f);
typedef int (VS_CC *VSGetFrameWidth)(const VSFrameRef *f, int plane);
typedef int (VS_CC *VSGetFrameHeight)(const VSFrameRef *f, int plane);
typedef const VSMap *(VS_CC *VSGetFramePropsRO)(const VSFrameRef *f);
typedef VSMap *(VS_CC *VSGetFramePropsRW)(VSFrameRef *f);
typedef int (VS_CC *VSPropNumKeys)(const VSMap *map);
typedef const char *(VS_CC *VSPropGetKey)(const VSMap *map, int index);
typedef int (VS_CC *VSPropNumElements)(const VSMap *map, const char *key);
typedef char(VS_CC *VSPropGetType)(const VSMap *map, const char *key);

typedef VSMap *(VS_CC *VSNewMap)(void);
typedef void (VS_CC *VSFreeMap)(VSMap *map);
typedef void (VS_CC *VSClearMap)(VSMap *map);

typedef int64_t (VS_CC *VSPropGetInt)(const VSMap *map, const char *key, int index, int *error);
typedef double(VS_CC *VSPropGetFloat)(const VSMap *map, const char *key, int index, int *error);
typedef const char *(VS_CC *VSPropGetData)(const VSMap *map, const char *key, int index, int *error);
typedef int (VS_CC *VSPropGetDataSize)(const VSMap *map, const char *key, int index, int *error);
typedef const VSNodeRef *(VS_CC *VSPropGetNode)(const VSMap *map, const char *key, int index, int *error);
typedef const VSFrameRef *(VS_CC *VSPropGetFrame)(const VSMap *map, const char *key, int index, int *error);

typedef int (VS_CC *VSPropDeleteKey)(VSMap *map, const char *key);
typedef int (VS_CC *VSPropSetInt)(VSMap *map, const char *key, int64_t i, int append);
typedef int (VS_CC *VSPropSetFloat)(VSMap *map, const char *key, double d, int append);
typedef int (VS_CC *VSPropSetData)(VSMap *map, const char *key, const char *data, int size, int append);
typedef int (VS_CC *VSPropSetNode)(VSMap *map, const char *key, const VSNodeRef *node, int append);
typedef int (VS_CC *VSPropSetFrame)(VSMap *map, const char *key, const VSFrameRef *f, int append);

typedef void (VS_CC *VSConfigPlugin)(const char *identifier, const char *defaultNamespace, const char *name, int apiVersion, int readonly, VSPlugin *plugin);
typedef void (VS_CC *VSInitPlugin)(VSConfigPlugin configFunc, VSRegisterFunction registerFunc, VSPlugin *plugin);

typedef VSPlugin *(VS_CC *VSGetPluginId)(const char *identifier, VSCore *core);
typedef VSPlugin *(VS_CC *VSGetPluginNs)(const char *ns, VSCore *core);

typedef VSMap *(VS_CC *VSGetPlugins)(VSCore *core);
typedef VSMap *(VS_CC *VSGetFunctions)(VSPlugin *plugin);

// fixme, move this stuff around before the API is final
typedef const VSVersion *(VS_CC *VSGetVersion)(void);

typedef VSFuncRef *(VS_CC *VSPropGetFunc)(const VSMap *map, const char *key, int index, int *error);
typedef int (VS_CC *VSPropSetFunc)(VSMap *map, const char *key, VSFuncRef *func, int append);
typedef void (VS_CC *VSCallFunc)(VSFuncRef *func, const VSMap *in, VSMap *out, VSCore *core, const VSAPI *vsapi);
typedef VSFuncRef *(VS_CC *VSCreateFunc)(VSPublicFunction func, void *userData, VSFreeFuncData free);
typedef void (VS_CC *VSFreeFunc)(VSFuncRef *f);

typedef void (VS_CC *VSQueryCompletedFrame)(const VSNodeRef **node, int *n, VSFrameContext *frameCtx);
typedef void (VS_CC *VSReleaseFrameEarly)(const VSNodeRef *node, int n, VSFrameContext *frameCtx);

// make plane memory allocation independent and provide a complicated copy function?
// VSCopyFrame equivalent to VSCopyFrame2({f, f, f}, f.vi.format, 0, core); VSCopyFrameProps()
//typedef VSFrameRef *(VS_CC *VSCopyFrame2)(const VSFrameRef *f[], const VSFormat *format, int clobber, VSCore *core);


typedef struct VSAPI {
    VSCreateVSCore createVSCore;
    VSFreeVSCore freeVSCore;
    VSCloneFrameRef cloneFrameRef;
    VSCloneNodeRef cloneNodeRef;
    VSFreeFrame freeFrame;
    VSFreeNode freeNode;
    VSNewVideoFrame newVideoFrame;
    VSCopyFrame copyFrame;
    VSCopyFrameProps copyFrameProps;
    VSRegisterFunction registerFunction;
    VSGetPluginId getPluginId;
    VSGetPluginNs getPluginNs;
    VSGetPlugins getPlugins;
    VSGetFunctions getFunctions;
    VSCreateFilter createFilter;
    VSSetError setError;
    VSGetError getError;
    VSSetFilterError setFilterError; //use to signal errors in the filter getframe function
    VSInvoke invoke;
    VSGetFormatPreset getFormatPreset;
    VSRegisterFormat registerFormat;
    VSGetFrame getFrame; // for external applications using the core as a library/for exceptional use if frame requests are necessary during filter initialization
    VSGetFrameAsync getFrameAsync; // for external applications using the core as a library
    VSGetFrameFilter getFrameFilter; // only use inside a filter's getframe function
    VSRequestFrameFilter requestFrameFilter; // only use inside a filter's getframe function
    VSGetStride getStride;
    VSGetReadPtr getReadPtr;
    VSGetWritePtr getWritePtr;

    //property access functions
    VSNewMap newMap;
    VSFreeMap freeMap;
    VSClearMap clearMap;

    VSGetVideoInfo getVideoInfo;
    VSSetVideoInfo setVideoInfo;
    VSGetFrameFormat getFrameFormat;
    VSGetFrameWidth getFrameWidth;
    VSGetFrameHeight getFrameHeight;
    VSGetFramePropsRO getFramePropsRO;
    VSGetFramePropsRW getFramePropsRW;

    VSPropNumKeys propNumKeys;
    VSPropGetKey propGetKey;
    VSPropNumElements propNumElements;
    VSPropGetType propGetType;
    VSPropGetInt propGetInt;
    VSPropGetFloat propGetFloat;
    VSPropGetData propGetData;
    VSPropGetDataSize propGetDataSize;
    VSPropGetNode propGetNode;
    VSPropGetFrame propGetFrame;

    VSPropDeleteKey propDeleteKey;
    VSPropSetInt propSetInt;
    VSPropSetFloat propSetFloat;
    VSPropSetData propSetData;
    VSPropSetNode propSetNode;
    VSPropSetFrame propSetFrame;

    VSGetVersion getVersion;

    VSPropGetFunc propGetFunc;
    VSPropSetFunc propSetFunc;
    VSCallFunc callFunc;
    VSCreateFunc createFunc;
    VSFreeFunc freeFunc;

    VSQueryCompletedFrame queryCompletedFrame;
    VSReleaseFrameEarly releaseFrameEarly;
} VSAPI;

VS_API(const VSAPI *) VS_CC getVapourSynthAPI(int version);

#endif // VAPOURSYNTH_H
