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
#ifndef VSCUDA
#define VSCUDA

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

//GPU Specific Defines.
//////////////////////////////////////////////////////////////////

//Some usefull CUDA error checking functions.
#define CHECKCUDA(err)  __checkCudaErrors (err, __FILE__, __LINE__)

inline void __checkCudaErrors( cudaError err, const char *file, const int line )
{
    if( cudaSuccess != err) {
        fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",
            file, line, (int)err, cudaGetErrorString( err ) );
        exit(-1);
    }
}

static cudaDeviceProp *VSCUDAGetDefaultDeviceProperties() {
    int deviceID = 0;
    static int propertiesRetrieved = 0;
    static cudaDeviceProp deviceProp;

    if(!propertiesRetrieved) {
        propertiesRetrieved = 1;
        CHECKCUDA(cudaGetDeviceProperties(&deviceProp, deviceID));
    }
    return &deviceProp;
}

static inline int VSCUDAGetBasicBlocksize() {
    cudaDeviceProp * deviceProp = VSCUDAGetDefaultDeviceProperties();
    return (deviceProp->major < 2) ? 16 : 32;
}

struct VSCUDAStream {
    cudaStream_t stream;
};

#endif
