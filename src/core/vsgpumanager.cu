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

#include <stdexcept>
#include "vsgpumanager.h"

VSGPUManager::VSGPUManager() {
    cudaDeviceProp * deviceProps = VSCUDAGetDefaultDeviceProperties();
    numberOfStreams = 1;

    if (deviceProps->concurrentKernels) {
        //Devices with Compute Capability 3.5 support up to 32 concurrent kernels.
        //Devices with Compute Capability less than 3.5 support 16 concurrent kernels.
        if (deviceProps->major >=3 && deviceProps->minor >= 5)
            numberOfStreams = 32;
        else
            numberOfStreams = 16;
    }

    streams = (VSCUDAStream *) malloc(numberOfStreams * sizeof(VSCUDAStream));

    //Initialize our streams.
    for (int i = 0; i < numberOfStreams; i++)
        CHECKCUDA(cudaStreamCreate(&(streams[i].stream)));

    //We are going to assign a stream per frame, or per plane,
    //but either way we need to see what happens when we incorporate lots of streams.
    streamIndex = 0;
}

VSCUDAStream * VSGPUManager::getNextStream() {
    VSCUDAStream *stream;

    lock.lock();
    stream = &streams[streamIndex];
    streamIndex = (streamIndex + 1) % numberOfStreams;
    lock.unlock();

    return stream;
}

VSGPUManager::~VSGPUManager() {
    for (int i = 0; i < numberOfStreams; i++) {
        CHECKCUDA(cudaStreamDestroy(streams[i].stream));
    }

    free(streams);

    CHECKCUDA(cudaDeviceSynchronize());
    //For some reason, calling cudaDeviceReset() here
    //causes a crash on script exit. Not quite sure why,
    //needs to be investigated more.
    //CHECKCUDA(cudaDeviceReset());
}
