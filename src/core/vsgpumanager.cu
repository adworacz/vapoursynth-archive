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

#include "vsgpumanager.h"
#include "VSCuda.h"

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

    streams = (cudaStream_t *) malloc(numberOfStreams * sizeof(cudaStream_t));

    //Initialize our streams.
    for (int i = 0; i < numberOfStreams; i++)
        cudaStreamCreate(&streams[i]);

    //Reserve our send and receive streams.
    if (deviceProps->asyncEngineCount > 0) {
        sendStream = streams[0];
        receiveStream = streams[1];
    } else {
        sendStream = receiveStream = streams[0];
    }

    streamIndex = 2;
}

void VSGPUManager::getSendStream(cudaStream_t *stream) {
    *stream = sendStream;
}

void VSGPUManager::getReceiveStream(cudaStream_t *stream) {
    *stream = receiveStream;
}

void VSGPUManager::getStreams(cudaStream_t **desiredStreams, int numStreams) {
    lock.lock();
    for (int i = 0; i < numStreams; i++){
        (*desiredStreams)[i] = streams[streamIndex];

        streamIndex++;
        if (streamIndex > numberOfStreams)
            streamIndex = 2;
    }
    lock.unlock();
}

VSGPUManager::~VSGPUManager() {
    for (int i = 0; i < numberOfStreams; i++)
        cudaStreamDestroy(streams[i]);
}