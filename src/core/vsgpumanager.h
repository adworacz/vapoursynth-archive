/*
* Copyright (c) 2013 Fredrik Mellbin, Austin Dworaczyk Wiltshire
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
#ifndef VSGPUMANAGER
#define VSGPUMANAGER

#include <QtCore/QtCore>
#include <cuda_runtime.h>

struct VSGPUManager {
private:
    cudaStream_t *streams;
    int numberOfStreams;
    int streamIndex;
    QMutex lock;

public:
    int getStream(cudaStream_t *stream, int index = -1);
    int getStreams(cudaStream_t **desiredStreams, int numStreams);

    VSGPUManager();
    ~VSGPUManager();
};

#endif //VSGPUMANAGER
