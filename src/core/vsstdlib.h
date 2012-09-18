//  Copyright (c) 2012 Fredrik Mellbin
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to deal
//  in the Software without restriction, including without limitation the rights
//  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in
//  all copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
//  THE SOFTWARE.

///////////////////////////////////////////////////////////////////////////////
// Licensing notes for this file:
// This file contains the implementation of all base functions and therefore
// makes quite good example code showing how the VapourSynth API may be used.
// You are welcome to copy and paste code from single filters here and I will
// not be too picky with attribution.
// I also hope that if someone writes another backend for the VapourSynth API
// they will include this library of standard functions so that the
// implementations don't diverge too much in base functionality.

#ifndef VSSTDLIB_H
#define VSSTDLIB_H

#include "../../include/VapourSynth.h"

void stdlibInitialize(VSConfigPlugin configFunc, VSRegisterFunction registerFunc, VSPlugin *plugin);

#endif // VSSTDLIB_H
