#ifndef VSCUDA
#define VSCUDA

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

//GPU Specific Defines.
//////////////////////////////////////////////////////////////////
//Maps to a single instruction on G8x / G9x / G10x
#define IMAD(a, b, c) ( mul24((a), (b)) + (c) )

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

#endif