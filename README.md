DO NOT USE THIS
===============

This represents the pre-ist of pre-alphas for CUDA support in Vapoursynth and is completely unstable.

I repeat, DO NOT USE THIS BRANCH (YET).

TODO (for my use):
   * Investigate ways to improve perforance with multiple CPU threads.
   * Port all of the standard library over (the filters that actually operate on the video data).
   * Add support 9/10/16-bit video to the GPU kernels. (More likely add specific kernels for those cases.)
   * Try and remove the depency on <cuda_runtime.h> in VapourSynth.h, as it forces non-cuda filters to import cuda libs.
   * Write tests for comparing filter output between the GPU and CPU.

