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
   * Improve frame cache handling on GPU. Right now I'm not sure that it's properly handling the different CPU and GPU frame caches.


Filters to be ported:
   * Lut2
   * MaskedMerge
   * AddBorders
   * BlankClip
   * CropAbs/CropRel
   * ShufflePlanes (should be easy)
   * SeperateFields
   * DoubleWeave
   * FlipVertical
   * FlipHorizontal
   * StackVertical
   * StackHorizontal
   * Transpose
   * PlaneAverage/PlaneDifference (maybe...)
   * Expr (This will probably see a large performance boost.)


Filters ported:
   * Invert
   * Merge
   * Lut


For the memory intensive stuff (shuffleplanes, transpose, etc...) more work and research should be put forth to avoid partition camping issues.
