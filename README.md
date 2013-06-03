DO NOT USE THIS
===============

This represents the pre-ist of pre-alphas for CUDA support in Vapoursynth and is completely unstable.

I repeat, DO NOT USE THIS BRANCH (YET).

TODO (for my use only):
   * Investigate ways to improve perforance with multiple CPU threads.
   * Port all of the standard library over (the filters that actually operate on the video data).
   * Add support 9/10/16-bit video to the GPU kernels. (Templates should solve this.)
   * Improve frame cache handling on GPU. Right now I'm not sure that it's properly handling the different CPU and GPU frame caches.
   * Investigate usage of shared memory in memory bound kernels in order to perform optimal coalescing. Items are currently coalesced,
     but there may be a possibility to increase DRAM utilization, either through the reduction of partition camping or larger global->shared
     memory acceses, a la CUDA SDK's Transpose kernel.
   * Remove the unnecessary VSFormat parameter passing in simplefilters.cu, as the VSVideoInfo class already has the format.
   * Change kernel wrappers to void from int for fully streamed frame data.
   * Add support for multiple GPUs.

Filters to be ported:
   * Lut2
   * CropAbs/CropRel
   * ShufflePlanes (should be easy)
   * SeperateFields
   * DoubleWeave
   * FlipVertical
   * FlipHorizontal
   * StackVertical
   * StackHorizontal
   * PlaneAverage/PlaneDifference (maybe...)

Filters ported:
   * Invert
   * Merge
   * Lut
   * MaskedMerge
   * BlankClip
   * AddBorders
   * Transpose
   * Expr

For the memory intensive stuff (shuffleplanes, transpose, etc...) more work and research should be put forth to avoid partition camping issues.
