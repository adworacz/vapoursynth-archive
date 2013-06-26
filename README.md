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

NOTES:
   * Expr: Expr does see a performance increase with the use of constant memory to contain the OPS. However, constant memory is scarce
   and heavy amounts of Expr use can hit the artificial limit set at compile time. The current implementation uses just Global Memory
   to store OPS, which is more adaptable and "infinitely scalable". Constant memory offers 5-10% performance boost for Expr, but
   it has limitations with respect to usability. So for now, we go scalable versus performance. Its still way faster than the SSE2,
   CPU version.

