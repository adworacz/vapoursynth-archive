DO NOT USE THIS
===============

This represents the pre-ist of pre-alphas for CUDA support in Vapoursynth and is completely unstable.

I repeat, DO NOT USE THIS BRANCH (YET).

TODO (for my use):
   * Cleanup the wscript, as we aren't building all of the filters currently.
   * Cleanup Merge and TransferFrame, as there is some alpha-ass code in there.
   * Move as much of the stream handling into the API as possible, including stream retrieval from Frame properties.
   * Investigate ways to improve perforance with multiple CPU threads.
   * Port all of the standard library over (the filters that actually operate on the video data).
   * Add support 9/10/16-bit video to the GPU kernels. (More likely add specific kernels for those cases.)

