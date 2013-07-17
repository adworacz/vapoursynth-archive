Lut2
=======

.. function:: Lut2(clip[] clips, int[] planes[, int[] lut, func function, int bits])
   :module: std

   Applies a lut that takes the pixel values of two clips into account. The *lut* needs to contain 2^(clip1.bits_per_sample + clip2.bits_per_sample) entries and will be applied to the planes listed in *planes*. Alternatively a *function* taking *x* and *y* as arguments can be used to make the lut. The other planes will be passed through unchanged.

   Lut2 also takes an optional bit depth parameter, *bits*. which defaults to the bit depth of the first input clip, and specifies the bit depth of the output clip. The user is responsible for understanding the effects of bit depth conversion, specifically from higher bit depths to lower bit depths, as no scaling or clamping is applied.

   How to average 2 clips::

      lut = []
      for y in range(2 ** clipy.format.bits_per_sample):
         for x in range(2 ** clipx.format.bits_per_sample):
            lut.append((x + y)//2)
      Lut2(clips=[clipx, clipy], lut=lut, planes=[0, 1, 2])

   How to average 2 clips with a 10-bit output::

      def function(x, y)
         return (x*4 + y)//2
      Lut2(clips=[clipx8bit, clipy10bit], planes=[0, 1, 2], function=, bits=10)
