ShufflePlanes
=======

.. function::   ShufflePlanes(clip[] clips, int[] planes, int format)
   :module: std
   
   ShufflePlanes can extract and combine planes from different clips in the most general way possible.
   This is both good and bad as there almost no error checks.
   
   Most of the returned clip's properties are implicitly determined from the first clip given to *clips*.
   Takes between 1 and 3 clips for 3 plane formats. In this case clips=[A] is equivalent to clips=[A, A, A] and clips=[A, B] is equivalent to clips=[A, B, B]. For 1 plane formats it takes exactly one clip.
   
   The argument *planes* controls which of the input clips' planes to use. Zero indexed.

   The only things that needs to be specified is *format*, which controls which color family (YUV, RGB, GRAY) the output clip will be.
   Properties such as subsampling are determined from the relative size of the given planes to combine.
   
   Here are some examples of useful operations...
   
   Extract plane with index X. X=0 will mean luma on a YUV clip and R on an RGB clip, likewise 1 will return the U and G channel respectively::
   
      ShufflePlanes(clips=clip, planes=X, format=vs.GRAY)
   
   Swap U and V in a YUV clip::
   
      ShufflePlanes(clips=clip, planes=[0, 2, 1], format=vs.YUV)
   
   Merge 3 grayscale clips into a YUV clip::
   
      ShufflePlanes(clips=[Yclip, Uclip, Vclip], planes=[0, 0, 0], format=vs.YUV)
   
   Cast a YUV clip to RGB::
   
      ShufflePlanes(clips=[YUVclip], planes=[0, 1, 2], format=vs.RGB)
   
   ShufflePlanes only accepts variable format and size frames when extracting a single plane.