ClipToProp
==========

.. function:: ClipToProp(clip clip, clip mclip[, data prop='_Alpha'])
   :module: std
   
   Stores each frame of *mclip* as a frame property named *prop* in *clip*. This is primarily intended
   to attach mask/alpha clips to another clip so that editing operations will apply to both.
   If the attached *mclip* does not represent the alpha channel you should set *prop* to something else.
   It is the inverse of PropToClip().