# pix2pix with tf.keras -->tf.enable_eager_execution()
Pix2pix model for decolorization RGB2NIR
This repo is a reimplementation of [iPython version](https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/contrib/eager/python/examples/pix2pix/pix2pix_eager.ipynb) of pix2pix. In other words, is a copy with just a few modifications ;)

The images used for this task is from EPFL [RGBNIR dataset](https://ivrl.epfl.ch/research-2/research-downloads/supplementary_material-cvpr11-index-html/). 421 images were taken from the whole of dataset (90% for training and validation, and 10% for testing). The indoor scenes were not considered. They could be used for testing to :)
# How is it training

<div align='center'>
  <img src="figs/epoch2.png" width="800"/>
</div>

# Requeriments

* Python 3.7
* Tensorflow 1.13.1
* Matplotlib
* numpy

# Issues
* Memory ram leak in every interaction, it restarts with new epoch.

