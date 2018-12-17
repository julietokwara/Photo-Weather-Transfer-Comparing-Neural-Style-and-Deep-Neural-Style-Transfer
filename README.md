# Photo-Weather-Transfer-Comparing-Neural-Style-and-Deep-Neural-Style-Transfer

Authors: Juliet Okwara and Nicholas Seay

The aim of this repo is to tackle the challenge of photorealistic style transfer, specifically in the context of transferring the weather, sky, and ambience elements of one photo into another. Existing methodologies such as Neural Style Transfer by Gatys et al. has been proven to be successful in generating images that merge the content of a content reference image and the style of a style reference image into a single image. However, these generated images tend to work best in an artistic context. The noisy image which the network trains on is often changed in ways reminiscent of a painting, possibly created by the artist who made the style image, rather than a photo. This repo shows the results of introducing to the original algorithm the ideas of semantic image segmentation and the Matting Laplacian. 
## Our Dataset
Our dataset can be found in the examples folder. It is separated between our style, content, and segmentation images. In there you can also find fully run examples in the processed_examples folder. 

## Running

This code is written using Python 2.7

`python neural_style.py --content <content file> -- content_seg <content seg file> --styles <style file> --output <output file>`

Run `python neural_style.py --help` to see a list of all options.

Use `--checkpoint-output` and `--checkpoint-iterations` to save checkpoint images.

Use `--iterations` to change the number of iterations (default 1000).

## Example
python neural_style.py --content_seg ./examples/content_segs/content1_seg.png --content ./examples/content/content1.png --styles ./examples/style/style1.png  --output ./c1s1_matting/c1s1_matting_final.jpg --checkpoint-output './c1s1_matting/c1s1_matting_intermediate%s.jpg' --iterations 1000 --checkpoint-iterations 100 --style-layer-weight-exp 0.2 --pooling avg


## Requirements

### Data Files
* [Pre-trained VGG network][http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat] (MD5 `8ee3263992981a1d26e73b3ca028a123`) - put it in the top level of this repository, or specify its location using the `--network` option.

### Dependencies

You can install Python dependencies using `pip install -r requirements.txt`,
and it should just work. If you want to install the packages manually, here's a
list:

* [TensorFlow](https://www.tensorflow.org/versions/master/get_started/os_setup.html#download-and-setup)
* [NumPy](https://github.com/numpy/numpy/blob/master/INSTALL.rst.txt)
* [SciPy](https://github.com/scipy/scipy/blob/master/INSTALL.rst.txt)
* [Pillow](http://pillow.readthedocs.io/en/3.3.x/installation.html#installation)

You need at least one GPU to run this repo.

## Citations

Anish Athalye. Neural Style.
https://github.com/anishathalye/neural-style. 2015.

Luan, Fujun and Paris, Sylvain and Shechtman, Eli and Bala, Kavita. Deep Photo Style Transfer. 
https://github.com/luanfujun/deep-photo-styletransfer. 2017. 

Smith, Cameron. neural-style-tf. https://github.com/cysmith/neural-style-tf. 2016.

Yang Liu. deep-photo-style-transfer-tf.  https://github.com/LouieYang/deep-photo-styletransfer-tf. 2017.

Y. Shih, S. Paris, F. Durand, and W. T. Freeman, “Data-driven hallucination of different times of day from a single outdoor photo,” ACM Transactions on Graphics, vol. 32, no. 6, pp. 1–11, Jan. 2013.
