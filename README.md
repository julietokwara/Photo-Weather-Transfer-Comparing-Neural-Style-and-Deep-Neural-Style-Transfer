# Photo-Weather-Transfer-Comparing-Neural-Style-and-Deep-Neural-Style-Transfer
The aim of this repo is to tackle the challenge of photorealistic style transfer, specifically in the context of transferring the weather, sky, and ambience elements of one photo into another. Existing methodologies such as Neural Style Transfer by Gatys et al. has been proven to be successful in generating images that merge the content of a content reference image and the style of a style reference image into a single image. However, these generated images tend to work best in an artistic context. The noisy image which the network trains on is often changed in ways reminiscent of a painting, possibly created by the artist who made the style image, rather than a photo. This repo shows the results of introducing to the original algorithm the ideas of semantic image segmentation and the Matting Laplacian. 

## Running

`python neural_style.py --content <content file> --styles <style file> --output <output file>`

Run `python neural_style.py --help` to see a list of all options.

If you are running this project on [Floydhub](https://www.floydhub.com) you can use the following syntax (this pulls in the pre-trained VGG network automatically):

`floyd run --gpu --env tensorflow-1.3
--data  floydhub/datasets/imagenet-vgg-verydeep-19/3:vgg
"python neural_style.py --network /vgg/imagenet-vgg-verydeep-19.mat --content <content file> --styles <style file> --output <output file>"`


Use `--checkpoint-output` and `--checkpoint-iterations` to save checkpoint images.

Use `--iterations` to change the number of iterations (default 1000).  For a 512×512 pixel content file, 1000 iterations take 60 seconds on a GTX 1080 Ti, 90 seconds on a Maxwell Titan X, or 60 minutes on an Intel Core i7-5930K. Using a GPU is highly recommended due to the huge speedup.

## Example


## Requirements

### Data Files

* [Pre-trained VGG network][net] (MD5 `8ee3263992981a1d26e73b3ca028a123`) - put it in the top level of this repository, or specify its location using the `--network` option.

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

```
@misc{athalye2015neuralstyle,
  author = {Anish Athalye},
  title = {Neural Style},
  year = {2015},
  howpublished = {\url{https://github.com/anishathalye/neural-style}},
  note = {commit xxxxxxx}
}
```
Luan, Fujun and Paris, Sylvain and Shechtman, Eli and Bala, Kavita. Deep Photo Style Transfer. 
https://github.com/luanfujun/deep-photo-styletransfer. 2017. 


Smith, Cameron. neural-style-tf. https://github.com/cysmith/neural-style-tf. 2016.

Yang Liu. deep-photo-style-transfer-tf.  https://github.com/LouieYang/deep-photo-styletransfer-tf. 2017.

Y. Shih, S. Paris, F. Durand, and W. T. Freeman, “Data-driven hallucination of different times of day from a single outdoor photo,” ACM Transactions on Graphics, vol. 32, no. 6, pp. 1–11, Jan. 2013.
