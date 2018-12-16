# Adapte from Anish Athalye. Released under GPLv3.

import os
import time
from collections import OrderedDict

from PIL import Image
from scipy import misc
from closed_form_matting import getLaplacian
from functools import partial
import numpy as np
import tensorflow as tf
import cv2

import vgg


CONTENT_LAYERS = ('relu4_2', 'relu5_2')
STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
VGG_MEAN = [103.939, 116.779, 123.68]

try:
    reduce
except NameError:
    from functools import reduce


def get_loss_vals(loss_store):
    return OrderedDict((key, val.eval()) for key,val in loss_store.items())


def print_progress(loss_vals):
    for key,val in loss_vals.items():
        print()
def affine_loss(output, M, weight):
    loss_affine = 0.0
    output_t = output / 255.
    for Vc in tf.unstack(output_t, axis=-1):
        Vc_ravel = tf.reshape(tf.transpose(Vc), [-1])
        loss_affine += tf.matmul(tf.expand_dims(Vc_ravel, 0), tf.sparse_tensor_dense_matmul(M, tf.expand_dims(Vc_ravel, -1)))

    return loss_affine * weight

def print_loss(max_iter, loss_content, loss_styles_list, loss_tv, loss_affine, overall_loss, output_image, serial='./', save_iter=100):
    global iter_count, min_loss, best_image
    if iter_count % print_iter == 0:
        print('Iteration {} / {}\n\tContent loss: {}'.format(iter_count, max_iter, loss_content))
        for j, style_loss in enumerate(loss_styles_list):
            print('\tStyle {} loss: {}'.format(j + 1, style_loss))
        print('\tTV loss: {}'.format(loss_tv))
        print('\tAffine loss: {}'.format(loss_affine))
        print('\tTotal loss: {}'.format(overall_loss - loss_affine))

    if overall_loss < min_loss:
        min_loss, best_image = overall_loss, output_image

    if iter_count % save_iter == 0 and iter_count != 0:
        save_result(best_image[:, :, ::-1], os.path.join(serial, 'out_iter_{}.png'.format(iter_count)))

    iter_count += 1


def stylize(network, initial, initial_noiseblend, content, styles,content_seg, style_seg, preserve_colors, iterations,
        content_weight, content_weight_blend, style_weight, style_layer_weight_exp, style_blend_weights, tv_weight,
        learning_rate, beta1, beta2, epsilon, pooling,
        print_iterations=None, checkpoint_iterations=None):
    """
    Stylize images.

    This function yields tuples (iteration, image, loss_vals) at every
    iteration. However `image` and `loss_vals` are None by default. Each
    `checkpoint_iterations`, `image` is not None. Each `print_iterations`,
    `loss_vals` is not None.

    `loss_vals` is a dict with loss values for the current iteration, e.g.
    ``{'content': 1.23, 'style': 4.56, 'tv': 7.89, 'total': 13.68}``.

    :rtype: iterator[tuple[int,image]]
    """
    shape = (1,) + content.shape
    style_shapes = [(1,) + style.shape for style in styles]
    content_features = {}
    style_features = [{} for _ in styles]

    vgg_weights, vgg_mean_pixel = vgg.load_net(network)

    layer_weight = 1.0
    style_layers_weights = {}
    for style_layer in STYLE_LAYERS:
        style_layers_weights[style_layer] = layer_weight
        layer_weight *= style_layer_weight_exp
    layer_weights_sum = 0
    for style_layer in STYLE_LAYERS:
        layer_weights_sum += style_layers_weights[style_layer]
    for style_layer in STYLE_LAYERS:
        style_layers_weights[style_layer] /= layer_weights_sum

    # compute content features in feedforward mode
    g = tf.Graph()
    with g.as_default(), g.device('/gpu:0'), tf.Session() as sess:
        image = tf.placeholder('float', shape=shape)
        net = vgg.net_preloaded(vgg_weights, image, pooling)
        content_pre = np.array([vgg.preprocess(content, vgg_mean_pixel)])
        for layer in CONTENT_LAYERS:
            content_features[layer] = net[layer].eval(feed_dict={image: content_pre})

    # compute style features in feedforward mode
    rgb2 = content_seg
    rgb2 = rgb2.astype(np.float32)
    mx = np.amax(rgb2)
    rgb2 /= mx
    for i in range(len(styles)):
        g = tf.Graph()
        with g.as_default(), g.device('/gpu:0'), tf.Session() as sess:

            image = tf.placeholder('float', shape=style_shapes[i])
            net = vgg.net_preloaded(vgg_weights, image, pooling)
            style_pre = np.array([vgg.preprocess(styles[i], vgg_mean_pixel)])
            for layer in STYLE_LAYERS:
                features = net[layer].eval(feed_dict={image: style_pre})#''', mask: content_seg''' 
                rgb2 = cv2.resize(rgb2, dsize=(features.shape[2], features.shape[1]), interpolation=cv2.INTER_AREA)
                rgb2tense = tf.convert_to_tensor(rgb2)
                n, height, width, d = map(lambda i: i.value, net[layer].get_shape()) #expand dims 3
                tensors = []
                for _ in range(d):
                    tensors.append(rgb2tense)
                rgb2tense = tf.stack(tensors, axis=2)
                rgb2tense = tf.stack(rgb2tense, axis=0)
                rgb2tense = tf.expand_dims(rgb2tense, 0)
                features =  tf.multiply( features,   tf.cast(rgb2tense, tf.float32)     )
                features = np.reshape(features.eval(feed_dict={image: style_pre}), (-1, features.shape[3]))
                gram = np.matmul(features.T, features) / features.size
                style_features[i][layer] = gram

    initial_content_noise_coeff = 1.0 - initial_noiseblend

    # make stylized image using backpropogation
    with tf.Graph().as_default():
        if initial is None:
            noise = np.random.normal(size=shape, scale=np.std(content) * 0.1)
            initial = tf.random_normal(shape) * 0.256
        else:
            initial = np.array([vgg.preprocess(initial, vgg_mean_pixel)])
            initial = initial.astype('float32')
            noise = np.random.normal(size=shape, scale=np.std(content) * 0.1)
            initial = (initial) * initial_content_noise_coeff + (tf.random_normal(shape) * 0.256) * (1.0 - initial_content_noise_coeff) 
        image = tf.Variable(initial)
        net = vgg.net_preloaded(vgg_weights, image, pooling)
        # content loss
        content_layers_weights = {}
        content_layers_weights['relu4_2'] = content_weight_blend
        content_layers_weights['relu5_2'] = 1.0 - content_weight_blend

        content_loss = 0
        content_losses = []
        for content_layer in CONTENT_LAYERS:
            content_losses.append(content_layers_weights[content_layer] * content_weight * (2 * tf.nn.l2_loss(
                    net[content_layer] - content_features[content_layer]) /
                    content_features[content_layer].size))
        content_loss += reduce(tf.add, content_losses)

        # style loss
        style_loss = 0
        rgb= content_seg
        rgb = rgb.astype(np.float32)
        mx = np.amax(rgb)
        rgb /= mx
        for i in range(len(styles)):
            style_losses = []
            for style_layer in STYLE_LAYERS:
                rgb = cv2.resize(rgb, dsize=(net[style_layer].shape[2], net[style_layer].shape[1]), interpolation=cv2.INTER_AREA)
                rgbtense = tf.convert_to_tensor(rgb)
                layer = net[style_layer]
                n, height, width, number = map(lambda i: i.value, layer.get_shape())
                tensors = []
                for _ in range(number):
                    tensors.append(rgbtense)
                rgbtense = tf.stack(tensors, axis=2)
                rgbtense = tf.stack(rgbtense, axis=0)
                rgbtense = tf.expand_dims(rgbtense, 0)
                layer = tf.multiply( net[style_layer],   tf.cast(rgbtense, tf.float32) )
                size = height * width * number
                feats = tf.reshape(layer, (-1, number))
                gram = tf.matmul(tf.transpose(feats), feats) / size
                style_gram = style_features[i][style_layer]
                style_losses.append(style_layers_weights[style_layer]  * 2 * tf.nn.l2_loss(gram - style_gram) / style_gram.size)
            style_loss += style_weight * style_blend_weights[i] * reduce(tf.add, style_losses)

        # total variation denoising
        tv_y_size = _tensor_size(image[:,1:,:,:])
        tv_x_size = _tensor_size(image[:,:,1:,:])
        tv_loss = tv_weight * 2 * (
                (tf.nn.l2_loss(image[:,1:,:,:] - image[:,:shape[1]-1,:,:]) /
                    tv_y_size) +
                (tf.nn.l2_loss(image[:,:,1:,:] - image[:,:,:shape[2]-1,:]) /
                    tv_x_size))

        # total loss
        M = tf.to_float(getLaplacian(content / 255.))
        mean_pixel = tf.constant(VGG_MEAN)
        input_image = tf.Variable(initial)
        input_image_plus = tf.squeeze(input_image + mean_pixel, [0])
        loss_affine = affine_loss(input_image_plus, M, 1e4)

        loss = content_loss + style_loss + tv_loss+ loss_affine


        # We use OrderedDict to make sure we have the same order of loss types
        # (content, tv, style, total) as defined by the initial costruction of
        # the loss_store dict. This is important for print_progress() and
        # saving loss_arrs (column order) in the main script.
        #
        # Subtle Gotcha (tested with Python 3.5): The syntax
        # OrderedDict(key1=val1, key2=val2, ...) does /not/ create the same
        # order since, apparently, it first creates a normal dict with random
        # order (< Python 3.7) and then wraps that in an OrderedDict. We have
        # to pass in a data structure which is already ordered. I'd call this a
        # bug, since both constructor syntax variants result in different
        # objects. In 3.6, the order is preserved in dict() in CPython, in 3.7
        # they finally made it part of the language spec. Thank you!
        loss_store = OrderedDict([('content', content_loss),
                                  ('style', style_loss),
                                  ('tv', tv_loss),
                                  ('total', loss),
                                  ('affine_loss', loss_affine)
                                  ])

        # optimizer setup
        optimizer  = tf.contrib.opt.ScipyOptimizerInterface(loss, method='L-BFGS-B', options={'maxiter': iterations, 'disp': 0})
	
        # optimization
        best_loss = float('inf')
        best = None
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print_loss_partial = partial(print_loss, iterations)
            optimizer.minimize(sess, fetches=[loss_store['content'], loss_store['style'], loss_store['tv'], loss_store['affine_loss'], loss_store['total'], input_image_plus])
            print('Optimization started...')
            iteration_times = []
            start = time.time()
            for i in range(iterations):
                iteration_start = time.time()
                if i > 0:
                    elapsed = time.time() - start
                    # take average of last couple steps to get time per iteration
                    remaining = np.mean(iteration_times[-10:]) * (iterations - i)
                    print('Iteration %4d/%4d (%s elapsed, %s remaining)' % (
                        i + 1,
                        iterations,
                        hms(elapsed),
                        hms(remaining)
                    ))
                else:
                    print('Iteration %4d/%4d' % (i + 1, iterations))
                last_step = (i == iterations - 1)
                if last_step or (print_iterations and i % print_iterations == 0):
                    loss_vals = get_loss_vals(loss_store)
                else:
                    loss_vals = None

                if (checkpoint_iterations and i % checkpoint_iterations == 0) or last_step:
                    this_loss = loss.eval()
                    print("loss: ", this_loss)
                    print("affine_loss : " ,loss_store['affine_loss'])
                    if this_loss < best_loss:
                        best_loss = this_loss
                        best = image.eval()

                    img_out = vgg.unprocess(best.reshape(shape[1:]), vgg_mean_pixel)

                    if preserve_colors and preserve_colors == True:
                        original_image = np.clip(content, 0, 255)
                        styled_image = np.clip(img_out, 0, 255)

                        # Luminosity transfer steps:
                        # 1. Convert stylized RGB->grayscale accoriding to Rec.601 luma (0.299, 0.587, 0.114)
                        # 2. Convert stylized grayscale into YUV (YCbCr)
                        # 3. Convert original image into YUV (YCbCr)
                        # 4. Recombine (stylizedYUV.Y, originalYUV.U, originalYUV.V)
                        # 5. Convert recombined image from YUV back to RGB

                        # 1
                        styled_grayscale = rgb2gray(styled_image)
                        styled_grayscale_rgb = gray2rgb(styled_grayscale)

                        # 2
                        styled_grayscale_yuv = np.array(Image.fromarray(styled_grayscale_rgb.astype(np.uint8)).convert('YCbCr'))

                        # 3
                        original_yuv = np.array(Image.fromarray(original_image.astype(np.uint8)).convert('YCbCr'))

                        # 4
                        w, h, _ = original_image.shape
                        combined_yuv = np.empty((w, h, 3), dtype=np.uint8)
                        combined_yuv[..., 0] = styled_grayscale_yuv[..., 0]
                        combined_yuv[..., 1] = original_yuv[..., 1]
                        combined_yuv[..., 2] = original_yuv[..., 2]

                        # 5
                        img_out = np.array(Image.fromarray(combined_yuv, 'YCbCr').convert('RGB'))
                else:
                    img_out = None

                yield i+1 if last_step else i, img_out, loss_vals

                iteration_end = time.time()
                iteration_times.append(iteration_end - iteration_start)


def _tensor_size(tensor):
    from operator import mul
    return reduce(mul, (d.value for d in tensor.get_shape()), 1)

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def gray2rgb(gray):
    w, h = gray.shape
    rgb = np.empty((w, h, 3), dtype=np.float32)
    rgb[:, :, 2] = rgb[:, :, 1] = rgb[:, :, 0] = gray
    return rgb

def hms(seconds):
    seconds = int(seconds)
    hours = (seconds // (60 * 60))
    minutes = (seconds // 60) % 60
    seconds = seconds % 60
    if hours > 0:
        return '%d hr %d min' % (hours, minutes)
    elif minutes > 0:
        return '%d min %d sec' % (minutes, seconds)
    else:
        return '%d sec' % seconds
