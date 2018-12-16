# Photo-Weather-Transfer-Comparing-Neural-Style-and-Deep-Neural-Style-Transfer
he aim of this repo is to tackle the challenge of photorealistic style transfer, specifically in the context of transferring the weather, sky, and ambience elements of one photo into another. Existing methodologies such as Neural Style Transfer by Gatys et al. has been proven to be successful in generating images that merge the content of a content reference image and the style of a style reference image into a single image. However, these generated images tend to work best in an artistic context. The noisy image which the network trains on is often changed in ways reminiscent of a painting, possibly created by the artist who made the style image, rather than a photo. This repo shows the results of introducing to the original algorithm the ideas of semantic image segmentation and the Matting Laplacian. 