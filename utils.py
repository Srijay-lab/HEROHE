import numpy as np
from imgaug import augmenters as iaa
import random

def make_batches_from_indices_list(iterable, n):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def split_list(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference

def augment_image(img,k):
    seq = iaa.Sequential([
        iaa.Affine(rotate=(-180, 180)),
        #iaa.AdditiveGaussianNoise(scale=(10, 60)),
        iaa.Crop(percent=(0, 0.2))
    ], random_order=True)
    augmented_images = []
    for i in range(k):
        image_aug = seq.augment_image(img)
        augmented_images.append(image_aug)
    return augmented_images

def one_label_smoothing(label):
    #alpha = np.random.uniform(0,1)
    alpha = 0.1
    smooth = (1-alpha)*label + (alpha/2)
    return smooth

def shuffle_list_pair(a,b,c,seed):
    z = list(zip(a, b, c))
    random.Random(seed).shuffle(z)
    a, b, c = zip(*z)
    return a,b,c