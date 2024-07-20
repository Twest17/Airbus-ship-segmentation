import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import shutil
from patchify import patchify

import imageio.v2 as imageio
import skimage.draw
from PIL import Image


IMG_SIZE = 768
DATASET_IMG_SIZE = 128
mask_folder = 'train_masks/'
img_folder = 'train_v2/'
test_folder = 'test_v2/'
img_patches = os.path.join('all_patches', 'image_patches')
mask_patches = os.path.join('all_patches', 'mask_patches')
new_img_patches = os.path.join('all_patches', 'clear_img_patches')


def get_points(s):
    """
    decode pixels
    :param s: string of space-delimited pairs of numbers such as: n1 n2 n3 n4
    :return: list of points in pixel coords format [x1, y1, x2, y2, ...]
    """
    nums = list(map(int, s.split(' ')))
    vertices = []
    for i in range(len(nums)):
        if i % 2 == 0:
            vertices.append(nums[i])
        else:
            vertices.append(nums[i] + nums[i - 1] - 1)

    coords = []
    for v in vertices:
        r, c = divmod(v - 1, IMG_SIZE)
        coords.extend((r, c))

    return coords


def create_mask(img, points, output_folder):
    """
    create masks
    :param img: image filename, string
    :param points: coords of object points, list
    :param output_folder: destination to save mask file, string
    :return: path to mask file, string
    """
    mask_path = os.path.join(output_folder, f"{img.replace('.jpg', '.png')}")
    if os.path.exists(mask_path):
        mask = imageio.imread(mask_path)
    else:
        mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    rr, cc = skimage.draw.polygon(points[1::2], points[0::2], mask.shape)
    mask[rr, cc] = 1

    im = Image.fromarray(mask)
    im.save(mask_path)

    return mask_path


def make_patches(img_path, output_folder):
    """
    make patches from image
    :param img_path: path to image file, string
    :param output_folder: destination to save patches from image, string
    :return: None
    """
    img = imageio.imread(img_path)
    img_name, img_ext = img_path.split('/')[-1].split('.')
    if img_ext == 'jpg':
        out_shape = (DATASET_IMG_SIZE, DATASET_IMG_SIZE, 3)
    else:
        out_shape = (DATASET_IMG_SIZE, DATASET_IMG_SIZE)
    patches = patchify(img, out_shape, step=DATASET_IMG_SIZE)
    # patches.shape (6, 6, 128, 128) for mask
    # patches.shape (6, 6, 1, 128, 128, 3) for image
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            single_patch = np.squeeze(patches[i, j, :, :])
            if img_ext == 'jpg' or (img_ext == 'png' and np.sum(single_patch) != 0):
                pass
                imageio.imwrite(os.path.join(output_folder, f'{img_name}_{i}_{j}.{img_ext}'), single_patch)


def process_path(image_path, mask_path):
    """
    read image and mask files. decode them.
    :param image_path: path to image file, string
    :param mask_path: path to mask file, string
    :return: image, mask tensors of dtype=uint8
    """
    img = tf.io.read_file(image_path)
    img = tf.io.decode_jpeg(img, channels=3)
    mask = tf.io.read_file(mask_path)
    mask = tf.io.decode_png(mask, channels=1)

    return img, mask


def resize(image, mask):
    """
    resize image and mask tensors
    :param image: image tensor
    :param mask: mask tensor
    :return: image and mask tensor
    """
    input_image = tf.image.resize(image, (DATASET_IMG_SIZE, DATASET_IMG_SIZE), method='nearest')
    input_mask = tf.image.resize(mask, (DATASET_IMG_SIZE, DATASET_IMG_SIZE), method='nearest')

    return input_image, input_mask


def preprocess_testdata(img_path):
    """
    read and decode image file
    :param img_path: image filename, string
    :return: image tensor of dtype=uint8
    """
    img = tf.io.read_file(img_path)
    img = tf.io.decode_jpeg(img, channels=3)

    return img


def make_dataset(x, y, full_image):
    """
    make tf.Dataset from lists of filenames (x,y), then read, decode, resize
    :param x: list of image filenames
    :param y: list of mask filenames
    :param full_image: not resize images if True, bool
    :return: tf.Dataset
    """
    train_dataset = tf.data.Dataset.from_tensor_slices((x, y))
    image_ds = train_dataset.map(process_path)
    if full_image:
        processed_image_ds = image_ds
    else:
        processed_image_ds = image_ds.map(resize)

    return processed_image_ds


def prepare_datafiles():
    """
    1.create masks
    2.make patches from image and mask files (for mask patches only with objects)
    3.save to new folder image patches with objects only
    :return: None
    """
    df = pd.read_csv('train_ship_segmentations_v2.csv')
    df = df[df.EncodedPixels.notna()]

    df.loc[:, ['Points']] = df.EncodedPixels.apply(get_points)

    if not os.path.exists(mask_folder):
        os.makedirs(mask_folder)
        df['Mask'] = df.apply(lambda row: create_mask(row['ImageId'], row['Points'], mask_folder), axis=1)

    df = df.drop_duplicates('ImageId')
    df.loc[:, ['ImageId']] = df['ImageId'].apply(lambda img_name: os.path.join(img_folder, img_name))

    x = df['ImageId'].values
    y = df['Mask'].values

    if not os.path.exists(img_patches):
        os.makedirs(img_patches)

    if not os.path.exists(mask_patches):
        os.makedirs(mask_patches)

    for img, mask in zip(x, y):
        make_patches(mask, mask_patches)
        make_patches(img, img_patches)

    if not os.path.exists(new_img_patches):
        os.makedirs(new_img_patches)

    for mask_patch in os.listdir(mask_patches):
        name = mask_patch.split('.')[0]
        img_path = os.path.join(img_patches, f'{name}.jpg')
        new_img_path = os.path.join(new_img_patches, f'{name}.jpg')
        shutil.copy2(img_path, new_img_path)


def get_datasets(full_image=True):
    """
    create train and valid datasets
    :param full_image: whether to get images/masks in full size or in patches, bool
    :return: train, valid tf.Datasets
    """
    x = []
    y = []

    if full_image:
        for f in os.listdir(mask_folder):
            x.append(os.path.join(img_folder, f.replace('.png', '.jpg')))
            y.append(os.path.join(mask_folder, f))
    else:
        for f in os.listdir(mask_patches):
            x.append(os.path.join(new_img_patches, f.replace('.png', '.jpg')))
            y.append(os.path.join(mask_patches, f))

    x_train, x_valid, y_train, y_valid = train_test_split(x[:100], y[:100], test_size=0.05, random_state=42)

    processed_image_ds = make_dataset(x_train, y_train, full_image)
    valid_processed_image_ds = make_dataset(x_valid, y_valid, full_image)

    return processed_image_ds, valid_processed_image_ds


def get_test_dataset():
    """
    create test dataset from test image files
    :return: test tf.Dataset
    """
    x = []

    for f in os.listdir(test_folder):
        x.append(os.path.join(test_folder, f))

    test_dataset = tf.data.Dataset.from_tensor_slices(x)
    test_processed_image_ds = test_dataset.map(preprocess_testdata)

    return test_processed_image_ds


#--- TIME CONSUMING OPERATION! ---
# it takes about 30-40 minutes(Ryzen 5 2600) to prepare files for masks and patches
# if __name__ == '__main__':
#     prepare_datafiles()
