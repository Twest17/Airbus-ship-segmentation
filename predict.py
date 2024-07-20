import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
from patchify import patchify, unpatchify
import matplotlib.pyplot as plt

from train_model import get_trained, dice_loss


IMG_SIZE = 768
DATASET_IMG_SIZE = 128


history, unet = get_trained()


def predict_full_images(image, model):
    """
    make prediction on full size image, for example (768,768)
    or another square size height=width % 128 == 0
    1.make patches from image
    2.get predictions on these patches
    3.reconstruct full size prediction from patches
    :param image: image tensor
    :param model: model to use for prediction
    :return: prediction
    """
    image = image.numpy()
    patches = patchify(image, (DATASET_IMG_SIZE, DATASET_IMG_SIZE, 3), step=DATASET_IMG_SIZE)
    patches = patches.reshape(36, DATASET_IMG_SIZE, DATASET_IMG_SIZE, 3)

    preds = binarize(model.predict(patches))

    n_cols = n_rows = int(IMG_SIZE / DATASET_IMG_SIZE)

    preds = preds.reshape(n_rows, n_cols, DATASET_IMG_SIZE, DATASET_IMG_SIZE, 1)
    preds = np.squeeze(preds)

    reconstructed_pred = unpatchify(preds, (IMG_SIZE, IMG_SIZE))
    reconstructed_pred = reconstructed_pred[..., np.newaxis]

    return reconstructed_pred


def binarize(preds):
    return preds > 0.5


def display(display_list):
    """
    show image, [true mask,] predicted mask
    :param display_list: list of images as arrays
    :return: None
    """
    plt.figure(figsize=(10, 10))

    if len(display_list) == 3:
        title = ['Input Image', 'True Mask', 'Predicted Mask']
    else:
        title = ['Input Image', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()


def predict_on_samples(dataset, model=unet, full_image=True, num=1):
    """
    predict on train images, first one from every batch
    :param dataset: data to predict on, tf.data.Dataset
    :param model: model to use for prediction
    :param full_image: whether to predict on full size images or patches, bool
    :param num: number of batches to predict on
    :return: list of tuples [(image, prediction)]
    """
    tuples_to_display = []

    for items in dataset.take(num):
        if len(items) == 2:
            image = items[0]
            mask = items[1]
        else:
            image = items

        if full_image:
            pred = predict_full_images(image, model=model)
        else:
            image_batch = image[np.newaxis, ...]
            pred = binarize(model.predict(image_batch))[0]

        if len(items) == 2:
            tuples_to_display.append((image, mask, pred))
        else:
            tuples_to_display.append((image, pred))

    return tuples_to_display


def encode_for_submission(img):
    """
    :param img: numpy array, 1 - mask, 0 - background
    :return: run length as string formatted
    """
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def get_true_accuracy(dataset):
    """
    calculate accuracy on object pixels only(not background)
    uses dice loss
    :param dataset: tf.data.Dataset or any iterable
    :return: accuracy, float
    """
    accuracy = 0
    n = 1
    
    for x, y in dataset:
        x = x[np.newaxis, ...]
        pred = unet.predict(x, verbose=0).astype(np.float32)
        y = tf.cast(y, tf.dtypes.float32)
        accuracy += -float(dice_loss(y, pred))+1
        n += 1
    accuracy = accuracy / n
    
    return accuracy
