import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import tensorflow.keras.layers as tkl
import tensorflow.keras.backend as K

from prepare_data import get_datasets
import pickle


IMG_SIZE = 768
DATASET_IMG_SIZE = 128
checkpoint_folder = 'checkpoints'
checkpoint_path = os.path.join(checkpoint_folder, 'new_patches_e-4.keras')
history_path = os.path.join(checkpoint_folder, 'new_patches_history_e4.pkl')
prev_checkpoint = os.path.join(checkpoint_folder, 'patches_e-4_15epochs.keras')
prev_history = os.path.join(checkpoint_folder, 'unet_dice_patches_history_e4.pkl')

BATCH_SIZE = 128
BUFFER_SIZE = 4*BATCH_SIZE
LR = 1e-4
EPOCHS = 1


def conv_block(inputs=None, n_filters=32, dropout_prob=0.0, max_pooling=True):
    """
    Convolutional downsampling block
    """
    conv = tkl.Conv2D(n_filters,
                      kernel_size=3,
                      activation='relu',
                      padding='same',
                      kernel_initializer='he_normal')(inputs)
    conv = tkl.Conv2D(n_filters,
                      kernel_size=3,
                      activation='relu',
                      padding='same',
                      kernel_initializer='he_normal')(conv)
    if dropout_prob > 0:
        conv = tkl.Dropout(dropout_prob)(conv)
    if max_pooling:
        next_layer = tkl.MaxPooling2D(2)(conv)
    else:
        next_layer = conv
    skip_connection = conv

    return next_layer, skip_connection


def upsampling_block(expansive_input, contractive_input, n_filters=32):
    """
    Convolutional upsampling block
    """
    up = tkl.Conv2DTranspose(
                 n_filters,
                 kernel_size=3,
                 strides=2,
                 padding='same')(expansive_input)
    merge = tkl.concatenate([up, contractive_input], axis=3)
    conv = tkl.Conv2D(n_filters,
                 kernel_size=3,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(merge)
    conv = tkl.Conv2D(n_filters,
                 kernel_size=3,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(conv)

    return conv


def unet_model(input_size=(DATASET_IMG_SIZE, DATASET_IMG_SIZE, 3), n_filters=32, n_classes=1):
    """

    :param input_size: image size, int tuple (height, width, n_channels)
    :param n_filters: int
    :param n_classes: int
    :return: U-net model, tf.keras.Model
    """
    inputs = tkl.Input(input_size)
    lam = tkl.Lambda(lambda x: x / 255.)(inputs)

    cblock1 = conv_block(lam, n_filters)
    cblock2 = conv_block(cblock1[0], n_filters * 2)
    cblock3 = conv_block(cblock2[0], n_filters * 4)
    cblock4 = conv_block(cblock3[0], n_filters * 8, dropout_prob=0.3)
    cblock5 = conv_block(cblock4[0], n_filters * 16, dropout_prob=0.3, max_pooling=False)

    ublock6 = upsampling_block(cblock5[0], cblock4[1], n_filters * 8)
    ublock7 = upsampling_block(ublock6, cblock3[1], n_filters * 4)
    ublock8 = upsampling_block(ublock7, cblock2[1], n_filters * 2)
    ublock9 = upsampling_block(ublock8, cblock1[1], n_filters)

    conv9 = tkl.Conv2D(n_filters,
                       kernel_size=3,
                       activation='relu',
                       padding='same',
                       kernel_initializer='he_normal')(ublock9)
    conv10 = tkl.Conv2D(n_classes, kernel_size=1, padding='same', activation='sigmoid')(conv9)

    model = tf.keras.Model(inputs=inputs, outputs=conv10)

    return model


def dice_loss(y_true, y_pred, smooth=1):
    """
    dice loss function suitable for unbalanced data
    :param y_true:
    :param y_pred:
    :param smooth: make penalty for missed pixels smaller, int
    :return: loss value, int
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    dice = 1 - dice

    return dice


def train():
    """
    main train activity. there are defined callbacks for model.
    after training saves history and model weights
    :return: history and trained unet model, (dict,tf.keras.Model)
    """
    processed_image_ds, valid_processed_image_ds = get_datasets(full_image=False)
    train_dataset = processed_image_ds.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    valid_dataset = valid_processed_image_ds.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)

    # unet_dice_e-3: e-3, smooth 1, epoch 5, batch 128, lr_plato
    # unet_dice_e-4: e-4, smooth 1, epoch 5, batch 128, lr_plato

    earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=2)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)
    lr_plato = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=1, factor=0.1, mode='min')

    unet = unet_model((DATASET_IMG_SIZE, DATASET_IMG_SIZE, 3))
    unet.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
                 loss=dice_loss,
                 metrics=['accuracy'])

    history = unet.fit(train_dataset, epochs=EPOCHS, validation_data=valid_dataset,
                       callbacks=[checkpoint, earlystop, lr_plato])

    with open(history_path, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    print(f'saves model weights in {checkpoint_path}')
    print(f'saves training history in {history_path}')
    return history, unet


def get_trained():
    """
    get U-net model with already trained parameters and history of this training
    :return: history and trained unet model, (dict,tf.keras.Model)
    """
    unet = unet_model((DATASET_IMG_SIZE, DATASET_IMG_SIZE, 3))
    history = {}

    if os.path.exists(prev_checkpoint):
        unet.load_weights(prev_checkpoint)
    else:
        print('no trained weights available')
    if os.path.exists(prev_history):
        with open(prev_history, "rb") as file_pi:
            history = pickle.load(file_pi)
    else:
        print('no training history available')

    unet.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
                 loss=dice_loss,
                 metrics=['accuracy'])

    return history, unet

#--- TIME CONSUMING OPERATION! ---
# it takes about 30-40 minutes(P100 Tesla) to train model on patches for 5 epochs
# if __name__ == '__main__':
#     train()
