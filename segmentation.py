import glob

import cv2
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Input
from keras.utils import Sequence
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.layers import Activation, BatchNormalization, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate
from keras.models import Model
from segmentation_models import Unet

from skimage import io

from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# Подключение видеокарты
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def fix_gpu():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.InteractiveSession(config=config)


fix_gpu()

EPOCHS = 128
BATCH_SIZE = 16
width = height = 256

train_files = []
mask_files = glob.glob('Dataset_MRI/*/*_mask*')

for i in mask_files:
    train_files.append(i.replace('_mask', ''))

print(train_files[:10])
print(mask_files[:10])

data_MRI = pd.DataFrame(data={"изображение": train_files, 'маска': mask_files})
train, test = train_test_split(data_MRI, test_size=0.2)
train, val = train_test_split(train, test_size=0.3)

def data_generator(data_frame, batch_size, aug_params,
                   image_color_mode="rgb",
                   mask_color_mode="grayscale",
                   image_save_prefix="image",
                   mask_save_prefix="mask",
                   save_to_dir=None,
                   target_size=(height, width),
                   seed=1):
    image_datagen = ImageDataGenerator(**aug_params)
    mask_datagen = ImageDataGenerator(**aug_params)

    image_generator = image_datagen.flow_from_dataframe(
        data_frame,
        x_col="изображение",
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        seed=seed)

    mask_generator = mask_datagen.flow_from_dataframe(
        data_frame,
        x_col="маска",
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=mask_save_prefix,
        seed=seed)

    train_data = zip(image_generator, mask_generator)

    for (img, mask) in train_data:
        img, mask = pos_neg_diagnosis(img, mask)
        yield img, mask


def pos_neg_diagnosis(img, mask):
    img = img / 255
    mask = mask / 255
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0

    return img, mask


epsilon = 1e-5
smooth = 1


def tversky(y_true, y_pred):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)


def focal_tversky(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return K.pow((1 - pt_1), gamma)


def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true, y_pred)


def unet(input_size=(256, 256, 3)):
    inputs = Input(input_size)

    conv1 = Conv2D(64, (3, 3), padding='same')(inputs)
    bn1 = Activation('relu')(conv1)
    conv1 = Conv2D(64, (3, 3), padding='same')(bn1)
    bn1 = BatchNormalization(axis=3)(conv1)
    bn1 = Activation('relu')(bn1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(bn1)

    conv2 = Conv2D(128, (3, 3), padding='same')(pool1)
    bn2 = Activation('relu')(conv2)
    conv2 = Conv2D(128, (3, 3), padding='same')(bn2)
    bn2 = BatchNormalization(axis=3)(conv2)
    bn2 = Activation('relu')(bn2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(bn2)

    conv3 = Conv2D(256, (3, 3), padding='same')(pool2)
    bn3 = Activation('relu')(conv3)
    conv3 = Conv2D(256, (3, 3), padding='same')(bn3)
    bn3 = BatchNormalization(axis=3)(conv3)
    bn3 = Activation('relu')(bn3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(bn3)

    conv4 = Conv2D(512, (3, 3), padding='same')(pool3)
    bn4 = Activation('relu')(conv4)
    conv4 = Conv2D(512, (3, 3), padding='same')(bn4)
    bn4 = BatchNormalization(axis=3)(conv4)
    bn4 = Activation('relu')(bn4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(bn4)

    conv5 = Conv2D(1024, (3, 3), padding='same')(pool4)
    bn5 = Activation('relu')(conv5)
    conv5 = Conv2D(1024, (3, 3), padding='same')(bn5)
    bn5 = BatchNormalization(axis=3)(conv5)
    bn5 = Activation('relu')(bn5)

    up6 = concatenate([Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(bn5), conv4], axis=3)
    conv6 = Conv2D(512, (3, 3), padding='same')(up6)
    bn6 = Activation('relu')(conv6)
    conv6 = Conv2D(512, (3, 3), padding='same')(bn6)
    bn6 = BatchNormalization(axis=3)(conv6)
    bn6 = Activation('relu')(bn6)

    up7 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(bn6), conv3], axis=3)
    conv7 = Conv2D(256, (3, 3), padding='same')(up7)
    bn7 = Activation('relu')(conv7)
    conv7 = Conv2D(256, (3, 3), padding='same')(bn7)
    bn7 = BatchNormalization(axis=3)(conv7)
    bn7 = Activation('relu')(bn7)

    up8 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(bn7), conv2], axis=3)
    conv8 = Conv2D(128, (3, 3), padding='same')(up8)
    bn8 = Activation('relu')(conv8)
    conv8 = Conv2D(128, (3, 3), padding='same')(bn8)
    bn8 = BatchNormalization(axis=3)(conv8)
    bn8 = Activation('relu')(bn8)

    up9 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(bn8), conv1], axis=3)
    conv9 = Conv2D(64, (3, 3), padding='same')(up9)
    bn9 = Activation('relu')(conv9)
    conv9 = Conv2D(64, (3, 3), padding='same')(bn9)
    bn9 = BatchNormalization(axis=3)(conv9)
    bn9 = Activation('relu')(bn9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(bn9)

    return Model(inputs=[inputs], outputs=[conv10])


model = unet()
model.summary()

generator_args = dict(rotation_range=0.5,
                      width_shift_range=0.05,
                      height_shift_range=0.05,
                      shear_range=0.05,
                      zoom_range=0.05,
                      horizontal_flip=True,
                      vertical_flip=True,
                      fill_mode='nearest')

train_generator = data_generator(train, BATCH_SIZE,
                                 generator_args,
                                 target_size=(height, width))

val_generator = data_generator(val, BATCH_SIZE,
                               dict(),
                               target_size=(height, width))

test_generator = data_generator(test, BATCH_SIZE,
                                dict(),
                                target_size=(height, width))

#unet = segmentation_models.Unet('resnet34', input_shape=(height, width, 3), encoder_weights='imagenet')

unet = unet(input_size=(height, width, 3))

unet_earlystopping = EarlyStopping(monitor='val_loss',
                                   mode='min',
                                   verbose=1,
                                   patience=15)

unet_checkpointer = ModelCheckpoint(filepath="unet-weights.hdf5",
                                    verbose=1,
                                    save_best_only=True)

unet_reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                   factor=0.1,
                                   mode='min',
                                   verbose=1,
                                   patience=10,
                                   min_delta=1e-4,
                                   min_lr=1e-6)

unet_callbacks = [unet_checkpointer, unet_earlystopping, unet_reduce_lr]

unet.compile(optimizer='adam', loss=focal_tversky, metrics=[tversky])

unet_history = unet.fit(train_generator,
                        steps_per_epoch=len(train) // BATCH_SIZE,
                        epochs=EPOCHS,
                        validation_data=val_generator,
                        validation_steps=len(val) // BATCH_SIZE,
                        callbacks=unet_callbacks)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(unet_history.history['loss'])
plt.plot(unet_history.history['val_loss'])
plt.title("Unet loss")
plt.ylabel("focal tversky loss")
plt.xlabel("Epochs")
plt.legend(['train', 'val'])

plt.subplot(1, 2, 2)
plt.plot(unet_history.history['tversky'])
plt.plot(unet_history.history['val_tversky'])
plt.title("Unet accuracy")
plt.ylabel("tversky accuracy")
plt.xlabel("Epochs")
plt.legend(['train', 'val'])

plt.savefig('Unet.png')

#unet.load_weights('unet-weights.hdf5')

loss, accuracy = unet.evaluate(test_generator,
                               batch_size=BATCH_SIZE,
                               steps=len(test) // BATCH_SIZE)

print("Test loss: ", loss)
print("Test accuracy: ", accuracy)

for i in range(5):
    index = np.random.randint(1, len(test))
    img = cv2.imread(test['изображение'].iloc[index])
    img = cv2.resize(img, (height, width))
    img = img / 255
    img = img[np.newaxis, :, :, :]
    img_prediction = unet.predict(img)

    plt.figure(figsize=(12, 12))
    plt.subplot(1, 3, 1)
    plt.imshow(np.squeeze(img))
    plt.title('Изображение')
    plt.subplot(1, 3, 2)
    plt.imshow(np.squeeze(cv2.imread(test['маска'].iloc[index])))
    plt.title('Исходная маска')
    plt.subplot(1, 3, 3)
    plt.imshow(np.squeeze(img_prediction) > .5)
    plt.title('Предсказанная маска')
    plt.savefig('prediction' + str(i) + ".png")
    plt.show()



class DataGenerator(Sequence):
    def __init__(self, ids, mask, image_dir='./', batch_size=BATCH_SIZE, img_h=height, img_w=width, shuffle=True):

        self.ids = ids
        self.mask = mask
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.img_h = img_h
        self.img_w = img_w
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Get the number of batches per epoch'

        return int(np.floor(len(self.ids)) / self.batch_size)

    def __getitem__(self, index):
        'Generate a batch of data'

        # generate index of batch_size length
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]

        # get the ImageId corresponding to the indexes created above based on batch size
        list_ids = [self.ids[i] for i in indexes]

        # get the MaskId corresponding to the indexes created above based on batch size
        list_mask = [self.mask[i] for i in indexes]

        # generate data for the X(features) and y(label)
        X, y = self.__data_generation(list_ids, list_mask)

        # returning the data
        return X, y

    def on_epoch_end(self):
        'Used for updating the indices after each epoch, once at the beginning as well as at the end of each epoch'

        # getting the array of indices based on the input dataframe
        self.indexes = np.arange(len(self.ids))

        # if shuffle is true, shuffle the indices
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_ids, list_mask):
        'generate the data corresponding the indexes in a given batch of images'

        # create empty arrays of shape (batch_size,height,width,depth)
        # Depth is 3 for input and depth is taken as 1 for output becasue mask consist only of 1 channel.
        X = np.empty((self.batch_size, self.img_h, self.img_w, 3))
        y = np.empty((self.batch_size, self.img_h, self.img_w, 1))

        # iterate through the dataframe rows, whose size is equal to the batch_size
        for i in range(len(list_ids)):
            # path of the image
            img_path = str(list_ids[i])

            # mask path
            mask_path = str(list_mask[i])

            # reading the original image and the corresponding mask image
            img = io.imread(img_path)
            mask = io.imread(mask_path)

            # resizing and coverting them to array of type float64
            img = cv2.resize(img, (self.img_h, self.img_w))
            img = np.array(img, dtype=np.float64)

            mask = cv2.resize(mask, (self.img_h, self.img_w))
            mask = np.array(mask, dtype=np.float64)

            # standardising
            img -= img.mean()
            img /= img.std()

            mask -= mask.mean()
            mask /= mask.std()

            # Adding image to the empty array
            X[i,] = img

            # expanding the dimnesion of the image from (256,256) to (256,256,1)
            y[i,] = np.expand_dims(mask, axis=2)

        # normalizing y
        y = (y > 0).astype(int)

        return X, y
