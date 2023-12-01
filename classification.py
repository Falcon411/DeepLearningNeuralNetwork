import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from keras import Input
from keras.applications import ResNet152V2, Xception, VGG19
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator

from skimage import io
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split

# Подключение видеокарты
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def fix_gpu():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.InteractiveSession(config=config)


fix_gpu()

data_map = []

BASE_LEN = END_IMG_LEN = END_MASK_LEN = 0

for subdir, dirs, files in os.walk("Dataset_MRI"):
    for file in files:
        id_file = subdir.removeprefix("Dataset_MRI/")
        data_map.extend([id_file, os.path.join(subdir, file)])

data_MRI = pd.DataFrame({
    "пациент": data_map[::2],
    "изображение / маска": data_map[1::2]})

df_imgs = data_MRI[~data_MRI['изображение / маска'].str.contains("mask")]
df_masks = data_MRI[data_MRI['изображение / маска'].str.contains("mask")]

index_arr = [x for x in range(len(df_imgs))]
df_imgs = df_imgs.set_index([index_arr])
df_masks = df_masks.set_index([index_arr])

img_lenght = len(df_imgs['изображение / маска'][0])
end_img_lenght = 0

for i in range(img_lenght):
    if df_imgs['изображение / маска'][0].startswith(".", i):
        end_img_lenght = img_lenght - i

base_lenght = img_lenght - end_img_lenght
file_name = df_imgs['изображение / маска'][0][:base_lenght]

for i in range(1, base_lenght):
    if file_name[-i] == "_":
        file_name = file_name[:-i + 1]
        break

base_lenght = len(file_name)
mask_lenght = len(df_imgs['изображение / маска'][0])
end_mask_lenght = 0

for i in range(mask_lenght):
    if df_imgs['изображение / маска'][0].startswith(".", i):
        end_mask_lenght = mask_lenght - i + len("_mask")

imgs = sorted(df_imgs['изображение / маска'], key=lambda x: int(x[base_lenght:-end_img_lenght]))
masks = sorted(df_masks['изображение / маска'], key=lambda x: int(x[base_lenght:-end_mask_lenght]))

print("Path to the Image:", imgs[0], "\nPath to the Mask:", masks[0])

pacient_id = []

for arr_iter in imgs:
    for df_iter in df_imgs["пациент"]:
        if df_iter in arr_iter:
            pacient_id.append(df_iter)
            break

data_MRI = pd.DataFrame({
    "пациент": pacient_id,
    "изображение": imgs,
    "маска": masks})


def pos_neg_diagnosis(mask_path):
    value = np.max(cv2.imread(mask_path))
    if value > 0:
        return 1
    else:
        return 0


data_MRI['заболевание'] = data_MRI['маска'].apply(lambda x: pos_neg_diagnosis(x))

fig, axs = plt.subplots(4, 2, figsize=(10, 15))

count = 0

for x in range(4):
    i = random.randint(0, len(data_MRI))
    axs[count][0].title.set_text("Снимок головного мозга")
    axs[count][0].imshow(cv2.imread(data_MRI['изображение'][i]))
    axs[count][1].title.set_text("Заболевание - " + str(data_MRI['заболевание'][i]))
    axs[count][1].imshow(cv2.imread(data_MRI['маска'][i]))
    count += 1

fig.tight_layout()
fig.savefig('MRI_random_image_mask.png')
plt.close(fig)

iter_i = 0
iter_j = 0

figure, axis = plt.subplots(4, 3, figsize=(15, 15))

for mask in data_MRI['заболевание']:
    if mask == 1:
        imgMRI = io.imread(data_MRI['изображение'][iter_j])
        axis[iter_i][0].title.set_text("Снимок головного мозга")
        axis[iter_i][0].imshow(imgMRI)

        imgMask = io.imread(data_MRI['маска'][iter_j])
        axis[iter_i][1].title.set_text("Заболевание")
        axis[iter_i][1].imshow(imgMask, cmap='gray')

        imgMRI[imgMask == 255] = (0, 128, 255)
        axis[iter_i][2].title.set_text("Снимок головного мозга с заболеванием")
        axis[iter_i][2].imshow(imgMRI)
        iter_i += 1
    iter_j += 1
    if iter_i == 4:
        break

figure.tight_layout()
figure.savefig('MRI_mask_to_image.png')
plt.close(figure)

data_MRI_model = data_MRI.drop(columns=['пациент'])
data_MRI_model['заболевание'] = data_MRI_model['заболевание'].apply(lambda x: str(x))

train, test = train_test_split(data_MRI_model, test_size=0.3)

datagen = ImageDataGenerator(rescale=1. / 255.,
                             rotation_range=180,
                             width_shift_range=0.3,
                             height_shift_range=0.3,
                             horizontal_flip=True,
                             vertical_flip=True,
                             validation_split=0.3)

train_generator = datagen.flow_from_dataframe(train,
                                              directory='./',
                                              x_col='изображение',
                                              y_col='заболевание',
                                              subset='training',
                                              class_mode='categorical',
                                              batch_size=16,
                                              shuffle=True,
                                              target_size=(256, 256))

valid_generator = datagen.flow_from_dataframe(train,
                                              directory='./',
                                              x_col='изображение',
                                              y_col='заболевание',
                                              subset='validation',
                                              class_mode='categorical',
                                              batch_size=16,
                                              shuffle=True,
                                              target_size=(256, 256))

test_datagen = ImageDataGenerator(rescale=1. / 255.,
                                  rotation_range=180,
                                  width_shift_range=0.3,
                                  height_shift_range=0.3,
                                  horizontal_flip=True,
                                  vertical_flip=True)

test_generator = test_datagen.flow_from_dataframe(test,
                                                  directory='./',
                                                  x_col='изображение',
                                                  y_col='заболевание',
                                                  class_mode='categorical',
                                                  batch_size=16,
                                                  shuffle=False,
                                                  target_size=(256, 256))

# Модель ResNet152V2
resnet = ResNet152V2(weights='imagenet', include_top=False, input_tensor=Input(shape=(256, 256, 3)))

for layer in resnet.layers:
    layer.trainable = False

resnet_model = resnet.output
resnet_model = GlobalAveragePooling2D()(resnet_model)
resnet_model = Flatten(name='Flatten')(resnet_model)
resnet_model = Dense(256, activation='relu')(resnet_model)
resnet_model = Dropout(0.3)(resnet_model)
resnet_model = Dense(256, activation='relu')(resnet_model)
resnet_model = Dropout(0.3)(resnet_model)
resnet_model = Dense(2, activation='softmax')(resnet_model)

resnet = Model(resnet.input, resnet_model)
resnet.compile(loss='categorical_crossentropy',
               optimizer='adam',
               metrics=["accuracy"])

resnet.summary()

resnet_earlystopping = EarlyStopping(monitor='val_loss',
                                     mode='min',
                                     verbose=1,
                                     patience=15)

resnet_checkpointer = ModelCheckpoint(filepath="resnet152v2-weights.hdf5",
                                      verbose=1,
                                      save_best_only=True)

resnet_reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                     mode='min',
                                     verbose=1,
                                     patience=10,
                                     min_delta=0.0001,
                                     factor=0.2)

resnet_callbacks = [resnet_checkpointer, resnet_earlystopping, resnet_reduce_lr]

resnet_history = resnet.fit(train_generator,
                            steps_per_epoch=train_generator.n // train_generator.batch_size,
                            epochs=100,
                            validation_data=valid_generator,
                            validation_steps=valid_generator.n // valid_generator.batch_size,
                            callbacks=[resnet_checkpointer, resnet_earlystopping])

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(resnet_history.history['loss'])
plt.plot(resnet_history.history['val_loss'])
plt.title("ResNet152V2 loss")
plt.ylabel("loss")
plt.xlabel("Epochs")
plt.legend(['train', 'val'])

plt.subplot(1, 2, 2)
plt.plot(resnet_history.history['accuracy'])
plt.plot(resnet_history.history['val_accuracy'])
plt.title("ResNet152V2 accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epochs")
plt.legend(['train', 'val'])

plt.savefig('ResNet152V2.png')

resnet_loss, resnet_accuracy = resnet.evaluate(test_generator)
print("ResNet152V2 test loss : {} %".format(resnet_loss * 100))
print("ResNet152V2 test accuracy : {} %".format(resnet_accuracy * 100))

resnet_prediction = resnet.predict(test_generator)

resnet_argmax = np.argmax(resnet_prediction, axis=1)
original = np.asarray(test['заболевание']).astype('int')

resnet_accuracy_score = accuracy_score(original, resnet_argmax)
print("ResNet152V2 test accuracy score : {} %".format(resnet_accuracy_score * 100))

cm = confusion_matrix(original, resnet_argmax)

report = classification_report(original, resnet_argmax, labels=[0, 1])
print("ResNet152V2 confusion matrix", report)

plt.figure(figsize=(5, 5))
sns.heatmap(cm, annot=True)
plt.savefig("ResNet152V2_confusion_matrix.png")

# Модель VGG19
vgg19 = VGG19(weights='imagenet', include_top=False, input_tensor=Input(shape=(256, 256, 3)))

for layer in vgg19.layers:
    layer.trainable = False

vgg19_model = vgg19.output
vgg19_model = GlobalAveragePooling2D()(vgg19_model)
vgg19_model = Flatten(name='Flatten')(vgg19_model)
vgg19_model = Dense(256, activation='relu')(vgg19_model)
vgg19_model = Dropout(0.3)(vgg19_model)
vgg19_model = Dense(256, activation='relu')(vgg19_model)
vgg19_model = Dropout(0.3)(vgg19_model)
vgg19_model = Dense(2, activation='softmax')(vgg19_model)

vgg19 = Model(vgg19.input, vgg19_model)
vgg19.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=["accuracy"])

vgg19.summary()

vgg19_earlystopping = EarlyStopping(monitor='val_loss',
                                    mode='min',
                                    verbose=1,
                                    patience=15)

vgg19_checkpointer = ModelCheckpoint(filepath="resnet152v2-weights.hdf5",
                                     verbose=1,
                                     save_best_only=True)

vgg19_reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                    mode='min',
                                    verbose=1,
                                    patience=10,
                                    min_delta=0.0001,
                                    factor=0.2)

xception_callbacks = [vgg19_checkpointer, vgg19_earlystopping, vgg19_reduce_lr]

vgg19_history = vgg19.fit(train_generator,
                          steps_per_epoch=train_generator.n // train_generator.batch_size,
                          epochs=100,
                          validation_data=valid_generator,
                          validation_steps=valid_generator.n // valid_generator.batch_size,
                          callbacks=[vgg19_checkpointer, vgg19_earlystopping])

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(vgg19_history.history['loss'])
plt.plot(vgg19_history.history['val_loss'])
plt.title("VGG19 loss")
plt.ylabel("loss")
plt.xlabel("Epochs")
plt.legend(['train', 'val'])

plt.subplot(1, 2, 2)
plt.plot(vgg19_history.history['accuracy'])
plt.plot(vgg19_history.history['val_accuracy'])
plt.title("VGG19 accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epochs")
plt.legend(['train', 'val'])

plt.savefig('VGG19.png')

vgg19_loss, vgg19_accuracy = vgg19.evaluate(test_generator)
print("VGG19 test loss : {} %".format(vgg19_loss * 100))
print("VGG19 test accuracy : {} %".format(vgg19_accuracy * 100))

vgg19_prediction = vgg19.predict(test_generator)

vgg19_argmax = np.argmax(vgg19_prediction, axis=1)
vgg19_original = np.asarray(test['заболевание']).astype('int')

vgg19_accuracy_score = accuracy_score(vgg19_original, vgg19_argmax)
print("VGG19 test accuracy score : {} %".format(vgg19_accuracy_score * 100))

vgg19_cm = confusion_matrix(vgg19_original, vgg19_argmax)

vgg19_report = classification_report(vgg19_original, vgg19_argmax, labels=[0, 1])
print("VGG19 confusion matrix", vgg19_report)

plt.figure(figsize=(5, 5))
sns.heatmap(cm, annot=True)
plt.savefig("VGG19_confusion_matrix.png")

# Модель VGG19
xception = Xception(weights='imagenet', include_top=False, input_tensor=Input(shape=(256, 256, 3)))

for layer in xception.layers:
    layer.trainable = False

xception_model = xception.output
xception_model = GlobalAveragePooling2D()(xception_model)
xception_model = Flatten(name='Flatten')(xception_model)
xception_model = Dense(256, activation='relu')(xception_model)
xception_model = Dropout(0.3)(xception_model)
xception_model = Dense(256, activation='relu')(xception_model)
xception_model = Dropout(0.3)(xception_model)
xception_model = Dense(2, activation='softmax')(xception_model)

xception = Model(xception.input, xception_model)
xception.compile(loss='categorical_crossentropy',
                 optimizer='adam',
                 metrics=["accuracy"])

xception.summary()

xception_earlystopping = EarlyStopping(monitor='val_loss',
                                       mode='min',
                                       verbose=1,
                                       patience=15)

xception_checkpointer = ModelCheckpoint(filepath="resnet152v2-weights.hdf5",
                                        verbose=1,
                                        save_best_only=True)

xception_reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                       mode='min',
                                       verbose=1,
                                       patience=10,
                                       min_delta=0.0001,
                                       factor=0.2)

xception_callbacks = [xception_checkpointer, xception_earlystopping, xception_reduce_lr]

xception_history = xception.fit(train_generator,
                                steps_per_epoch=train_generator.n // train_generator.batch_size,
                                epochs=100,
                                validation_data=valid_generator,
                                validation_steps=valid_generator.n // valid_generator.batch_size,
                                callbacks=[xception_checkpointer, xception_earlystopping])

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(xception_history.history['loss'])
plt.plot(xception_history.history['val_loss'])
plt.title("Xception loss")
plt.ylabel("loss")
plt.xlabel("Epochs")
plt.legend(['train', 'val'])

plt.subplot(1, 2, 2)
plt.plot(xception_history.history['accuracy'])
plt.plot(xception_history.history['val_accuracy'])
plt.title("Xception accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epochs")
plt.legend(['train', 'val'])

plt.savefig('Xception.png')

xception_loss, xception_accuracy = xception.evaluate(test_generator)
print("Xception test loss : {} %".format(xception_loss * 100))
print("Xception test accuracy : {} %".format(xception_accuracy * 100))

xception_prediction = xception.predict(test_generator)

xception_argmax = np.argmax(xception_prediction, axis=1)
xception_original = np.asarray(test['заболевание']).astype('int')

xception_accuracy_score = accuracy_score(xception_original, xception_argmax)
print("Xception test accuracy score : {} %".format(xception_accuracy_score * 100))

xception_cm = confusion_matrix(xception_original, xception_argmax)

xception_report = classification_report(xception_original, xception_argmax, labels=[0, 1])
print("Xception confusion matrix", xception_report)

plt.figure(figsize=(5, 5))
sns.heatmap(cm, annot=True)
plt.savefig("Xception_confusion_matrix.png")
