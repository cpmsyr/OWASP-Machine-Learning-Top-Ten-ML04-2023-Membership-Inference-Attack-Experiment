import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.python.keras.layers import GlobalAveragePooling2D
import matplotlib.pyplot as plt



def train_val_generators(train_dir, validation_dir):

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,
        width_shift_range=.2,
        height_shift_range=.2,
        shear_range=.2,
        zoom_range=.2,
        horizontal_flip=False,
        fill_mode='nearest'
    )

    train_generator = train_datagen.flow_from_directory(train_dir,
                                                        batch_size=32,
                                                        class_mode='categorical',
                                                        target_size=(223, 223))

    validation_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=0,
        width_shift_range=0,
        height_shift_range=0,
        shear_range=0,
        zoom_range=0,
        horizontal_flip=False,
        fill_mode='nearest'
    )

    validation_generator = validation_datagen.flow_from_directory(validation_dir,
                                                                  batch_size=32,
                                                                  class_mode='categorical',
                                                                  target_size=(223, 223))

    return train_generator, validation_generator


def create_model():
    pre_trained_model = InceptionV3(input_shape=(223, 223, 3),
                                    include_top=False,
                                    weights='imagenet')
    x = pre_trained_model.output
    x = GlobalAveragePooling2D()(x)
    x = layers.Dense(1024, activation='relu')(x)
    predictions = layers.Dense(500, activation='softmax')(x)

    # this is the model we will train
    my_model = Model(inputs=pre_trained_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in pre_trained_model.layers:
        layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    my_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    return my_model


def display_metrics(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.figure()
    plt.show()


if __name__ == '__main__':
    TRAINING_DIR = 'train\\'
    VALIDATION_DIR = 'valid\\'
    train_generator, validation_generator = train_val_generators(TRAINING_DIR, VALIDATION_DIR)
    model = create_model()
    history = model.fit(train_generator, epochs=5, validation_data=validation_generator)
    model.save('saved\\')
    display_metrics(history)
