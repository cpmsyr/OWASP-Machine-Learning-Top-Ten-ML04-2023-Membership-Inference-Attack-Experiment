import tensorflow_privacy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.python.keras.layers import GlobalAveragePooling2D
import matplotlib.pyplot as plt
import argparse
import tensorflow as tf
import tensorflow_privacy
from tensorflow_privacy.privacy.optimizers import dp_optimizer_keras


def train_val_generators(train_dir, validation_dir):

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=15,
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
                                                        target_size=(225, 225))

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
                                                                  target_size=(225, 225))

    return train_generator, validation_generator

def predict_data_gen(predict_dir):
    predict_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=0,
        width_shift_range=0,
        height_shift_range=0,
        shear_range=0,
        zoom_range=0,
        horizontal_flip=False,
        fill_mode='nearest'
    )

    predict_generator = predict_datagen.flow_from_directory(predict_dir,
                                                        batch_size=32,
                                                        class_mode='categorical',
                                                        target_size=(225, 225))
    return predict_data_gen()


def create_model():

    pre_trained_model = InceptionV3(input_shape=(225, 225, 3),
                                    include_top=False,
                                    weights='imagenet')
    x = pre_trained_model.output
    x = GlobalAveragePooling2D()(x)
    # x = layers.Dense(1024, activation='relu')(x)
    predictions = layers.Dense(500, activation='softmax')(x)

    # this is the model we will train
    my_model = Model(inputs=pre_trained_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in pre_trained_model.layers:
        layer.trainable = False

    return my_model


def model_load_predict():
    model_1 = tf.keras.models.load_model('saved\\saved_model_1.pb', custom_objects=None, compile=True, safe_mode=True)
    model_2 = tf.keras.models.load_model('saved\\saved_model_2.pb', custom_objects=None, compile=True, safe_mode=True)
    PREDICT_DIR = 'images_to_predict\\'
    images = predict_data_gen(PREDICT_DIR)
    prediction_1 = model_1.predict(images)
    prediction_2 = model_2.predict(images)
    print('Predictions from model 1 are:', prediction_1)
    print('Predictions from model 2 are:', prediction_2)


def display_metrics(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'g', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'y', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.figure()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Set differential privacy for use in the model or run predict mode')
    parser.add_argument('-dp', type=str, help='T to turn on differential privacy. F to train without it')
    parser.add_argument('-pred', type=str, help='T to turn on predict only. F to not predict')
    args = parser.parse_args()
    dp_flag = args.dp
    pred_flag = args.pred

    if pred_flag =='T':
        model_load_predict()

    elif pred_flag == 'F':

        TRAINING_DIR = 'train\\'
        VALIDATION_DIR = 'valid\\'
        train_generator, validation_generator = train_val_generators(TRAINING_DIR, VALIDATION_DIR)
        model = create_model()

        if dp_flag == 'T':
             optimizer = tensorflow_privacy.DPKerasSGDOptimizer(
            l2_norm_clip=.5,
            noise_multiplier=.5,
            num_microbatches=1,
            learning_rate=.01)
             model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        elif dp_flag == 'F':
        # compile the model (should be done *after* setting layers to non-trainable)
            model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
            model.summary()


if pred_flag != 'T':
    history = model.fit(train_generator, epochs=12, validation_data=validation_generator)
    model.save('saved\\')
    display_metrics(history)
