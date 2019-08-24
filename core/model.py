import tensorflow as tf


def build_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=96, kernel_size=11, strides=4, input_shape=input_shape),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),

        tf.keras.layers.Conv2D(filters=256, kernal=5, padding=2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),

        tf.keras.layers.Conv2D(filters=384, kernel_size=3, padding=1),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(filters=384, kernel_size=3, padding=1),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding=1),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dense(units=2048),
        tf.keras.layers.Dense(units=2048)

    ])

    return model