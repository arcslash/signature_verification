import tensorflow as tf



def build_model(input_shape = (952, 1360)):
    """
        Model architecture
    """

    # Define the tensors for the two input images
    left_input = tf.keras.layers.Input(input_shape)
    right_input = tf.keras.layers.Input(input_shape)

    # Convolutional Neural Network
    model = tf.layers.Sequential([
        tf.keras.layers.Conv2D(64, (10, 10), activation = 'relu', input_shape = input_shape, kernel_initializer = 'random_uniform', kernel_regularizer = tf.keras.regularizers.l2(2e-4)),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(128, (7, 7), activation='relu', kernel_initializer = 'random_uniform', bias_initializer = 'zeros', kernel_regularizer=tf.keras.regularizers.l2(2e-4)),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(128, (4, 4), activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros', kernel_regularizer=tf.keras.regularizers.l2(2e-4)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(256, (4, 4), activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros', kernel_regularizer=tf.keras.regularizers.l2(2e-4)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4096, activation='sigmoid',kernel_regularizer=l2(1e-3), kernel_initializer='random_uniform', bias_initializer='zeros')

    ])


    # Generate the encodings (feature vectors) for the two images
    encoded_l = model(left_input)
    encoded_r = model(right_input)

    # Add a customized layer to compute the absolute difference between the encodings
    L1_layer = tf.keras.Lambda(lambda tensors: tf.keras.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([encoded_l, encoded_r])

    # Add a dense layer with a sigmoid unit to generate the similarity score
    prediction = tf.keras.layers.Dense(1, activation='sigmoid', bias_initializer='zeros')(L1_distance)

    # Connect the inputs with the outputs
    siamese_net = tf.keras.Model(inputs=[left_input, right_input], outputs=prediction)



    return siamese_net

if __name__ == '__main__':
    build_model()