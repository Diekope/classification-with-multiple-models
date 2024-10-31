import tensorflow as tf
# -------------------------------------------
# Création du modèle de Mr Mahmood - APFA-Net
# ------------------------------------------------------------
# Il faut l'adapter à pytorch pour l'utiliser avec les autres
# ------------------------------------------------------------

def proposed_model(NUM_CLASSES):
    r,c = 224, 224
    inputs = tf.keras.Input(shape=(r, c, 3))

    x1 = conv_block(inputs, 16)
    x2 = conv_block(x1, 32)
    c1 = crop_and_concat(x1, x2)

    pool_1 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))(c1)
    SE_1 = SEBlock(se_ratio=4)(pool_1)

    x3 = conv_block(SE_1, 64)
    x4 = conv_block(x3, 128)
    c2 = crop_and_concat(x3, x4)

    c1_c2 = crop_and_concat(SE_1, c2)
    pool_2 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))(c1_c2)

    x5 = conv_block(pool_2, 128)
    x6 = conv_block(x5, 256)

    c3 = crop_and_concat(x5, x6)
    pool_3 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))(c3)

    SE_2 = SEBlock(se_ratio=4)(pool_3)

    x7 = conv_block(SE_2, 512)
    x8 = conv_block(x7, 1024)
    c4 = crop_and_concat(x7, x8)

    c3_c4 = crop_and_concat(SE_2, c4)

    temp1 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))(pool_2)
    cnn_features = crop_and_concat(temp1, c3_c4)

    cnn_features = tf.keras.layers.AveragePooling2D(pool_size=(4, 4))(cnn_features)
    #cnn_features = SEBlock(se_ratio=4)(cnn_features)

    x = tf.keras.layers.GlobalAveragePooling2D()(cnn_features)

    # x = tf.keras.layers.Dense(units=1024)(x)
    x = tf.keras.layers.Dropout(rate=0.5)(x)
    output = tf.keras.layers.Dense(units=NUM_CLASSES, activation='softmax')(x)

    # # Creating model and compiling
    model = tf.keras.Model(inputs=inputs, outputs=output)

    return model


def conv_block(inputs, num_filters=None):
    x1 = tf.keras.layers.SeparableConv2D(filters=num_filters, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.001))(inputs)
    x2 = tf.keras.layers.SeparableConv2D(filters=num_filters, kernel_size=(5, 5), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.001))(inputs)
    x = tf.keras.layers.Add()([x1, x2])
    x = tf.keras.layers.BatchNormalization()(x)
    return x

def crop_and_concat(x1,x2):
    with tf.name_scope("crop_and_concat"):
        return tf.keras.layers.Concatenate()([x1,x2])


def SEBlock(se_ratio = 8, activation = "relu", data_format = 'channels_last', ki = "he_normal"):
    '''
    se_ratio : ratio for reduce the filter number of first Dense layer(fc layer) in block
    activation : activation function that of first dense layer
    data_format : channel axis is at the first of dimension or the last
    ki : kernel initializer
    '''

    def f(input_x):

        channel_axis = -1 if data_format == 'channels_last' else 1
        input_channels = input_x.shape[channel_axis]

        reduced_channels = input_channels // se_ratio

        #Squeeze operation
        x = tf.keras.layers.GlobalAveragePooling2D()(input_x)
        x = tf.Reshape(1,1,input_channels)(x) if data_format == 'channels_first' else x
        x = tf.keras.layers.Dense(reduced_channels, kernel_initializer= ki)(x)
        x = tf.keras.layers.Activation(activation)(x)

        #Excitation operation
        x = tf.keras.layers.Dense(input_channels, kernel_initializer=ki, activation='sigmoid')(x)
        x = tf.keras.layers.Permute(dims=(3,1,2))(x) if data_format == 'channels_first' else x
        x = tf.keras.layers.multiply([input_x, x])
        return x
    return f
