# MA_LSTM_FCN model
# when tuning start with learning rate->mini_batch_size ->
# momentum-> #hidden_units -> # learning_rate_decay -> #layers
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import time
import tensorflow.keras.backend as K
from keras.layers import Layer
from dl4tsc.utils.layer_utils import AttentionLSTM

from dl4tsc.utils.utils import save_logs
from dl4tsc.utils.utils import calculate_metrics

def squeeze_excite_block(input):
    ''' Create a squeeze-excite block
    Args:
        input: input tensor
        filters: number of output filters
        k: width factor

    Returns: a keras tensor
    '''
    filters = input._keras_shape[-1] # channel_axis = -1 for TF

    se = keras.layers.GlobalAveragePooling1D()(input)
    se = keras.layers.Reshape((1, filters))(se)
    se = keras.layers.Dense(filters // 16,  activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = keras.layers.Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    se = keras.layers.multiply([input, se])
    return se

class Classifier_MALSTMFCN:

    def __init__(self, output_directory, input_shape, nb_classes, verbose=True,build=True, lr = 0.001, batch_size=128, epoch = 100):
        self.output_directory = output_directory

        if build == True:
            self.model = self.build_model(input_shape, nb_classes,lr)
            if (verbose == True):
                self.model.summary()
            self.verbose = verbose
            self.model.save_weights(self.output_directory + 'model_init.hdf5')
            self.lr = lr
            self.batch_size = batch_size
            self.epoch = epoch

        return

    def build_model(self, input_shape, nb_classes,lr):

        input_layer = keras.layers.Input(input_shape)

        ''' sabsample timesteps to prevent OOM due to Attention LSTM '''
        stride = 2

        x = keras.layers.Permute((2, 1))(input_layer)
        x = keras.layers.Conv1D(input_shape[-1] // stride, 8, strides=stride, padding='same', activation='relu', use_bias=False,
               kernel_initializer='he_uniform')(x) # (None, variables / stride, timesteps)
        x = keras.layers.Permute((2, 1))(x)

        x = keras.layers.Masking()(x)
        x = AttentionLSTM(units=128)(x)
        x = keras.layers.Dropout(0.8)(x)

        y = keras.layers.Permute((2, 1))(input_layer)
        y = keras.layers.Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
        y = keras.layers.BatchNormalization()(y)
        y = keras.layers.Activation('relu')(y)
        y = squeeze_excite_block(y)

        y = keras.layers.Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
        y = keras.layers.BatchNormalization()(y)
        y = keras.layers.Activation('relu')(y)
        y = squeeze_excite_block(y)

        y = keras.layers.Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
        y = keras.layers.BatchNormalization()(y)
        y = keras.layers.Activation('relu')(y)

        y = keras.layers.GlobalAveragePooling1D()(y)

        x = keras.layers.concatenate([x, y])


        output_layer = keras.layers.Dense(units=nb_classes,activation='softmax')(x)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=lr),
                      metrics=['accuracy'])

        file_path = self.output_directory + 'best_model.hdf5'

        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss',
                                                           save_best_only=True)

        self.callbacks = [model_checkpoint]

        return model

    def fit(self, x_train, y_train, x_val, y_val, y_true):
        if not tf.test.is_gpu_available:
            print('error')
            exit()

        # x_val and y_val are only used to monitor the test loss and NOT for training
        mini_batch_size = self.batch_size
        nb_epochs = self.epoch

        start_time = time.time()

        #
        hist = self.model.fit(x_train, y_train, epochs=nb_epochs,
                              verbose=self.verbose, batch_size=mini_batch_size,
                              validation_data=(x_val, y_val), callbacks=self.callbacks)

        duration = time.time() - start_time

        self.model.save(self.output_directory+'last_model.hdf5')

        model = keras.models.load_model(self.output_directory + 'best_model.hdf5')

        y_pred = model.predict(x_val)

        # convert the predicted from binary to integer
        y_pred = np.argmax(y_pred, axis=1)

        save_logs(self.output_directory, hist, y_pred, y_true, duration,lr=False)

        keras.backend.clear_session()

    def predict(self, x_test,y_true,x_train,y_train,y_test,return_df_metrics = True):
        model_path = self.output_directory + 'best_model.hdf5'
        model = keras.models.load_model(model_path)
        y_pred = model.predict(x_test)
        if return_df_metrics:
            y_pred = np.argmax(y_pred, axis=1)
            df_metrics = calculate_metrics(y_true, y_pred, 0.0)
            return df_metrics
        else:
            return y_pred
