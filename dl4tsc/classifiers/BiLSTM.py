# LSTM model
# when tuning start with learning rate->mini_batch_size ->
# momentum-> #hidden_units -> # learning_rate_decay -> #layers
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import time

from dl4tsc.utils.utils import save_logs
from dl4tsc.utils.utils import calculate_metrics

class Classifier_BiLSTM:

    def __init__(self, output_directory, input_shape, nb_classes, verbose=True,build=True, lr = 0.001, batch_size=64, epoch = 500):
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

        bilstm1 = keras.layers.Bidirectional(keras.layers.LSTM(8) )(input_layer)
        dropout1 = keras.layers.Dropout(0.1)(bilstm1)

        dense = keras.layers.Dense(8, activation='relu')(dropout1)

        dense2 = keras.layers.Dense(8, activation='relu')(dense)

        output_layer = keras.layers.Dense(units=nb_classes,activation='softmax')(dense2)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.RMSprop(learning_rate=lr),
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

        hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs,
                              verbose=self.verbose, validation_data=(x_val, y_val), callbacks=self.callbacks)

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
