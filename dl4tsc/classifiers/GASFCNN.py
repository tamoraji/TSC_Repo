# LSTM model
# when tuning start with learning rate->mini_batch_size ->
# momentum-> #hidden_units -> # learning_rate_decay -> #layers
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import time
import tensorflow.keras.backend as K
from keras.layers import Layer

from dl4tsc.utils.utils import save_logs
from dl4tsc.utils.utils import calculate_metrics
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report



class Classifier_GASFCNN:

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

        # Convolutional layers
        conv1 = keras.layers.Conv2D(64, (3, 3), activation='relu')(input_layer)
        pooling1 = keras.layers.MaxPooling2D((2, 2))(conv1)

        conv2 = keras.layers.Conv2D(64, (3, 3), activation='relu')(pooling1)
        pooling2 = keras.layers.MaxPooling2D((2, 2))(conv2)

        # Flatten the output
        flatten = keras.layers.Flatten()(pooling2)

        # Fully connected layers
        dense1 = keras.layers.Dense(128, activation='relu')(flatten)
        dense2 = keras.layers.Dense(64, activation='relu')(dense1)


        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(dense2)

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

        # calculate the evaluation metrics
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        confusion = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred, zero_division=1)

        return accuracy, f1, confusion, report


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
