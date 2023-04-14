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
from keras.utils import custom_object_scope
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

def squeeze_excite_block(input):
    ''' Create a squeeze-excite block
    Args:
        input: input tensor
        filters: number of output filters
        k: width factor

    Returns: a keras tensor
    '''
    filters = input.shape[-1] # channel_axis = -1 for TF

    se = keras.layers.GlobalAveragePooling1D()(input)
    se = keras.layers.Reshape((1, filters))(se)
    se = keras.layers.Dense(filters // 16,  activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = keras.layers.Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    se = keras.layers.multiply([input, se])
    return se

class attention(Layer):
    def __init__(self,name="attention",**kwargs):
        super(attention,self).__init__(**kwargs)
 
    def build(self,input_shape):
        self.W=self.add_weight(name='attention_weight', shape=(input_shape[-1],1), 
                               initializer='random_normal', trainable=True)
        self.b=self.add_weight(name='attention_bias', shape=(input_shape[1],1), 
                               initializer='zeros', trainable=True)        
        super(attention, self).build(input_shape)
 
    def call(self,x):
        # Alignment scores. Pass them through tanh function
        e = K.tanh(K.dot(x,self.W)+self.b)
        # Remove dimension of size 1
        e = K.squeeze(e, axis=-1)   
        # Compute the weights
        alpha = K.softmax(e)
        # Reshape to tensorFlow format
        alpha = K.expand_dims(alpha, axis=-1)
        # Compute the context vector
        context = x * alpha
        context = K.sum(context, axis=1)
        return context

class Classifier_MALSTMFCN:

    def __init__(self, output_directory, input_shape, nb_classes, verbose=True,build=True, lr = 0.001, batch_size=128, epoch = 100, units= 128):
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
            self.units=units

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
        x = attention()(x)
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
        
#         config = model.get_config()
        custom_objects = {"attention": attention}
#         with keras.utils.custom_object_scope(custom_objects):
#             model = keras.Model.from_config(config)
        model = keras.models.load_model(self.output_directory+'best_model.hdf5', custom_objects=custom_objects)
#         model = keras.models.load_model(self.output_directory + 'best_model.hdf5')

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
