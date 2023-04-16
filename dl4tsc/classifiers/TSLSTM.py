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



class attention(Layer):
    def __init__(self,**kwargs):
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

class Classifier_TSLSTM:

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

        # define LSTM network
        lstm1 = keras.layers.LSTM(units=128, return_sequences=True)(input_layer)
        print(lstm1.shape)
#         lstm2 = keras.layers.LSTM(units=128)(lstm1)
#         print(lstm2.shape)
        
        # define attention mechanism
        attention_layer = attention()(lstm1)
        print(attention_layer.shape)

#         attention_reshaped = tf.expand_dims(attention_layer, axis=-1)
#         print(attention_reshaped.shape)

#         result = tf.matmul(attention_reshaped,lstm2 )
#         print(result.shape)

#         print("apply the attention mechanism")
#         context_vector = keras.layers.multiply([lstm2, attention_reshaped], name='context_vector')
#         print(context_vector.shape)
#         context_vector = keras.layers.Lambda(lambda x: keras.backend.sum(x, axis=1), name='context_vector_sum')(context_vector)
#         print(context_vector.shape)
#         context_vector = tf.expand_dims(context_vector, axis=-1)
#         print(context_vector.shape)



        # lstm2 = keras.layers.LSTM(units=128)(lstm1)
        # lstm3 = keras.layers.LSTM(units=128, return_sequences=True)(lstm2)

        # # concatenate skip connection
        # skip_out = keras.layers.Concatenate()([input_layer, lstm3])

        # # define attention mechanism
        # attention_out = keras.layers.Dense(1, activation='tanh')(lstm1)
        # attention_out = keras.layers.Flatten()(attention_out)
        # attention_out = keras.layers.Activation('softmax')(attention_out)
        # attention_out = keras.layers.RepeatVector(128)(attention_out)
        # attention_out = keras.layers.Permute([2, 1])(attention_out)
        # attention_out = keras.layers.Reshape((input_shape[1],))(attention_out)


        # # apply the attention mechanism
        # context_vector = keras.layers.multiply([lstm1, attention_out], name='context_vector')
        # context_vector = keras.layers.Lambda(lambda x: keras.backend.sum(x, axis=1), name='context_vector_sum')(context_vector)


        # define fully-connected layer
        fc_out = keras.layers.Dense(units=128, activation='relu')(attention_layer)

        output_layer = keras.layers.Dense(units=nb_classes,activation='softmax')(fc_out)

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
