#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import keras
from keras import Model
from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Dropout, Input, Lambda
from keras import backend as K
from keras.callbacks import TensorBoard
import os

def get_model_paths(mark):
    model_dir = "model"
    
    struct_filename = "model_{}.json".format(mark)
    struct_filepath = os.path.join(model_dir, struct_filename)
    
    weights_filename = "model_{}.hdh5".format(mark)
    weights_filepath = os.path.join(model_dir, weights_filename)
    
    return struct_filepath, weights_filepath
    

class TimeMark:
    def __init__(self):
        self._mark = None
    
    def update(self):
        from time import strftime, localtime
        self._mark = strftime("%Y-%m-%d_%H-%M-%S", localtime())
        return self._mark
    
    def get(self):
        return self._mark        
    
    def __call__(self):
        return self._mark

    
try:
    last_train_start
except:
    last_train_start = TimeMark()  


def euclidean_distance(vects):
    a, b = vects
    sum_square = K.sum(K.square(a - b), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):    
    margin = 1.
    pred_sq = K.square(y_pred)
    margin_sq = K.square(K.maximum(margin - y_pred, 0))
    return K.mean((1 - y_true) * pred_sq + y_true * margin_sq)


def create_base_network_02(input_shape):    
    from keras.applications.vgg19 import VGG19
    vgg_model = VGG19(
        include_top=False,
        input_shape=input_shape
    )
    vgg_output = vgg_model.output
    x = Flatten()(vgg_output)
    x = Dense(units=1024, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(units=128, activation="relu")(x)
    
    model = keras.models.Model(
        inputs=vgg_model.input,
        outputs=x
    )
    
    return model


def create_base_network_01(input_shape):    
    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=(5, 5), activation="tanh", input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), padding="valid"))
    model.add(Conv2D(filters=64, kernel_size=(5, 5), activation="tanh"))
    model.add(MaxPooling2D(pool_size=(2, 2), padding="valid"))
    model.add(Conv2D(filters=128, kernel_size=(5, 5), activation="tanh"))
    model.add(MaxPooling2D(pool_size=(2, 2), padding="valid"))
    model.add(Flatten())
    model.add(Dropout(0.1))
    model.add(Dense(units=1024, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(units=128, activation="relu"))
    
    return model


def create_base_network_04(input_shape):    
    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=(5, 5), activation="tanh", input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), padding="valid"))
    model.add(Conv2D(filters=64, kernel_size=(5, 5), activation="tanh"))
    model.add(MaxPooling2D(pool_size=(2, 2), padding="valid"))
    model.add(Conv2D(filters=128, kernel_size=(5, 5), activation="tanh"))
    model.add(MaxPooling2D(pool_size=(2, 2), padding="valid"))
    model.add(Flatten())
    model.add(Dropout(0.1))
    model.add(Dense(units=128, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(units=32, activation="relu"))
    
    return model


def create_base_network_03(input_shape):    
    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=(5, 5), activation="tanh", input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), padding="valid"))
    model.add(Conv2D(filters=64, kernel_size=(5, 5), activation="tanh"))
    model.add(MaxPooling2D(pool_size=(2, 2), padding="valid"))
    model.add(Conv2D(filters=128, kernel_size=(5, 5), activation="tanh"))
    model.add(MaxPooling2D(pool_size=(2, 2), padding="valid"))
    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(units=1024, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(units=256, activation="relu"))
    model.add(Dropout(0.1))
    model.add(Dense(units=32, activation="relu"))
    
    return model


def create_base_network(input_shape):
    return create_base_network_01(input_shape)


def compute_accuracy(y_true, y_pred):
    '''
    Compute classification accuracy
    with a fixed threshold on distances
    '''
    
    pred = y_pred.ravel() > 0.5
    return np.mean(pred == y_true)


def tf_accuracy(y_true, y_pred):
    '''
    Compute classification accuracy
    with a fixed threshold on distances
    with TensorFlow backend
    '''
    
    return K.mean(K.equal(y_true, K.cast(y_pred > 0.5, y_true.dtype)))


def build_model(input_shape):
    ## network definition
    base_network = create_base_network(input_shape)
    
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)
    
    ## Use 2 instances of base network,
    ## but it would be trained as single one
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)
    
    distance = Lambda(
        euclidean_distance,
        output_shape=eucl_dist_output_shape
    )([processed_a, processed_b])
    
    model = Model([input_a, input_b], distance)
    
    return model


def compile_model(model):    
    ## Test optimizers
    optimizer = keras.optimizers.Adagrad() ## bad
    optimizer = keras.optimizers.RMSprop() ## bad
    optimizer = keras.optimizers.SGD() ## norm
    model.compile(
        loss=contrastive_loss,
        optimizer=optimizer,
        metrics=[tf_accuracy]
    )
    
    return model

def fit_this_feet(model, data_generator, epoch_count=20, thread_count=3):
    assert epoch_count > 0
    
    start_time = last_train_start.update()
    print("train start time: {}".format(start_time))
    
    tensorboard = TensorBoard(log_dir="logs/market-{}".format(start_time))
    
    model.fit_generator(
        generator=data_generator,
        use_multiprocessing=True,
        workers=thread_count,
        epochs=epoch_count,
        verbose=True,
        callbacks=[tensorboard]
    )
    
    
def save_model(model, mark=None):
    
    if mark is None:
        from time import strftime, gmtime, localtime
        slice_time = strftime("%Y-%m-%d_%H-%M-%S", localtime())
        mark = slice_time
        
    struct_filepath, weights_filepath = get_model_paths(mark)

    ## serialize model to json
    with open(struct_filepath, "w") as json_file:
        model_json = model.to_json()
        json_file.write(model_json)
        print("Model struct has been stored on {}".format(struct_filepath))

    ## serialize weights to HDF5
    model.save_weights(weights_filepath)
    print("Model weights has been stored on {}".format(weights_filepath))
    
    
def load_model(mark):
    from keras.models import model_from_json
    
    struct_filepath, weights_filepath = get_model_paths(mark)
    
    ## Loads model struct
    json_file = open(struct_filepath, 'r')
    model_json = json_file.read()
    json_file.close()
    
    loaded_model = model_from_json(model_json)
    
    ## Loads weights into the model
    loaded_model.load_weights(weights_filepath)
    
    return loaded_model


def open_base_model(train_model_mark):
    train_model = load_model(train_model_mark)
    #train_model.summary()
    
    train_model.layers.pop(3)
    train_model.layers.pop(0)
    seq = train_model.layers[-1]    
    #seq.summary()
    
    input_single = train_model.inputs[0]
    
    outputs = seq(input_single)
    
    base_model = keras.models.Model(
        input_single,
        outputs
    )    
    #base_model.summary()
    
    return base_model