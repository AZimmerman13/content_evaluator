import numpy as np
import pandas as pd
from time import time

from sklearn.model_selection import train_test_split
from skimage import io, color, filters, restoration, feature
from skimage.io import imread
from skimage.transform import resize, rotate

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, array_to_img, img_to_array
import os

def load_data(filepath, image_size, saveData=False, outpath=None, sample_size=-1):
    df = pd.read_json(filepath, lines=True)
    img_lst =[]
    ts = time()
    if sample_size>0:
        print('Taking Sample of Size ', sample_size)
        data = df.sample(sample_size, random_state=13).reset_index(drop=True)
    else:
        print('Taking All Train Data')
        data = df
    # for i in data.img:
    #     img = imread('data/{}'.format(i))
    #     img = resize(img, image_size)
    #     img_lst.append(img)
    #     print(i)
    data_size = data.shape[0]
    img_names = data.img.tolist()
    img_lst = [resize(imread(os.path.join('data/',fname)), image_size) for fname in img_names]

    X = np.array(img_lst).reshape(data_size, image_size[0], image_size[1], image_size[2])
    te = time()
    print('Image processing took {} seconds.'.format(te-ts))

    try: 
        y = data.label.values
    except AttributeError:
        y = np.ndarray([0,0,0])
    if saveData:
        if len(y)!=0:
            np.save(outpath+'_cnn_target', y.reshape(-1,1))
            np.save(outpath+'_cnn_feature',X)
        else:
            np.save(outpath+'_cnn_feature', X)
    return X, y 

def define_model(input_shape):
    model = Sequential()
    model.add(Conv2D(32, (3,3), padding='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model

def compile_model(model, optimizer, metrics):
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=metrics)
    return model

def train_model(model, data_aug, batch_size, epochs,
                X_train, X_test, y_train, y_test):
    if not data_aug:
        print('No data augmentation.')
        model.fit(X_train, y_train, 
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(X_test, y_test),
                  shuffle=True)
    else:
        print('Using real-time data augmentation.')
        datagen = ImageDataGenerator(rotation_range=40,
                                     width_shift_range=0.1,
                                     height_shift_range=0.1,
                                     shear_range=0.2,
                                     horizontal_flip=True,
                                     vertical_flip=False, 
                                     fill_model='nearest')
        datagen.fit(X_train)
        model.fit_generator(datagen.flow(X_train,y_train,batch_size=batch_size),
                                         epochs=epochs,
                                         validation_data=(X_test, y_test),
                                         workers=-1,
                                         steps_per_epoch=len(X_train) // batch_size, 
                                         use_multiprocessing=True)
    return model

def save_model(model, save_dir, model_name):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)

def evaluate_model(model, X_test, y_test):
    scores = model.evaluate(X_test, y_test, verbose=1)
    print('Test loss', scores[0])
    print('Test accuracy:', scores[1])


if __name__ == "__main__":
    
    process_data = False
    cnn_model = True

    image_size = (32,32,3)
    sample_size = -1
    batch_size = 32
    epochs = 3
    data_aug = True

    if process_data:
        X_train, y_train = load_data('data/train.jsonl', image_size=image_size, saveData=True, outpath='data/train', sample_size=sample_size)
        X_val, y_val = load_data('data/dev.jsonl', image_size=image_size, saveData=True, outpath='data/val', sample_size=sample_size)
        X_test, y_test = load_data('data/test.jsonl', image_size=image_size, saveData=True, outpath='data/test', sample_size=sample_size)

    if cnn_model:
        metrics = ['accuracy']
        save_dir = os.path.join(os.getcwd(), 'models')
        model_name = 'keras_cnn.model.h5'
        gd_optimizer = SGD(lr=1e-4, momentum=0.9)

        X = np.load('data/train_cnn_feature.npy')
        y = np.load('data/train_cnn_target.npy')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
        input_shape = X_train.shape[1:]

        model = define_model(input_shape)

        model = compile_model(model, gd_optimizer, metrics)

        model = train_test_split(model, data_aug, batch_size, epochs,
                                X_train, X_test, y_train, y_test)
        
        save_model(model, save_dir, model_name)

        evaluate_model(model, X_test, y_test)
