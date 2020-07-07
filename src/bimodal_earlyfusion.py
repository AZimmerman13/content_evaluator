# outline of project
# 1. read in the image / pre-processing with feature engineering => converting images to 16x16 grey, unravel to 1x256
# 2. extracting features from texts using nlp => 500 features, 1x500
# 3. horizontal stack / concatenate (1x756)
# 4. training set (~8500), dev set is balanced (~500) 
# 5. model selection - RF, boosted Classifier, NN classifier (MLP), 

import numpy as np
import pandas as pd
from time import time
import nltk
#nltk.download('wordnet')
from sklearn.model_selection import train_test_split
from skimage.io import imread
from skimage.transform import resize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from src.helpers import wordnet_lemmetize_tokenize
from src.pipeline import ImagePipeline

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.metrics import FalseNegatives, FalsePositives, TrueNegatives, TruePositives, Precision, Recall

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import accuracy_score, roc_auc_score

def load_and_featurize_data(image_size, df, img_col):
    out_lst=[]
    for i in df[img_col]:
        img = imread('data/{}'.format(i))
        img = resize(img, image_size)
        out_lst.append(image_to_vector(img))
        # print(i)
    return np.array(out_lst)

def image_to_vector(image):
    """
    Args:
    image: numpy array of shape (length, height, depth)

    Returns:
     v: a vector of shape (length x height x depth, 1)
    """
    length, height, depth = image.shape
    return image.reshape((length * height * depth, 1))

def get_weights(y_train):
    """
        calculate the weights used in fitting the mlp model
    Args:
        y_train: ndarray - 1D
    Returns:
        weight_for_0, weight_for_1 (float): two weights used in mlp model
    """
    counts = int(np.sum(y_train))
    total = len(y_train)
    print("Number of positive samples in training data: {} ({:.2f}% of total)".format(
          counts, 100*float(counts) / total))
    weight_for_0 = 1.0/(total-counts)
    weight_for_1 = 1.0/counts
    return weight_for_0, weight_for_1

def define_mlp_model(n_input):
    """ 
        define the multi-layer-perceptron neural network
    Args:
        n_input(int): number of features
    Returns:
        a defined mlp model, not fitted
    """
    model = Sequential()
    num_neurons = 256

    # hidden layer
    model.add(Dense(units=num_neurons,
                    input_dim=n_input,
                    kernel_initializer='he_uniform',
                    activation = 'relu'))
    model.add(Dense(units=num_neurons,
                    activation = 'relu')) 
    model.add(Dropout(0.3))
    model.add(Dense(units=num_neurons,
                    activation = 'relu')) 
    model.add(Dropout(0.3))
                   
    # output layer
    model.add(Dense(units=1,
                    activation = 'sigmoid'))
    model.summary()

    metrics = [FalseNegatives(name='fn'),
               FalsePositives(name='fp'),
               TrueNegatives(name='tn'),
               TruePositives(name='tp'),
               Precision(name='precision'),
               Recall(name='recall'),
    ]
    #sgd = SGD(lr=0.001, decay=1e-7, momentum=.9)
    adam = Adam(1e-2)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=metrics)
    return model

if __name__ == '__main__':


    train  = pd.read_json('data/train.jsonl', lines=True)
    validate = pd.read_json('data/dev.jsonl', lines=True)
    sample_size = 1000
    print("Taking Sample")
    train_sample = train.sample(1000, random_state=13).reset_index(drop=True)
    ts = time()
    print("Making image arrays")
    img_array = load_and_featurize_data((16,16,3), train_sample, 'img')
    img_array = img_array.reshape(sample_size,768)
    tp1 = time()
    print('Image processing took {} seconds'.format(tp1-ts))

    vc = TfidfVectorizer(stop_words ='english', tokenizer=wordnet_lemmetize_tokenize, max_features=1000)
    tfidf_train = vc.fit_transform(train_sample.text.values)
    tp2 = time()
    print('Vectorizing took {} seconds'.format(tp2-tp1))

    X = np.hstack((tfidf_train.todense(),img_array))
    y = train_sample.label

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
    
    ### Standardize feature variables
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    X_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)
    #X_holdout_std = scaler.transform(X_holdout)

    ### Building and tuning a neural network model
    n_input = X_std.shape[1]
    weight_for_0, weight_for_1 = get_weights(y_train)
    weights ={0:weight_for_0, 1:weight_for_1}
    
    #ts = time()
    model = define_mlp_model(n_input)
    model.fit(X_std, y_train, epochs=30, batch_size=2048, verbose=2, 
              validation_data=(X_test_std, y_test), class_weight=weights)
    yhat = model.predict(X_test_std)
    score = roc_auc_score(y_test, yhat)
    te= time()
    print('ROC AUC: %.3f' % score)
    print("Time passed:", te-tp2)

    
