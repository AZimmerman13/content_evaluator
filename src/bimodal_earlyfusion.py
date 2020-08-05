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

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.metrics import FalseNegatives, FalsePositives, TrueNegatives, TruePositives, Precision, Recall

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import accuracy_score, roc_auc_score

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

def thresh_test(thresh_list):
    for i in thresh_list:
            model = define_mlp_model(n_input)
            model.fit(X_std, y_train, epochs=30, batch_size=2048, verbose=2, 
                    validation_data=(X_test_std, y_test), class_weight=weights)
            yhat = model.predict(X_test_std)
            yhat = (yhat >= i).astype(int)


            score = roc_auc_score(y_test, yhat)
            print("With a threshold of {}:\n".format(i))
            print('ROC AUC: %.3f' % score)

def define_mlp_model(n_input):
    """ 
        define the multi-layer-perceptron neural network
    Args:
        n_input(int): number of features
    Returns:
        a defined mlp model, not fitted
    """
    model = Sequential()
    num_neurons = 256  #256

    # hidden layer
    model.add(Dense(units=num_neurons,
                    input_dim=n_input,
                    kernel_initializer='he_uniform',
                    activation = 'sigmoid'))
    model.add(Dense(units=num_neurons*2,
                    activation = 'sigmoid')) 
    model.add(Dropout(0.3))
    model.add(Dense(units=num_neurons*2,
                    activation = 'sigmoid')) 
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

    # validate = pd.read_json('data/dev.jsonl', lines=True)
    # submission = pd.read_json('data/test.jsonl', lines=True)

    data = np.load('data/train.npy')
    y = data[:,0]
    X = data[:, 1:] 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
    
    fit_mlp=True
    if fit_mlp:
        ts = time()
        ### Standardize feature variables
        scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
        X_std = scaler.fit_transform(X_train)
        X_test_std = scaler.transform(X_test)
        #X_holdout_std = scaler.transform(X_holdout)

        ### Building and tuning a neural network model
        weight_for_0, weight_for_1 = get_weights(y_train)
        weights ={0:weight_for_0, 1:weight_for_1}

        n_input = X_std.shape[1]
        thresh_list = np.arange(.1,1,.05)
        thresh = False
        if thresh:
            thresh_test(thresh_list)
        model = define_mlp_model(n_input)
        model.fit(X_std, y_train, epochs=30, batch_size=512, verbose=2, 
                    validation_data=(X_test_std, y_test), class_weight=weights)
        yhat = model.predict(X_test_std)
        # yhat = (yhat >= 0.6).astype(int)
        score = roc_auc_score(y_test, yhat)

        te= time()
        print('\n\nROC AUC: %.3f' % score)
        print("Time passed:", te-ts)


    validating = True
    if validating:
        validate_data = np.load('data/validate.npy')
        X_valid = validate_data[:,1:]
        y_valid = validate_data[:,0]
        X_valid_std = scaler.transform(X_valid)
        yhat = model.predict(X_valid_std)
        score = roc_auc_score(y_valid, yhat)
        te= time()
        print("\n\n\n\nValidation Set:\n\n")
        print('ROC AUC: %.3f' % score)
        print("Time passed:", te-ts)


    testing= False
    if testing:
        df = pd.read_json('data/test.jsonl', lines=True)
        submission = pd.DataFrame(columns=['id', 'proba', 'label'])
        test = np.load('data/test.npy')
        X_testing = test
        std = scaler.transform(X_testing)
        thresh = .5

        yhat = (thresh <= model.predict(std)).astype(int)
        y_proba = model.predict_proba(std)
        submission.id = df.id
        submission.proba = y_proba
        submission.label = yhat
        submission.set_index('id', inplace=True)
        submission.to_csv('data/submission.csv')
