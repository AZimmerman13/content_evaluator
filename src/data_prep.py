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


def load_and_featurize_data(filepath, image_size, savePickle=False, outpath=None, sample_size=-1):
    df  = pd.read_json(filepath, lines=True)
    out_lst=[]
    ts = time()
    if sample_size>0:
        print("Taking Sample")
        data = df.sample(sample_size, random_state=13).reset_index(drop=True)
    else:
        print("Taking All Data")
        data = df

    for i in data.img:
        img = imread('data/{}'.format(i))
        img = resize(img, image_size)
        out_lst.append(image_to_vector(img))
        print(i)

    data_size = data.shape[0]
    img_array = np.array(out_lst).reshape(data_size,image_size[0]*image_size[1]*image_size[2])
    #img_array = img_array.reshape(sample_size,768)
    tp1 = time()
    print('Image processing took {} seconds'.format(tp1-ts))

    vc = TfidfVectorizer(stop_words ='english', tokenizer=wordnet_lemmetize_tokenize, max_features=1000)
    tfidf_data = vc.fit_transform(data.text.values)
    tp2 = time()
    print('Vectorizing took {} seconds'.format(tp2-tp1))

    #breakpoint()
    X = np.hstack((tfidf_data.todense(),img_array))
    y = data.label.values
    
    if savePickle:
        np.save(outpath,np.hstack((y.reshape(-1,1),X)))
    return X, y 

def image_to_vector(image):
    """
    Args:
    image: numpy array of shape (length, height, depth)

    Returns:
     v: a vector of shape (length x height x depth, 1)
    """
    length, height, depth = image.shape
    return image.reshape((length * height * depth, 1))

if __name__ == '__main__':

    img_size = (16,16,3)
    sample_size = -1

    # validate = pd.read_json('data/dev.jsonl', lines=True)
    # submission = pd.read_json('data/test.jsonl', lines=True)

    X, y = load_and_featurize_data('data/train.jsonl', image_size=img_size, savePickle=True, outpath = 'data/train', sample_size=sample_size)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
    


    
