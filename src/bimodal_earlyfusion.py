# outline of project
# 1. read in the image / pre-processing with feature engineering => converting images to 16x16 grey, unravel to 1x256
# 2. extracting features from texts using nlp => 500 features, 1x500
# 3. horizontal stack / concatenate (1x756)
# 4. training set (~8500), dev set is balanced (~500) 
# 5. model selection - RF, boosted Classifier, NN classifier (MLP), 

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from skimage.io import imread
from skimage.transform import resize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from src.helpers import wordnet_lemmetize_tokenize
from src.pipeline import ImagePipeline

def load_and_featurize_data(image_size, df, img_col):
    out_lst=[]
    for i in df[img_col]:
        img = imread(f'data/{i}')
        img = resize(img, image_size)
        out_lst.append(image_to_vector(img))
        # breakpoint()
        print(i)
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

if __name__ == '__main__':

    train  = pd.read_json('data/train.jsonl', lines=True)
    validate = pd.read_json('data/dev.jsonl', lines=True)
    train_sample = train.sample(10, random_state=13).reset_index(drop=True)

    img_array = load_and_featurize_data((16,16,3), train_sample, 'img')

    # tfidf = TfidfVectorizer(stop_words ='english', tokenizer=wordnet_lemmetize_tokenize, max_features=1000)
    # pipeline = ImagePipeline('data/img')
    # 
    # df = pd.read_pickle('data/pickled_data')
    

    # train['image_vec'] = img_array