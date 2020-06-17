
from src.d2vcustom import CustomDoc2Vec
from gensim.models import Word2Vec
from src.helpers import wordnet_lemmetize_tokenize

import pandas as pd
import numpy as np
from string import punctuation
import re
import random
from joblib import dump, load
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.cluster import DBSCAN

def test_alphas(alphas):
    for a in alphas:
        nb_classifier = MultinomialNB(alpha=a)
        nb_classifier.fit(tfidf_train, y_train)
        y_pred = nb_classifier.predict(tfidf_train)
        print(a)
        print(f"Train accuracy: {accuracy_score(y_train, y_pred)}\n\n")
    return



if __name__ == '__main__':
    train  = pd.read_json('data/train.jsonl', lines=True)
    test = pd.read_json('data/test.jsonl', lines=True)
    tfidf = TfidfVectorizer(stop_words ='english', tokenizer=wordnet_lemmetize_tokenize)
    count_vec = CountVectorizer(stop_words='english', tokenizer=wordnet_lemmetize_tokenize)
    count_train = tfidf.fit_transform(train.text.values)
    tfidf_train = count_vec.fit_transform(train.text.values)
    count_test = tfidf.transform(test.text.values)
    tfidf_test = count_vec.transform(test.text.values)
    # X_train = train.text.values
    y_train = train.label.values
    # X_test = test.text.values
    # y_test = test.label.values

    # Featurize text

    nb = False
    if nb:
        nb_classifier = MultinomialNB()
        nb_classifier.fit(count_train, y_train)
        y_pred = nb_classifier.predict(count_train)
        print('Count Vectorizer')
        print(f"Train accuracy: {accuracy_score(y_train, y_pred)}")

        nb_classifier.fit(tfidf_train, y_train)
        y_pred = nb_classifier.predict(tfidf_train)
        print('\n\nTfidf Vectorizer')
        print(f"Train accuracy: {accuracy_score(y_train, y_pred)}")
        cm = confusion_matrix(y_train, y_pred)
        print(cm)



    random_forest = True
    if random_forest:
        rf = RandomForestClassifier(oob_score=True, n_jobs=-1)
        rf.fit(tfidf_train, y_train)
        y_pred = rf.predict(tfidf_train)
        y_proba = rf.predict_proba(tfidf_train)
        print('\n\nRandom Forest (w/ Tfidf) Vectorizer')
        print(f"Train accuracy: {accuracy_score(y_train, y_pred)}")
        print(f"AUC: {roc_auc_score(y_train, y_proba[:,1])}")
        cm = confusion_matrix(y_train, y_pred)
        print(cm)

    alphas = np.arange(0.05,1,0.05)
    # test_alphas(alphas)

    # Cluster text into topics
    
    # DBSCAN metric='cosine'
    cluster=True
    if cluster:
        dbs = DBSCAN(n_jobs=-1, metric='cosine')
        dbs.fit(tfidf_train)
        print(dbs.labels_)
    

'''
    d2v = CustomDoc2Vec(
                            seed=123,
                            dm=0,
                            vector_size=50,
                            epochs=5,
                            window=20,
                            alpha=0.025,
                            min_alpha=0.001
                        )

    pipe = Pipeline([
        ('doc2vec', d2v),
        ('model', RandomForestClassifier(
                            random_state=42,
                            n_jobs=-1,
                            n_estimators=400,
                            max_features='auto',
                            oob_score=True,
                            class_weight='balanced_subsample',
                            min_samples_split=10,
                            min_samples_leaf=1,
                        ))
    ])


    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_train)
    print('Training accuracy %s' % accuracy_score(y_train, y_pred))
    print(f"Training F1 score: {f1_score(y_train, y_pred, average='macro')}")

    # print_model_metrics(y_test, y_pred)

    pipe.fit(X, y)
    '''