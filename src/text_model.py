
from src.d2vcustom import CustomDoc2Vec
from gensim.models import Word2Vec
from src.helpers import wordnet_lemmetize_tokenize

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from string import punctuation
import re
import random
from joblib import dump, load
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score, recall_score, precision_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.cluster import DBSCAN, KMeans

from textblob import TextBlob

plt.style.use('fivethirtyeight')

def test_alphas(alphas):
    for a in alphas:
        nb_classifier = MultinomialNB(alpha=a)
        nb_classifier.fit(tfidf_train, y_train)
        y_pred = nb_classifier.predict(tfidf_train)
        print(a)
        print(f"Train accuracy: {accuracy_score(y_train, y_pred)}\n\n")
    return

def test_thresholds(fit_model):
    thresh_list = np.arange(0.1,1, 0.05)
    score_list = {}
    for i in thresh_list:
        # rf = RandomForestClassifier(class_weight='balanced', n_estimators=300, max_features=3, max_leaf_nodes=50, random_state=42, n_jobs=-2, oob_score=True)
        # model.fit(tfidf_train, y_train)

        preds = fit_model.predict_proba(tfidf_train)
        preds = (preds[:,1] >= i).astype('int')

        holdout_preds = fit_model.predict_proba(tfidf_test)
        holdout_preds = (holdout_preds[:,1] >= i).astype('int')

        auc = roc_auc_score(y_test, holdout_preds)
        rec = recall_score(y_test, holdout_preds)
        score_list[rec] = i

        print(f"With a threshold of {i}:\n")
        print(f"Training: \nF1: {f1_score(y_train, preds)}, \nRecall: {recall_score(y_train, preds)}, \nAccuracy: {fit_model.score(tfidf_train, y_train)}, \nPrecision: {precision_score(y_train, preds)}\n")
        print(f"Test: \nF1: {f1_score(y_test, holdout_preds)}, \nRecall: {recall_score(y_test, holdout_preds)}, \nAccuracy: {fit_model.score(tfidf_test, y_test)}, \nPrecision: {precision_score(y_test, holdout_preds)}\n")
        print(f"AUC: {auc}\n")

        cm = confusion_matrix(y_test, holdout_preds)
        print(cm, '\n\n\n')

    print(f"best Recall: {max(score_list.keys())} \nAt threshold :{score_list[max(score_list.keys())]}")


def blobify(df):
    '''
    adds a feature to the given dataframe that contains a
    TextBlob polarity score for the te
    '''
    sent = [TextBlob(str(i)).polarity for i in df.text]
    df['blob_polarity'] = sent
    return df

 # Try CalibratedClassifierCV 
 # Try textblob
 # Show wordcloud for each category

 # predict_proba values only between 0.47 and 0.53
if __name__ == '__main__':
    # instantiate vectorizer
    tfidf = TfidfVectorizer(stop_words ='english', tokenizer=wordnet_lemmetize_tokenize, max_features=5000)
    # Load and tranform data
    train  = pd.read_json('data/train.jsonl', lines=True)
    train_df = train.copy()
    train_df = blobify(train_df)

    tfidf_train = tfidf.fit_transform(train.text.values)
    train_df['vector'] = list(tfidf_train.toarray())
    X_train = np.array(train_df.iloc[:,-2]).reshape(-1,1)
    y_train = train.label.values

    # Load and transform test data
    test = pd.read_json('data/dev.jsonl', lines=True)
    test_df = test.copy()
    test_df = blobify(test_df)
    
    tfidf_test = tfidf.transform(test.text.values)
    test_df['vector'] = list(tfidf_test.toarray())
    X_test = np.array(test_df.iloc[:,-2]).reshape(-1,1)
    y_test = test.label.values.reshape(-1,1)

    # Featurize text

    nb = False
    if nb:
        nb_classifier = MultinomialNB()
        nb_classifier.fit(X_train, y_train)
        # Train score
        y_pred = nb_classifier.predict(X_train)
        print('\n\nNaive Bayes')
        print(f"Train accuracy: {accuracy_score(y_train, y_pred)}")
        # Validation Score
        y_pred = nb_classifier.predict(X_test)
        print(f"Test accuracy: {accuracy_score(y_test, y_pred)}")
        y_proba = nb_classifier.predict_proba(X_test)
        print(f"AUC: {roc_auc_score(y_test, y_proba[:,1])}")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        test_thresholds(nb_classifier)



    random_forest = True
    if random_forest:
        rf = RandomForestClassifier(oob_score=True, n_jobs=-1, class_weight='balanced', max_depth=3)
        rf.fit(X_train, y_train)
        # Train score
        y_pred = rf.predict(X_train)
        y_proba = rf.predict_proba(X_train)
        print('\n\nRandom Forest')
        print(f"Train accuracy: {accuracy_score(y_train, y_pred)}")
        # Validation score
        y_pred = rf.predict(X_test)
        y_proba = rf.predict_proba(X_test)
        print(f"Test accuracy: {accuracy_score(y_test, y_pred)}")
        print(f"Recall: {recall_score(y_test, y_pred)}")
        print(f"AUC: {roc_auc_score(y_test, y_proba[:,1])}")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        tn, fp, fn, tp = cm.ravel()
        print(tn, fp, fn, tp)

        # test_thresholds(rf)

    alphas = np.arange(0.05,1,0.05)
    # test_alphas(alphas)

    # Cluster text into topics
    
    # DBSCAN metric='cosine'
    cluster=False
    if cluster:
        # dbs = DBSCAN(n_jobs=-1, metric='cosine')
        # dbs.fit(tfidf_train)
        # print(dbs.labels_)
        
        km = KMeans(n_clusters=150, verbose=1, n_jobs=-1)
        km.fit(X_train)
        train_clusters = km.predict(X_train)

        # sum_squared_distances.append(km.inertia_)

        # fig, ax = plt.subplots()
        # ax.plot(K, sum_squared_distances, 'bx-')
        # plt.show()


    stack = cluster
    if stack:
        stack_df = pd.DataFrame(train_clusters, columns=['nlp_cluster'])
        top_model = RandomForestClassifier(n_jobs=-1)
        top_model.fit(train_clusters.reshape(-1,1), y_train)
        y_proba = top_model.predict_proba(train_clusters.reshape(-1,1))
        y_pred = top_model.predict(train_clusters.reshape(-1,1))
        print('\n\nRandom Forest (w/ Tfidf) Vectorizer')
        print(f"Train accuracy: {accuracy_score(y_train, y_pred)}")
        print(f"AUC: {roc_auc_score(y_train, y_proba[:,1])}")
        cm = confusion_matrix(y_train, y_pred)
        print(cm)


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