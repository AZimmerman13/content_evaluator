from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import re
from string import punctuation
from sklearn import utils
from sklearn.base import BaseEstimator
import numpy as np


class CustomDoc2Vec(BaseEstimator):  #Credit: Sam Silver
    '''
    Wrapper for gensim Doc2Vec for use in SKLearn pipeline
    ginsem has this functionality built in (see: https://radimrehurek.com/gensim/sklearn_api/d2vmodel.html)
    This custom class exists due to some inconsistencies in the built-in SKLearn wrapper
    '''
    def __init__(
                self,
                seed=123,
                dm=0,
                vector_size=50,
                epochs=3,
                window=5,
                alpha=0.025,
                min_alpha=0.0001):
        self.model = None
        self.seed = seed
        self.dm = dm
        self.vector_size = vector_size
        self.epochs = epochs
        self.window = window
        self.alpha = alpha
        self.min_alpha = min_alpha

    def fit(self, X, y):
        self.model = Doc2Vec(
                            seed=self.seed,
                            dm=self.dm,
                            vector_size=self.vector_size,
                            epochs=self.epochs,
                            window=self.window,
                            alpha=self.alpha
                        )
        tagged_train = self._tag(X, y)
        self.model.build_vocab(tagged_train)
        shuffled_train = utils.shuffle(tagged_train, random_state=123)
        self.model.train(
            shuffled_train,
            total_examples=len(tagged_train),
            epochs=self.model.epochs
        )
        self.model.delete_temporary_training_data(
                        keep_doctags_vectors=True,
                        keep_inference=True
                    )
        return

    def transform(self, X):
        _, transformed_X = self.infer_vecs(self._tag(X))
        return transformed_X

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def _tag(self, X, y=None):
        idx_range = range(0, len(X))
        if y is not None:
            return [TaggedDocument(self._tokenize(X[idx]), y[idx]) for idx in idx_range]
        else:
            return [TaggedDocument(self._tokenize(X[idx]), idx) for idx in idx_range]

    def infer_vecs(self, tagged_docs, epochs=40):
        targets, vectors = zip(*[(doc.tags, self.model.infer_vector(doc.words, epochs=epochs)) for doc in tagged_docs])
        return targets, vectors

    def _tokenize(self, string):
        return string.strip().split(" ")