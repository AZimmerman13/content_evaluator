import numpy as np
import pandas as pd
import nltk
# nltk.download('stopwords')
import re
import json

from nltk.corpus import stopwords
from collections import Counter
from tensorflow import keras
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Embedding, LSTM, Dropout
from tensorflow.keras.metrics import FalseNegatives, FalsePositives, TrueNegatives, TruePositives, Precision, Recall

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import accuracy_score, roc_auc_score

class Vocabulary: # Credit http://www.insightsbot.com/keras-lstm-example-sequence-binary-classification/
    
    def __init__(self, vocabulary, wordFrequencyFilePath):
        self.vocabulary = vocabulary
        self.WORD_FREQUENCY_FILE_FULL_PATH = wordFrequencyFilePath
        self.input_word_index = {}
        self.reverse_input_word_index = {}
        
        self.MaxSentenceLength = None
        
    def PrepareVocabulary(self,reviews):
        self._prepare_Word_Frequency_Count_File(reviews)
        self._create_Vocab_Indexes()
        
        self.MaxSentenceLength = max([len(txt.split(" ")) for txt in reviews])
      
    def Get_Top_Words(self, number_words = None):
        if number_words == None:
            number_words = self.vocabulary
        
        chars = json.loads(open(self.WORD_FREQUENCY_FILE_FULL_PATH).read())
        counter = Counter(chars)
        most_popular_words = {key for key, _value in counter.most_common(number_words)}
        return most_popular_words
    
    def _prepare_Word_Frequency_Count_File(self,reviews):
        counter = Counter()    
        for s in reviews:
            counter.update(s.split(" "))
            
        with open(self.WORD_FREQUENCY_FILE_FULL_PATH, 'w') as output_file:
            output_file.write(json.dumps(counter))
                 
    def _create_Vocab_Indexes(self):
        INPUT_WORDS = self.Get_Top_Words(self.vocabulary)
        for i, word in enumerate(INPUT_WORDS):
            self.input_word_index[word] = i
        
        for word, i in self.input_word_index.items():
            self.reverse_input_word_index[i] = word
       
    def TransformSentencesToId(self, sentences):
        vectors = []
        for r in sentences:
            words = r.split(" ")
            vector = np.zeros(len(words))
            for t, word in enumerate(words):
                if word in self.input_word_index:
                    vector[t] = self.input_word_index[word]
                else:
                    pass
                
            vectors.append(vector)
            
        return vectors
    
    def ReverseTransformSentencesToId(self, sentences):
        vectors = []
        for r in sentences:
            words = r.split(" ")
            vector = np.zeros(len(words))
            for t, word in enumerate(words):
                if word in self.input_word_index:
                    vector[t] = self.input_word_index[word]
                else:
                    pass
                    #vector[t] = 2 #unk
            vectors.append(vector)
            
        return vectors

REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|()|()|()|()")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
def preprocess_data(reviews):
    default_stop_words = nltk.corpus.stopwords.words('english')
    stopwords = set(default_stop_words)
    
    reviews = [REPLACE_NO_SPACE.sub("", line.lower()) for line in reviews]
    reviews = [REPLACE_WITH_SPACE.sub(" ", line) for line in reviews]
    reviews = [RemoveStopWords(line,stopwords) for line in reviews]
    
    return reviews

def RemoveStopWords(line, stopwords):
    words = []
    for word in line.split(" "):
        word = word.strip()
        if word not in stopwords and word != "" and word != "&":
            words.append(word)

    return " ".join(words)


def define_lstm(TOP_WORDS, max_sent_len):
    embedding_vector_length = 16
    model = Sequential()
    num_neurons = 16
    # Input layer
    model.add(Embedding(TOP_WORDS, 
                        embedding_vector_length, 
                        input_length=max_sent_len))

    model.add(LSTM(16, return_sequences=True))
    model.add(Dropout(0.4))

    model.add(LSTM(16))
    model.add(Dropout(0.4))

    # model.add(Dense(units=num_neurons,
    #                 activation = 'sigmoid'))
    # model.add(Dropout(0.3))

    model.add(Dense(units=num_neurons,
                    activation = 'relu'))
    model.add(Dropout(0.4))


    
    # Output layer
    model.add(Dense(1, 
                    activation='sigmoid'))

    metrics = [FalseNegatives(name='fn'),
               FalsePositives(name='fp'),
               TrueNegatives(name='tn'),
               TruePositives(name='tp'),
               Precision(name='precision'),
               Recall(name='recall'),
    ]

    model.summary()

    model.compile(loss='binary_crossentropy', 
                optimizer='adam', 
                metrics=metrics)
    
    return model

if __name__ == '__main__':

    # Load train data
    df = pd.read_json('data/train.jsonl', lines=True)
    data = df.text.values
    data = preprocess_data(data)

    # Load validation data
    val = pd.read_json('data/dev.jsonl', lines=True)
    val_data = val.text.values
    val_data = preprocess_data(val_data)

    # Create Vocabulary
    TOP_WORDS = 1000
    vocab = Vocabulary(TOP_WORDS,"analysis.vocab")
    vocab.PrepareVocabulary(data)

    # Convert text to int
    val_int = vocab.TransformSentencesToId(val_data)
    val['text_int'] = val_int
    text_int = vocab.TransformSentencesToId(data)
    df['text_int'] = text_int

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(df.text_int.values, df.label.values)
    X_val = val.text_int.values
    y_val = val.label.values

    # Padding
    length_lst = [len(i) for i in df.text_int]
    max_sent_len = max(length_lst) # 47
    X_train = sequence.pad_sequences(X_train, maxlen=max_sent_len) 
    X_test = sequence.pad_sequences(X_test, maxlen=max_sent_len)
    X_val = sequence.pad_sequences(X_val, maxlen=max_sent_len)

    model = define_lstm(TOP_WORDS, max_sent_len)
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=64, verbose=2)

    cv_auc = roc_auc_score(y_test, model.predict(X_test))
    val_auc = roc_auc_score(y_val, model.predict(X_val))
    print("AUC SCORE ON CROSS VALIDATION:\n {}".format(cv_auc))
    print("\n\nAUC SCORE ON HOLDOUT:\n {}".format(val_auc))

    testing = False
    if testing:

        sub_df = pd.read_json('data/test.jsonl', lines=True)
        sub_data = sub_df.text.values
        sub_int = vocab.TransformSentencesToId(sub_data)
        sub_df['text_int'] = sub_int
        X_sub = sub_df.text_int.values
        X_sub = sequence.pad_sequences(X_sub, maxlen=max_sent_len)

        thresh = .5

        yhat = (thresh <= model.predict(X_sub)).astype(int)
        y_proba = model.predict_proba(X_sub)
        submission = pd.read_csv('data/blank_submission.csv')
        submission.proba = y_proba
        submission.label = yhat
        submission.set_index('id', inplace=True)
        submission.drop('Unnamed: 0', inplace=True, axis=1)
        submission.to_csv('data/lstm_submission.csv')
