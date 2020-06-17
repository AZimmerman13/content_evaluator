from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

sw = ENGLISH_STOP_WORDS

def wordnet_lemmetize_tokenize(text):
    '''
    Custom tokenizer object that applies WordNetLemmatizer
    Intended to be passed into CountVectorizer as a tokenizer object
    '''
    lemmatizer = WordNetLemmatizer()
    words = text.split()

    # additional lemmatization terms
    additional_lemmatize_dict = {
        "accidentally": "accident",
        "accidently": "accident",
        "thanks": "thank"
    }
    
    tokens = []
    for word in words:
        if word not in sw:
            if word in additional_lemmatize_dict:
                clean_word = additional_lemmatize_dict[word]
            else:
                clean_word = lemmatizer.lemmatize(word)
            tokens.append(clean_word)
    return tokens

    def print_model_metrics(y_test, y_preds):
        '''
        Print classification matrix and confusion matrix for a given prediction
        '''
        class_rept_dict = classification_report(y_test, y_preds, output_dict=True)
        class_rept_df = pd.DataFrame(class_rept_dict).transpose()
        print(class_rept_df.to_markdown())
        cmtx = pd.DataFrame(
            confusion_matrix(y_test, y_preds, labels=[
                            'negative', 'neutral', 'positive']),
            index=['true:negative', 'true:neutral', 'true:positive'],
            columns=['pred:negative', 'pred:neutral', 'pred:positive']
        )
        print("\n")
        print(cmtx.to_markdown())
        return