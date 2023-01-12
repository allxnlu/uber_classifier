#cleaning function
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from textblob import TextBlob
import json
import requests
from cleantext import clean



#evaluate model with metrics for each label
def model_evaluate(model, X, y_test, y_pred):
    y_pred = model.predict(X)
    print(accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

#correct spelling of words
def correct_spelling(text):
    text = TextBlob(text)
    text = text.correct()
    return str(text)

#remove stopwords
def remove_stopwords(text, stop=[]):
    text_list = word_tokenize(text)
    text_list = [word for word in text_list if word not in stop]
    return " ".join(text_list)

#collect stopwords
def get_stopwords():
    url = "https://countwordsfree.com/stopwords/english/json"
    response = pd.DataFrame(data = json.loads(requests.get(url).text))
    stop = list(response)
    # nltk.download('stopwords')
    # nltk.download('punkt')
    stop.extend(stopwords.words('english'))
    return stop

#stem the data
def stemmize(sentence):
    ps = PorterStemmer()
    words = word_tokenize(sentence)
    words = [ps.stem(word) for word in words]
    words = " ".join(words)
    return words

#lemmatize the data
def lemmize(sentence):
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(sentence)
    words = [lemmatizer.lemmatize(word) for word in words]
    words = " ".join(words)
    return words


#clean function
import string
import emoji

def cleaning(text):
    # text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    # text = re.sub(r"http\S+", "", text)
    # # text = re.sub("@\w+|\d+"," ",text)
    # text = re.sub("@[A-Za-z0-9_]+", "", text)
    # text = text.lower()
    # text = re.sub("  ", " ", text)
    # text = re.sub('#[^A-Za-z]+', '', text)
    # # text = re.sub(r'\\n', '', text)
    # # text = re.sub(r"\\\"" , '', text)
    # # text = re.sub(r"\\" , '', text)
    # text = re.sub("\?", '', text)
    # text = re.sub("\'", '', text)
    # text = re.sub("\d+", '', text)
    # text = re.sub("!", '', text)
    # # text = text.translate(text.maketrans('', '', string.punctuation))
    # text = clean(text, no_emoji=True)
    text = text.lower()
    text = re.sub("@[A-Za-z0-9_]+", ' ', text)
    text = re.sub("#[A-Za-z0-9_]+", ' ' , text)
    text = re.sub(r'(.)1+', r'1', text)
    text = re.sub('((www.[^s]+)|(https?://[^s]+))',' ', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub('[^a-zA-Z]', ' ', text)     # punctuations
    text = emoji.replace_emoji(text, replace='') 
    text = re.sub(r"\s+[a-zA-Z]\s+", ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(' +', ' ', text)
    text = text.strip()
    return text





