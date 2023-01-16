#cleaning function
import re
import string
import nltk
nltk.download('punkt')
import emoji
import pandas as pd
import json
import requests
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.metrics import accuracy_score, classification_report
from textblob import TextBlob
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
    nltk.download('stopwords')
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

def cleaning(text):
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





