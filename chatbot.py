import pandas as pd
import joblib

from functions import *

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


def intent_label(label):
    return labels[int(label)]

def runLogisticRegression(x_train, x_test, y_train, y_test):
    clf = LogisticRegression()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred, normalize=True)
    accuracy_count = accuracy_score(y_test, y_pred, normalize=False)
    f1score = f1_score(y_test,y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall= recall_score(y_test, y_pred, average='weighted')
    print ('Logistic Regression Accuracy: ', accuracy)
    print ('Logistic Regression Accuracy Count: ', accuracy_count)
    print ('Logistic Regression F1 Score: ', f1score)
    print ('Logistic Regression Precision: ', precision)
    print ('Logistic Regression Recall: ', recall)
    print('*'*40)
    model_evaluate(clf, x_test, y_test, y_pred)
    joblib.dump(clf, "models/LR.pkl")


def runRandomForest(x_train, x_test, y_train, y_test):
    clf = RandomForestClassifier()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred, normalize=True)
    accuracy_count = accuracy_score(y_test, y_pred, normalize=False)
    f1score = f1_score(y_test,y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall= recall_score(y_test, y_pred, average='weighted')
    print ('Random Forest Accuracy: ', accuracy)
    print ('Random Forest Accuracy Count: ', accuracy_count)
    print ('Random Forest F1 Score: ', f1score)
    print ('Random Forest Precision: ', precision)
    print ('Random Forest Recall: ', recall)
    print('*'*40)
    model_evaluate(clf, x_test, y_test, y_pred)
    joblib.dump(clf, "models/RF.pkl")

def runSVM(x_train, x_test, y_train, y_test):
    clf = LinearSVC()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    accuracy =  accuracy_score(y_test, y_pred, normalize=True)
    accuracy_count = accuracy_score(y_test, y_pred, normalize=False)
    f1score = f1_score(y_test,y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    print ('SVM Accuracy: ', accuracy)
    print ('SVM Accuracy Count: ', accuracy_count)
    print ('SVM F1 Score: ', f1score)
    print ('SVM Precision: ', precision)
    print ('SVM Recall: ', recall)
    print ('*'*40)
    model_evaluate(clf, x_test, y_test, y_pred)
    joblib.dump(clf, "models/SVM2.pkl")

def runCalibratedSVM(x_train, x_test, y_train, y_test):
    svc = LinearSVC(C=0.09)
    clf = CalibratedClassifierCV(svc)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred, normalize=True)
    accuracy_count = accuracy_score(y_test, y_pred, normalize=False)
    f1score = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall= recall_score(y_test, y_pred, average='weighted')
    print ('Calibrated SVM Accuracy: ', accuracy)
    print ('Calibrated SVM Accuracy Count: ', accuracy_count)
    print ('Calibrated SVM F1 Score : ', f1score)
    print ('Calibrated SVM Precision: ', precision)
    print ('Calibrated SVM Recall: ', recall)
    print('*'*40)   
    model_evaluate(clf, x_test, y_test, y_pred)
    joblib.dump(clf, "models/CSVM2.pkl")


def runMLP(x_train, x_test, y_train, y_test):
    clf = MLPClassifier()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred, normalize=True)
    accuracy_count = accuracy_score(y_test, y_pred, normalize=False)
    f1score = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall= recall_score(y_test, y_pred, average='weighted')
    print ('MLP Accuracy: ', accuracy)
    print ('MLP Accuracy Count: ', accuracy_count)
    print ('MLP F1 Score : ', f1score)
    print ('MLP Precision: ', precision)
    print ('MLP Recall: ', recall)
    print('*'*40)
    model_evaluate(clf, x_test, y_test, y_pred)
    joblib.dump(clf, "models/MLP.pkl")

################################################################################################

df = pd.read_csv('datasets/uber_dataset.csv', usecols=["question", "answer", "labels"])
labels = list(df.labels.unique())
print(df)

labels = ["App Help", "Contact DM", "Driver Issues", "Time Management", "Trip Issues", "UberEats Issues", "Contact Email", "Help Need", "Promo Code Issues", "Service Request", "Customer Service Feedback", "Uber Company Feedback", "Account Issues", "Lost and Found", "DM Sent"]

intents = [intent_label(label) for label in df.labels]
df["intent"]= intents
print(df)

stopp = get_stopwords()

df.question = df.question.astype(str).apply(cleaning)
# df.questions = df.question.apply(remove_stopwords, stop=stopp)
df.question = df.question.astype(str).apply(lemmize)
X = df.question.astype(str)
y = df.intent

tv = TfidfVectorizer(stop_words="english", ngram_range=(1,3), min_df=2, max_df=.5)
tv_vector = tv.fit_transform(X)

joblib.dump(tv, "models/uber_vectorizer.pkl")

X = tv_vector
y = df.intent

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, stratify=y)
X_train.shape

runSVM(X_train, X_test, y_train, y_test)
runCalibratedSVM(X_train, X_test, y_train, y_test)
runRandomForest(X_train, X_test, y_train, y_test)
runMLP(X_train, X_test, y_train, y_test)
runLogisticRegression(X_train, X_test, y_train, y_test)