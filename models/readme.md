# Trained Models Folder
In this folder the vectorizer and trained machine learning models are stored. 

To use the machine learning models, call `joblib` 

    joblib.load(open('{VECTORIZER_NAME.pkl}', 'rb'))

or

    joblib.load(open('{CLASSIFIER_MODEL.pkl}', 'rb'))
