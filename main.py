import re

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

RANDOM_SEED = 42
TEST_SIZE = 0.2

nltk.download('stopwords')

stop_words = set(stopwords.words('english'))


def preprocess_text(message):
    message = re.sub('[^a-zA-Z]', ' ', message).lower().strip().split()
    message = [word for word in message if word not in stop_words]
    return ' '.join(message)


data = pd.read_csv('spam.csv', encoding='latin1')

data_renamed = data[['v1', 'v2']].rename(columns={"v1": "label", "v2": "message"})

data_renamed['message_len'] = data_renamed['message'].str.len()

avg = data_renamed.groupby(['label'])['message_len'].mean()
print(f"AVG:\n{avg}\n")

data_renamed['cleaned_message'] = data_renamed['message'].apply(preprocess_text)
print(f"\nDATA:\n{data_renamed[['message', 'cleaned_message']].head()}\n")

X = data_renamed['cleaned_message']
y = data_renamed['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED)

vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = MultinomialNB()

model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

# First way to check model accuracy (handmade)
correct = sum([list(y_test)[i] == y_pred[i] for i in range(len(y_pred))])
accuracy_1 = correct / len(y_pred)
print(f"\nAccuracy 1: {accuracy_1}\n")

# Second way to check accuracy (handmade, better, with numpy)
matches = (y_test == y_pred)
accuracy_2 = np.sum(matches) / len(y_test)
print(f"\nAccuracy 2: {accuracy_2}\n")

# Third way to check accuracy (auto)
accuracy_3 = model.score(X_test_tfidf, y_test)
print(f"\nAccuracy 3: {accuracy_3}\n")

# 4th way to check accuracy (and other info)
class_rep = classification_report(y_test, y_pred)
print(f"\nClassification Report:\n{class_rep}\n")

conf_max = confusion_matrix(y_test, y_pred)

print(f"\nConfusion Matrix:\n{conf_max}")
