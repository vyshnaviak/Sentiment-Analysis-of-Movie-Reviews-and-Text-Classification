import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_files

import urllib.request, zipfile, os

url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
filename = "aclImdb_v1.tar.gz"

if not os.path.exists(filename):
    urllib.request.urlretrieve(url, filename)

import tarfile

tar = tarfile.open(filename, "r:gz")
tar.extractall()
tar.close()

reviews_train = load_files("aclImdb/train/", categories=["pos", "neg"], encoding="utf-8")
reviews_test = load_files("aclImdb/test/", categories=["pos", "neg"], encoding="utf-8")

x_train = reviews_train.data
y_train = reviews_train.target
x_test = reviews_test.data
y_test = reviews_test.target

vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
x_train_tfidf = vectorizer.fit_transform(x_train)
x_test_tfidf = vectorizer.fit_transform(x_test)

model = MultinomialNB()
model.fit(x_train_tfidf, y_train)

y_pred = model.predict(x_train_tfidf)

print("Accuracy: ", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

sample_reviews = ["This was an amazing move, loved the feels and music",
                  "I slept through the movie"]

sample_features = vectorizer.transform(sample_reviews)
predictions = model.predict(sample_features)
map = {1: "Negative", 0: "Positive"}
for review, sentiment in zip(sample_reviews, predictions):
    print(review)
    print(map[sentiment])