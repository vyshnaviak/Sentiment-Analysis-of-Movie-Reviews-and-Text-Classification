import math
import re
from collections import defaultdict, Counter

docs = ["I love football", "I like to watch to Haikyuu", "PM will address the nation today", "Budget for the financial year will be out soon", "Elections start this week"]
labels = ["sports", "sports", "politics", "politics", "politics"]

def tokenize(docs):
  return re.findall(r"\b\w+\b", docs.lower())

vocab = set()
for i in docs:
  vocab.update(tokenize(i))

class_counts = Counter()
class_word_counts = defaultdict(Counter)

for doc, label in zip(docs, labels):
  words = tokenize(doc)
  class_counts[label] += 1
  class_word_counts[label].update(words)

def train_bayes(class_counts, class_word_counts, vocab):
  class_prob = {}
  word_prob = {}
  total_class = sum(class_counts.values())
  for c in class_counts:
    class_prob[c] = class_counts[c]/total_class

    word_prob[c] = {}
    total_words = sum(class_word_counts[c].values())

    for w in vocab:
      word_prob[c][w] = (class_word_counts[c][w] + 1)/(total_words +len(vocab))

  return class_prob, word_prob

def predict(ip, class_prob, word_prob):
  words = tokenize(ip)
  scores = {}

  for c in class_prob:
    scores[c] = math.log(class_prob[c])

    for w in words:
      if w in word_prob[c]:
        scores[c] += math.log(word_prob[c][w])

  return max(scores, key=scores.get)

class_prob, word_prob = train_bayes(class_counts, class_word_counts, vocab)

s1 = "I liked volleyball after watching haikyuu."
s2 = "the budget is not good."
print(predict(s1, class_prob, word_prob))
print(predict(s2, class_prob, word_prob))





