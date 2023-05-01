# -- Imports --
import numpy as np
import pandas as pd

# -- Load the Data --
data = pd.read_csv('train.En.csv')
test = pd.read_csv('task_A_En_test.csv')

# We only need the text and whether it's sarcastic
data = data[['tweet', 'sarcastic']]
# Rename tweet to text to stay consistent with the test set
data.rename(columns={'tweet': 'text'}, inplace=True)
# Ensure datatypes are what we expect
data['text'] = data['text'].astype('string')
data['sarcastic'] = data['sarcastic'].astype('int')
data.dropna(inplace=True)
# Display the test data
data


# -- Tokenize Data --
from sklearn.model_selection import train_test_split
from collections import Counter
import re

# This regex selects anything that is not a lowercase alphabet or a space
alpha = re.compile('[^a-z ]')

# This is where the bulk of our preprocessing happens
class Tokenizer:
  def __init__(self, sentences):
    # gets every pair of words in the entire text
    words = [w for s in self.get_words(sentences) for w in s]
    # maps the 2200 most common bigram to a number
    # throws out the 200 most common though
    self.wtoi = {w:i for i,(w,c) in enumerate(Counter(words).most_common(2200)[200:])}
    # counts the number of words in the mapping
    self.n_words = len(self.wtoi)
  def get_words(self, sentences):
    # makes every sentence lowercase, removes all non-letters, and then splits based on spaces
    a = [alpha.sub('', s.lower()).split() for s in sentences]
    # gets every pair of words (bigrams) unless the tweet is one word in which case it's a unigram
    return [[' '.join(p) for p in zip(ws, ws[1:])] if len(ws) > 2 else ' '.join(ws) for ws in a]
  def tokenize(self, sentences):
    # Makes a vector object to hold the tokens
    vec = np.zeros((len(sentences), self.n_words))
    # This code count how many occurances of each bigram occur in the sentence
    # and increments the corresponding index in the vector based on self.wtoi
    for i, s in enumerate(self.get_words(sentences)):
      for w in s:
        if w in self.wtoi:
          vec[i][self.wtoi[w]] += 1
    # Returns the vector
    return vec

# Initialize the tokenizer
tz = Tokenizer(data['text'])
# Get our training and test data
x_train, x_valid, y_train, y_valid = train_test_split(tz.tokenize(data['text']), data['sarcastic'], test_size = 0.1, random_state=42)


# -- Train Model --

# We are using a decision tree classifier for our model
from sklearn.tree import DecisionTreeClassifier

# Train it with a random state for consistency
model = DecisionTreeClassifier(random_state=42, max_depth=300)
model.fit(x_train, y_train)
model.score(x_valid, y_valid)

# -- Save predictions to a file --
test['sarcastic'] = model.predict(tz.tokenize(test['text']))
test.to_csv('task_A_En_output.csv', index=False)