import nltk
import pandas as pd
import numpy as np
import os
import re
import string
import random
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
stop_words=set(nltk.corpus.stopwords.words('english'))
import gensim
from gensim import corpora
# libraries for visualization
import pyLDAvis
import pyLDAvis.gensim_models
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

path = "/home/subinay/Legal/Document-Clustering-Doc2vec-master/Clustering/Clean1/" #Add the path to Articles folder
seed = 137
def load_data(path,seed):
  train_texts = []
  for fname in sorted(os.listdir(path)):
    if fname.endswith('.txt'):
      with open(os.path.join(path,fname),'r',encoding='utf8') as f:
        train_texts.append(f.read())
  random.seed(seed)
  random.shuffle(train_texts)
  return train_texts
train_texts = load_data(path,seed)



from nltk.tokenize import RegexpTokenizer

# Split the documents into tokens.
tokenizer = RegexpTokenizer(r'\w+')
for idx in range(len(train_texts)):
    #train_texts[idx] = train_texts[idx].lower()  # Convert to lowercase.
    train_texts[idx] = tokenizer.tokenize(train_texts[idx])  # Split into words.

# Remove numbers, but not words that contain numbers.
train_texts = [[token for token in doc if not token.isnumeric()] for doc in train_texts]

# Remove words that are only one character.
train_texts = [[token for token in doc if len(token) > 1] for doc in train_texts]
from nltk.stem.wordnet import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
train_texts= [[lemmatizer.lemmatize(token) for token in doc] for doc in train_texts]

from gensim.corpora import Dictionary

# Create a dictionary representation of the documents.
dictionary =corpora.Dictionary(train_texts)

# Filter out words that occur less than 20 documents, or more than 50% of the documents.
dictionary.filter_extremes(no_below=2, no_above=0.5)

corpus = [dictionary.doc2bow(doc) for doc in train_texts]


print('Number of unique tokens: %d' % len(dictionary))
print('Number of documents: %d' % len(corpus))
lda = gensim.models.ldamodel.LdaModel
num_topics=10
%time ldamodel = lda(corpus,num_topics=num_topics,id2word=dictionary,passes=50,minimum_probability=0)
ldamodel.print_topics(num_topics=num_topics)
lda_corpus = ldamodel[corpus]

[doc for doc in lda_corpus]

L=[doc for doc in lda_corpus]
with open("Doc-topic.txt", "w") as file1:
    for i in range(len(corpus)):
          file1.write(','.join('%s %s' % x for x in L[i]))
          file1.write('\n')
          