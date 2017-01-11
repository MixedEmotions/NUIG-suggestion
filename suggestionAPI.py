from __future__ import print_function
from flask import Flask, jsonify, request
import operator
import numpy as np
import sys

np.random.seed(1337)  # for reproducibility

import os
import codecs
import csv
import re
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing import sequence
from keras.models import load_model
from keras.utils.np_utils import to_categorical
from sklearn.metrics import f1_score
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Embedding
from keras import backend as K
app = Flask(__name__)


# set parameters:
testPath = "/Users/sapna/PycharmProjects/MEsentiment/data/suggestion/tweets_microsoft.csv"
#savedModelPath = http://server1.nlp.insight-centre.org/MEsavedModels/CNN_twitterSugg_model.h5
Embedding_dir = "/Users/sapna/PycharmProjects/Keyphrase-Extractor/Glove/glove.twitter.27B.50d.txt"
savedModelPath = "/Users/sapna/PycharmProjects/MEsentiment/savedModels/CNN_twitterSugg_model.h5"
max_no_words = 27000000000 #fixed, should be same as the number of words in embeddings
maxlen= 40
text_instance = "some tweet"


# set parameters:
testPath = "/Users/sapna/PycharmProjects/MEsentiment/data/suggestion/tweets_microsoft.csv"
#savedModelPath = http://server1.nlp.insight-centre.org/MEsavedModels/CNN_twitterSugg_model.h5
Embedding_dir = "/Users/sapna/PycharmProjects/Keyphrase-Extractor/Glove/glove.twitter.27B.50d.txt"
savedModelPath = "/Users/sapna/PycharmProjects/MEsentiment/savedModels/CNN_twitterSugg_model.h5"
max_no_words = 27000000000 #fixed, should be same as the number of words in the pretrained embeddings
maxlen= 40
text_instance = "some tweet"


# start process_tweet
def cleanTweet(tweet):
    tweet = tweet.lower()
    tweet = " " + tweet
    tweet = re.sub(r'[^\x00-\x7F]+', '', tweet)
    tweet = re.sub(' rt ', '', tweet)
    tweet = re.sub('(\.)+', '.', tweet)
    # tweet = re.sub('((www\.[^\s]+)|(https://[^\s]+) | (http://[^\s]+))','URL',tweet)
    tweet = re.sub('((www\.[^\s]+))', '', tweet)
    tweet = re.sub('((http://[^\s]+))', '', tweet)
    tweet = re.sub('((https://[^\s]+))', '', tweet)
    tweet = re.sub('@[^\s]+', '', tweet)
    tweet = re.sub('[\s]+', ' ', tweet)
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    tweet = re.sub('_', '', tweet)
    tweet = re.sub('\$', '', tweet)
    tweet = re.sub('%', '', tweet)
    tweet = re.sub('^', '', tweet)
    tweet = re.sub('&', '', tweet)
    tweet = re.sub('\*', '', tweet)
    tweet = re.sub('\(', '', tweet)
    tweet = re.sub('\)', '', tweet)
    tweet = re.sub('-', '', tweet)
    tweet = re.sub('\+', '', tweet)
    tweet = re.sub('=', '', tweet)
    tweet = re.sub('"', '', tweet)
    tweet = re.sub('~', '', tweet)
    tweet = re.sub('`', '', tweet)
    tweet = re.sub('!', '', tweet)
    tweet = re.sub(':', '', tweet)
    tweet = re.sub('^-?[0-9]+$', '', tweet)
    tweet = tweet.strip('\'"')
    return tweet
    # end

    p = codecs.open('data/twitterSemEval2013.csv', encoding='latin-1')
    read = csv.reader(p)
    outputFile = open('allNewProcessed.csv', 'w', newline='')
    outputWriter = csv.writer(outputFile)
    for row in read:
        processedTweet = cleanTweet(row[2])
        outputWriter.writerow([processedTweet])

    outputFile.close();
    p.close();

def pretrainedEmbeddings():
    embedding_index = {}
    f = open(Embedding_dir)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:])
        embedding_index[word] = coefs
    f.close()
    return embedding_index

def classify(text):
    classifierModel = load_model(savedModelPath)

    embedding_index = pretrainedEmbeddings()
    max_no_words = len(embedding_index.keys())
    vectorizerTrainDocList = []

    #vectorizerTrainDocList.append(next(iter(embedding_index.keys())))
    for word in embedding_index.keys():
        vectorizerTrainDocList.append(word)

    vectorizer = CountVectorizer(max_features=max_no_words,stop_words=None, binary=True)
    vectorizer.fit(vectorizerTrainDocList) #vectorizer train docs are all the words in the pretrained embedding
    sparse_test = vectorizer.transform([text])

    Xj = [row.indices for row in sparse_test]

    X_test = np.array(Xj)

    X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
    #print('X_test shape:', X_test.shape)
    #print(len(X_test), 'test sequences')

    y_test_predict = classifierModel.predict_classes(X_test)

    if(y_test_predict==1):
            print("positive")
    else:
            print("negative")

def split_sentences(text):
    """
    Utility function to return a list of sentences.
    @param text The text that must be split in to sentences.
    """
    sentence_delimiters = re.compile(u'[.!?,;:\t\\\\"\\(\\)\\\'\u2019\u2013]|\\s\\-\\s')
    sentences = sentence_delimiters.split(text)
    return sentences



def run(text):
    sentence_list = split_sentences(cleanTweet(text))
    label = "non-suggestion"
    for sentence in sentence_list:
        label_temp = classify(sentence)
        if label_temp=="suggestion":
            label = "suggestion"

    return label


@app.route('/<inputSentence>', methods = ['GET'])
def test(inputSentence):
    text = inputSentence
    sentenceList = split_sentences(text)
    return run(text)

if __name__ == '__main__':
    app.run(debug = True)
