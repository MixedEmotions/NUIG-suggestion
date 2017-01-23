from __future__ import print_function
from flask import Flask, jsonify, request
import numpy as np
import os
import codecs
import csv
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer
os.environ['KERAS_BACKEND']='theano'
from keras.preprocessing import sequence
from keras.models import load_model
app = Flask(__name__)


# set parameters:
Root_dir = os.path.dirname(os.path.abspath(__file__))
Embedding_dir = Root_dir + "/Embeddings/glove.twitter.27B.50d.txt"
savedModelPath = Root_dir + "/savedModels/CNN_twitterSugg_model.h5"
max_no_words = 27000000000 #fixed, should be same as the number of words in embeddings
maxlen= 40
np.random.seed(1337)  # for reproducibility

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
            return("suggestion")
    else:
            return("non-suggestion")

def split_sentences(text):
    """
    Utility function to return a list of sentences.
    @param text The text that must be split in to sentences.
    """
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = tokenizer.tokenize(text)
    return sentences

@app.route('/<text>', methods = ['GET'])
def run(text):
    sentence_list = split_sentences(cleanTweet(text))
    label = "non-suggestion"
    for sentence in sentence_list:
        label_temp = classify(sentence)
        if label_temp=="suggestion":
            label = "suggestion"

    return label

if __name__ == '__main__':
    app.run(debug = True, host='0.0.0.0')
