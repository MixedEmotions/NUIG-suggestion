
# coding: utf-8

# In[ ]:



from __future__ import division, print_function
import logging
import numpy as np
import os, codecs, csv, re, nltk
import xml.etree.ElementTree as ET

from senpy.plugins import SenpyPlugin, SentimentPlugin
from senpy.models import Results, Entry, Suggestion

logger = logging.getLogger(__name__)

# _ 
import math, itertools

from sklearn.feature_extraction.text import CountVectorizer
os.environ['KERAS_BACKEND']='theano'
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.models import load_model

from datetime import datetime 


# In[ ]:

class SuggestionMiningDL(SentimentPlugin):
    
    def __init__(self, info, *args, **kwargs):
        super(SuggestionMiningDL, self).__init__(info, *args, **kwargs)
        self.name = info['name']
        self.id = info['module']
        self._info = info
        self.async = info['async']
        local_path=os.path.dirname(os.path.abspath(__file__))
        
        self.Embedding_dir = local_path + "/Embeddings/glove.6B.200d.txt"
        self.savedModelPath = local_path + "/savedModels/glove200_trainable_WikitipsAndWikipediaBalanced.h5"

        self.max_no_words = 300000 #fixed, should be same as the number of words in embeddings
        self.maxlen = 40
        
               
        

    def activate(self, *args, **kwargs):
                
        np.random.seed(1337)  # for reproducibility
        
        st = datetime.now()
        self._classifierModel = load_model(self.savedModelPath)       
        logger.info("{} {}".format(datetime.now() - st, "loaded _classifierModel"))
        
        st = datetime.now()
        self._tokenizer = self.get_tokenizer()
        logger.info("{} {}".format(datetime.now() - st, "loaded _tokenizer"))
        
        #st = datetime.now()
        #nltk.download()
        #self._tokenizer_nltk = nltk.data.load('tokenizers/punkt/english.pickle')
        #logger.info("{} {}".format(datetime.now() - st, "loaded _tokenizer_nltk"))
        
        logger.info("SuggestionMiningDL plugin is ready to go!")
        
    def deactivate(self, *args, **kwargs):
        try:
            logger.info("SuggestionMiningDL plugin is being deactivated...")
        except Exception:
            print("Exception in logger while reporting deactivation of SuggestionMiningDL")

    # MY FUNCTIONS
    def cleanTweet(self, tweet):
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
        tweet = tweet.strip('\'"').replace('.','').strip()
        return tweet        
    

    def pretrainedEmbeddings(self, EmbeddingPath):
        embedding_index = {}
        f = open(EmbeddingPath)
        next(iter(f))
        embedding_wordsList = []
        for line in f:
            values = line.split(" ")
            word = values[0]
            coefs = np.asarray(values[1:])
            embedding_index[word] = coefs
            embedding_wordsList.append(word)
        f.close()
        return (embedding_index, embedding_wordsList)

    def get_tokenizer(self):
        st = datetime.now()
        self._embedding_index, self._embedding_wordsList = self.pretrainedEmbeddings(self.Embedding_dir)
        logger.info("{} {}".format(datetime.now() - st, "loaded WordEmbeddings"))

        #self.max_no_words = len(self._embedding_index.keys())# !? max_no_words already defined above

        tokenizer = Tokenizer(self.max_no_words)
        tokenizer.fit_on_texts(self._embedding_wordsList)
        return tokenizer


    def convert_text_to_vector(self, text, tokenizer):
        #st = datetime.now()
        test_sequences = self._tokenizer.texts_to_sequences(text)
        #logger.info("{} {}".format(datetime.now() - st, "test_sequences"))
        
        #st = datetime.now()
        X_test = sequence.pad_sequences(test_sequences, maxlen=self.maxlen)
        #logger.info("{} {}".format(datetime.now() - st, "X_test"))
        return X_test
    
    def classify(self, X_test):    
        #st = datetime.now()
        y_test_predict = self._classifierModel.predict_classes(X_test)
        #logger.info("{} {}".format(datetime.now() - st, "y_test_predict"))
        
        #print(y_test_predict)
        return y_test_predict
    
        """
        Utility function to return a list of sentences.
        @param text The text that must be split in to sentences.
        """        
        #sentences = self._tokenizer_nltk.tokenize(text)
        #return sentences


    def analyse(self, **params):
        logger.debug("SuggestionMiningDL Analysing with params {}".format(params))

        text_input = params.get("input", None)

        st = datetime.now()
        text_sentences = self.cleanTweet(text_input) #[self.cleanTweet(sentence) for sentence in self.split_into_sentences(text_input)] 
        
        #X_test = [ self.convert_text_to_vector(sentence, self._tokenizer) for sentence in text_sentences]
        X_test = self.convert_text_to_vector([text_sentences], self._tokenizer)
        
        y_pred = self.classify(X_test)
        logger.info("{} {}".format(datetime.now() - st, "tweet analysed, predicted: "+str(y_pred)))

        response = Results()
        
        entry = Entry()
        entry.nif__isString = text_input
           
        _mapping_labels = {0:False, 1:True}
        
        for sentence, y_i in zip([text_sentences], y_pred):
            suggestion = Suggestion()            
            suggestion['hasSuggestion'] = _mapping_labels[y_i]
            suggestion["nif:beginIndex"] = 0
            suggestion["nif:endIndex"] = len(sentence)
            suggestion["nif:anchorOf"] = sentence
            entry.suggestions.append(suggestion)
        
        
        response.entries.append(entry)
            
        return response

