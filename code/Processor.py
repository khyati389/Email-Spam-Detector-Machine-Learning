import numpy as np
import re
import math

class TextProcessor:

    def __init__(self):
        self.word_frequency = {}
        self.words_Ham = {}
        self.words_Spam = {}
        self.vocabulary = {}
    
    '''
    Tokenize the given text into words
    returns: the array of words
    '''
    def tokenize(self, text):
        return re.split('[^a-zA-Z]', text)

    '''
    Lower the each word and counts the word frequency in 
    all documents and store it in word_frequency

    e.g {word: count}
    '''
    def recordWordCount(self, words):
        for word in words:
            if word != '':
                if word.lower() in self.word_frequency:
                    self.word_frequency[word.lower()] += 1
                else:
                    self.word_frequency[word.lower()] = 1
    
    '''
    Calculate Frequency Of Word in a class
    where classType: (spam|ham)
    '''
    def updatefrequencyCountInClass(self, classType, words):
        for word in words:
            if word != '':
                if(classType == 'ham'):
                    if word.lower() in self.words_Ham:
                        self.words_Ham[word.lower()] += 1
                    else:
                        self.words_Ham[word.lower()] = 1
                        
                if(classType == 'spam'):
                    if word.lower() in self.words_Spam:
                        self.words_Spam[word.lower()] += 1
                    else:
                        self.words_Spam[word.lower()] = 1
    
    '''
    Function to calculate smoothed conditional probability of words against classType
    where classType: (spam|ham)
    smoothed conditional Probabilty = P(word | classType)
    '''
    def calculateCondProb(self, classType):
        pass
    
    '''
    Returns the frequency of word in all documents
    
    e.g word_frequency = {word : count}
    '''
    def getWordFrequency(self):
        return self.word_frequency
    
    '''
    Returns the frequency of word in class Ham
    
    e.g words_Ham = {word : count}
    '''
    def getWordsHam(self):
        return self.words_Ham
    
    '''
    Returns the frequency of word in class Spam
    
    e.g words_Spam = {word : count}
    '''
    def getWordsSpam(self):
        return self.words_Spam
    
    '''
    Returns vocabulary created from Train data
    '''
    def getVocabulary(self):
        return self.vocabulary
    
    '''
    Add Word and it's result in Vocabulary
    
    where;
    word = unique word
    result =  3 0.003 40 0.4
            where; result is a tuple
                index 0: The frequency of word in the class ham
                index 1: The smoothed conditional probability of word in the class ham, P(word | ham)
                index 2: The frequency of word in the class spam
                index 3: The smoothed conditional probability of word in spam, P(word | spam)
    '''
    def setVocabulary(self, word, result):
        self.vocabulary[word] = result

    
