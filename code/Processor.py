import numpy as np
import re
import math

class TextProcessor:

    def __init__(self):
        self.word_frequency = {}
        self.words_Ham = {}
        self.words_Spam = {}
        self.vocabulary = {}
        self.Delta = 0.5
        self.sizeOfHam = 0
        self.sizeOfSpam = 0
        self.sizeOfCorpus = 0
    
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
    returns smoothed conditional probability of words against classType
    where classType: (spam|ham)
    smoothed conditional Probabilty = P(word | classType)

                                                (frequency of word in class) + delta
    where; P(word | class) = ____________________________________________________________________________
                                (total number of words in class) + delta * (size of vocabulary or corpus)
    '''
    def calculateCondProb(self, word, classType):
        freqWord = 0
        sizeOfCorpus = self.sizeOfCorpus

        if(classType == 'ham'):
            freqWord = self.words_Ham[word]
            totalNoOfWords = self.sizeOfHam
        else:
            freqWord = self.words_Spam[word]
            totalNoOfWords = self.sizeOfSpam
        
        prob = ( (freqWord + self.Delta) / (totalNoOfWords + (self.Delta * sizeOfCorpus)) )
        return prob
    
    '''
    Returns the frequency of word in all documents
    
    e.g word_frequency = {word : count}
    '''
    def getWordFrequency(self):
        return self.word_frequency
    
    '''
    Returns vocabulary created from Train data
    '''
    def getVocabulary(self):
        return self.vocabulary
    
    '''
    Add Word and it's result in Vocabulary
    
    where;
    word = unique word
    result =  [3, 0.003, 40, 0.4]
            where; result is a list
                index 0: The frequency of word in the class ham
                index 1: The smoothed conditional probability of word in the class ham, P(word | ham)
                index 2: The frequency of word in the class spam
                index 3: The smoothed conditional probability of word in spam, P(word | spam)
    '''
    def setVocabulary(self, word, result):
        self.vocabulary[word] = result

    def setFreqHam(self, word, value):
        self.vocabulary.get(word, "")[0] = value

    def setFreqSpam(self, word, value):
        self.vocabulary.get(word, "")[2] = value
        pass

    def setConditinalProbHam(self, word, value):
        self.vocabulary.get(word, "")[1] = value

    def setConditinalProbSpam(self, word, value):
        self.vocabulary.get(word, "")[3] = value
    
    '''
    build sorted vocabulary from created word dictionaries
    '''
    def buildVocabulary(self):
        self.sizeOfCorpus = len(self.word_frequency)
        self.sizeOfHam = len(self.words_Ham)
        self.sizeOfSpam = len(self.words_Spam)

        sortedCorpus = sorted(self.word_frequency.keys(), key=lambda x:x.lower())

        for key in sortedCorpus:
            value = self.word_frequency[key]
            self.setVocabulary(key, [0, 0.0, 0, 0.0])

            if key in self.words_Ham:
                self.setFreqHam(key, self.words_Ham[key])
                probability = self.calculateCondProb(key, 'ham')
                self.setConditinalProbHam(key, probability)
                
            if key in self.words_Spam:
                self.setFreqSpam(key, self.words_Spam[key])
                probability = self.calculateCondProb(key, 'spam')
                self.setConditinalProbSpam(key, probability)
    
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
    

    
