import math

SPAM = 'spam'
HAM = 'ham'
WRONG = 'wrong'
RIGHT = 'right'

'''
Machine learning NaiveBayes classifier
'''
class NaiveBayesClassifier:

    def __init__(self):
        self.PriorH = 0.0
        self.PriorS = 0.0
        self.vocabulary = {}
        self.result = {}
        self.SPAM_Positive = 0
        self.SPAM_Negative = 0
        self.HAM_Positive = 0
        self.HAM_Negative = 0
    
    def getPriorHam(self):
        return self.PriorH
    
    def setPriorHam(self, totalDocuments, hamDocuments):
        self.PriorH = math.log10(hamDocuments / totalDocuments)

    def getPriorSpam(self):
        return self.PriorS
    
    def setPriorSpam(self, totalDocuments, spamDocuments):
        self.PriorS = math.log10(spamDocuments / totalDocuments)

    '''
    Method to calculate confusion matrix's variable
    '''
    def setConfusionMatrixVar(self, Target, Predicted):
        if Target == SPAM and Predicted == SPAM:
            self.SPAM_Positive += 1
        elif Target == SPAM and Predicted == HAM:
            self.SPAM_Negative += 1
        elif Target == HAM and Predicted == HAM:
            self.HAM_Positive += 1
        elif Target == HAM and Predicted == SPAM:
            self.HAM_Negative += 1

    '''
    fit the vocabulary for the model
    '''
    def fit(self, vocabulary):
        self.vocabulary = vocabulary
    
    '''
    Returns the classification result
    '''
    def getClassificationResult(self):
        return self.result
    
    '''
    Add classification result as,
    
    {
        test-ham-00001.txt: [ham, 0.004, 0.001, ham, right]
        test-ham-00002.txt: [spam, 0.002, 0.03, ham, wrong] 
    }

    '''
    def addClassificationResult(self, document, predictedClass, scoreHam, scoreSpam, actualClass, label):
        self.result[document] = [predictedClass, scoreHam, scoreSpam, actualClass, label]

    
    '''
    Method that predicts class for given document
    '''
    def predict(self, document, actualClass, words):
        scoreHam = self.getPriorHam()
        scoreSpam = self.getPriorSpam()
        predictedClass = ''
        label = ''

        for word in words:
            if word in self.vocabulary:
                hamProb = self.vocabulary[word][1]
                spamProb = self.vocabulary[word][3]
                scoreHam += math.log10(hamProb)
                scoreSpam += math.log10(spamProb)
        
        if scoreHam > scoreSpam:
            predictedClass = HAM
        else:
            predictedClass = SPAM
        
        if predictedClass == actualClass:
            label = RIGHT
        else:
            label = WRONG
        
        self.addClassificationResult(document, predictedClass, scoreHam, scoreSpam, actualClass, label)
        self.setConfusionMatrixVar(actualClass, predictedClass)

    '''
    method to calculate model's accuracy
    '''
    def getAccuracy(self):
        total = self.SPAM_Positive + self.HAM_Positive + self.SPAM_Negative + self.HAM_Negative
        accuracy = (self.SPAM_Positive + self.HAM_Positive) / total
        return accuracy

    '''
    method to calculate model's precison
    '''
    def getPrecision(self):
        precision = self.SPAM_Positive / (self.SPAM_Positive + self.HAM_Positive)
        return precision
    
    '''
    method to calculate model's recall
    '''
    def getRecall(self):
        recall = self.SPAM_Positive / (self.SPAM_Positive + self.HAM_Negative)
        return recall

    '''
    method to calculate model's f1-measure
    '''
    def getF1Measure(self):
        precision = self.getPrecision()
        recall = self.getRecall()
        f1Mesaure = 2 * ( (precision  * recall) / (precision  + recall) )
        return f1Mesaure
    
    '''
    method to print confusion matrix
    '''
    def printConfusionMatrix(self):
        SP = str(self.SPAM_Positive)
        SN = str(self.SPAM_Negative)
        HP = str(self.HAM_Positive)
        HN = str(self.HAM_Negative)
        
        message = (
            
            "          +------------+-----------+" + "\n" +
            "          |    SPAM    |    HAM    |" + "\n" +
            "+---------+------------+-----------+" + "\n" +
            "|  SPAM   |    "+SP+"     |     "+SN+"    |" + "\n" +
            "+---------+------------+-----------+" + "\n" +
            "|  HAM    |      "+HN+"     |    "+HP+"    |" + "\n" +
            "+---------+------------+-----------+" + "\n" 
        )
        print("          CONFUSION_MATRIX         "+ "\n")
        print(message)
        



